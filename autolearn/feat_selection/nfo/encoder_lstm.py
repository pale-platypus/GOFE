"""Original DIFER LSTM(+LeNet) encoder, restored from commit b88bcc5.

Interface matches the GAT encoder so the controller, decoder, predictor and
INFER loop are agnostic to which encoder is in use:

    forward(x, sentence_lens=None) -> (sentence_emb, encoder_outputs, src_mask)
    extract_feature(node_emb, node_mask) -> sentence_emb

``encoder_outputs`` is ``(B, seq_length, H)`` (per-timestep LSTM hidden) and
``src_mask`` is ``x != padding_idx``. ``extract_feature`` preserves the original
DIFER behavior (sum over time, no mask) — passing a mask is accepted for API
parity but ignored, since INFER/LSBO want bitwise reproduction of the original
pooling.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class _Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Encoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            seq_length,
            padding_idx,
            sentence_emb_size=None,
            embedding_size=512,
            lstm_layers=1,
            dropout=0.5,
            # accepted for backward-compat with the GAT signature; ignored.
            arity_table=None,
            gat_layers=None,
            gat_heads=None,
    ):
        del arity_table, gat_layers, gat_heads  # not used by the LSTM encoder
        super().__init__()
        self.seq_length = seq_length
        self.padding_idx = padding_idx
        self.embedding_size = embedding_size
        self.sentence_emb_size = embedding_size if sentence_emb_size is None else sentence_emb_size

        self.embedding = nn.Embedding(
            vocab_size + 1,
            embedding_size,
            padding_idx=padding_idx,
        )
        self.emb_dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            embedding_size, embedding_size, lstm_layers,
            batch_first=True,
        )

        def tmp_size(size):
            return int((int((size - 2) / 2) + 1 - 2) / 2) + 1

        self.extractor = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5), padding=(2, 2))),
            ('relu1', nn.LeakyReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5), padding=(2, 2))),
            ('relu2', nn.LeakyReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(16, 12, kernel_size=(5, 5), padding=(2, 2))),
            ('relu3', nn.LeakyReLU()),
            ('flatten', _Flatten()),
            ('linear', nn.Linear(
                12 * tmp_size(seq_length) * tmp_size(embedding_size),
                self.sentence_emb_size,
            )),
        ]))

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, sentence_lens=None):
        embedded = self.embedding(x)
        embedded = self.emb_dropout(embedded)
        if sentence_lens is not None:
            embedded = pack_padded_sequence(
                embedded, sentence_lens, batch_first=True, enforce_sorted=False,
            )
        out, (_, _) = self.rnn(embedded)
        if sentence_lens is not None:
            out, _ = pad_packed_sequence(
                out, batch_first=True, total_length=self.seq_length,
            )
        out = F.normalize(out, p=2, dim=-1)
        src_mask = x != self.padding_idx
        sentence_emb = self.extract_feature(out, src_mask)
        return sentence_emb, out, src_mask

    def extract_feature(self, encoder_outputs, src_mask=None):
        """Sum over time + L2-normalize.

        ``src_mask`` is accepted for API parity with the GAT encoder but
        ignored — the original DIFER summed across the full padded sequence,
        and reproducing that summation matters for INFER's gradient ascent
        (it walks the same surface the predictor was trained on).
        """
        del src_mask
        pooled = encoder_outputs.sum(dim=1)
        pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled
