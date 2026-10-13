import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from autolearn.utils import log
from autolearn.utils.torch_attention import Attention


class Decoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            seq_length,
            padding_idx,
            start_idx,
            lstm_layers=1,
            hidden_size=64,
            dropout=0.5
    ):
        super(Decoder, self).__init__()
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.start_idx = start_idx
        self.rnn = nn.LSTM(
            hidden_size, hidden_size, lstm_layers,
            batch_first=True
        )
        self.embedding = nn.Embedding(
            vocab_size + 1,
            hidden_size,
            padding_idx=padding_idx
        )
        self.emb_dropout = nn.Dropout(dropout)
        self.attention = Attention(self.hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward_step(self, x, hidden, attn):
        out, hidden = self.rnn(x, hidden)
        k, v, src_mask, valid_length = attn
        out, attn = self.attention(out, k)
        predicted_softmax = F.log_softmax(
            self.out(out), dim=-1
        )
        return predicted_softmax, hidden

    def forward(
            self,
            emb,
            encoder_outputs,
            src_mask,
            target_attn_mask,
            target_variable=None,
    ):
        """
        emb:              (B, H)    pooled sentence embedding, used as the
                                    initial (h0, c0) of the LSTM.
        encoder_outputs:  (B, S, H) per-node embeddings from the GNN
                                    encoder (S = max nodes in this batch).
        src_mask:         (B, S)    True for real nodes; gates the attention
                                    over encoder_outputs.
        target_attn_mask: (B, L)    True for non-PAD target token positions,
                                    where L == self.seq_length. Used to zero
                                    the loss past the real target length.
        target_variable:  (B, L) long or None for free-running decoding.
        """
        self.attention.set_mask(~src_mask.unsqueeze(1))
        emb = emb.unsqueeze(0)
        batch_size = encoder_outputs.size(0)
        decoder_hidden = (emb, emb)
        valid_length = target_attn_mask.sum(dim=-1)
        decoder_outputs = []
        sentences = []
        log(f"""
            emb {emb.size()}\n{emb}
            encoder_output {encoder_outputs.size()}:\n{encoder_outputs}
            src_mask {src_mask.size()}
        """, level="debug")

        decoder_input = torch.tensor(
            [self.start_idx] * batch_size, dtype=torch.long, device=emb.device
        ).view(-1)
        for i in range(-1, self.seq_length - 1):
            if target_variable is not None and i >= 0:
                decoder_input = target_variable[:, i]
            decoder_input = self.embedding(decoder_input).unsqueeze(1)
            decoder_input = self.emb_dropout(decoder_input)
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input,
                decoder_hidden,
                (encoder_outputs, encoder_outputs, src_mask, valid_length)
            )
            decoder_outputs.append(decoder_output)
            sentence = decoder_output.squeeze(1).argmax(dim=-1).detach().cpu().numpy()
            sentences.append(sentence)
            if target_variable is None:
                decoder_input = torch.tensor(
                    sentence, dtype=torch.long, device=emb.device
                ).view(-1)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        sentences = np.vstack(sentences).transpose(1, 0)
        log(f"decode output {decoder_outputs.size()}:\n{decoder_outputs}", level="debug")
        if self.training:
            # Include the position immediately after the last real token as
            # supervised (it predicts the first PAD, i.e. end-of-sequence),
            # then zero the rest of the loss past that.
            loss_mask = target_attn_mask.clone()
            for b in range(loss_mask.size(0)):
                pos = int(valid_length[b].item())
                if pos < loss_mask.size(1):
                    loss_mask[b, pos] = True
            decoder_outputs[~loss_mask] *= 0
        return decoder_outputs, sentences


def main():
    from feat_tree import random_generate_tree
    from tokenizer import Tokenizer
    from collections import defaultdict
    from controller import NFOController, Mode
    from autolearn.feat_selection.dataset import SequenceDataset
    from autolearn.utils import torch_train
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    feats = ["a", "b", "c", "d"]
    op_info = [
        ('+', 2, False),
        ('-', 2, True),
        ('*', 2, False),
        ('log', 1, False)
    ]
    ops = [op[0] for op in op_info]
    op_arity = defaultdict(lambda: 0)
    op_order = defaultdict(lambda: False)
    type_dict = defaultdict(lambda: 0)
    max_length = 2 ** 5 - 1
    for op in op_info:
        op_arity[op[0]] = op[1]
        op_order[op[0]] = op[2]
    sentences = [random_generate_tree(feats, ops, op_arity, op_order, type_dict, max_order=4).post_order()[0] for _ in range(20)]
    tokenizer = Tokenizer(
        feats + [op[0] for op in op_info],
        max_length=max_length
    )
    seqs = tokenizer.transform(sentences)
    nfo = NFOController(
        tokenizer.vocab_size,
        tokenizer.max_length,
        tokenizer.padding_idx,
        tokenizer.start_idx,
        op_arity=op_arity,
        tokenizer=tokenizer,
    )
    nfo.set_mode(Mode.ED)
    nfo.train()
    optimizer = torch.optim.Adam(
        nfo.parameters(), lr=0.0005
    )
    translate_set = SequenceDataset(seqs, seqs)
    torch_train(
        translate_set, nfo, optimizer, torch.nn.functional.nll_loss, device,
        patience=50,
        epochs=200, batch_size=32,
    )
    nfo.eval()
    _, decoded_seqs = nfo(torch.tensor(seqs, dtype=torch.long, device=device))
    decoded_feats = tokenizer.inverse_transform(decoded_seqs)
    log(f"original features {sentences}")
    log(f"decode features\nto: {decoded_feats}\nfrom: {decoded_seqs}")


if __name__ == '__main__':
    main()
