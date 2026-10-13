import os
import torch
import numpy as np
import torch.nn as nn
from enum import Enum
from pathlib import Path

from autolearn.feat_selection.nfo.encoder import Encoder as GATEncoder
from autolearn.feat_selection.nfo.encoder_lstm import Encoder as LSTMEncoder
from autolearn.feat_selection.nfo.decoder import Decoder
from autolearn.feat_selection.nfo.predictor import Predictor
from autolearn.feat_selection.nfo.graph_utils import build_arity_table
from autolearn.utils import log


Mode = Enum('Mode', ('ED', 'EP', 'EPD', 'INFER'))


def load_nfo(controller, ckp_path):
    use_iter = None
    if ckp_path is not None and os.path.exists(ckp_path) and len(os.listdir(ckp_path)) > 0:
        ckp_path = Path(ckp_path)
        ckps = [ckp for ckp in list(os.listdir(ckp_path)) if ckp.endswith('.ckp')]
        ckp_metrics = np.asarray([
            int(ele.split('.ckp')[0].split('_')[0]) for ele in ckps
        ])
        best_ckp = ckps[ckp_metrics.argmax()]
        use_iter = int(best_ckp.split('_')[0])
        controller.load_state_dict(torch.load(ckp_path / best_ckp))
        log(f"Use NFO controller checkpoint in  {ckp_path / best_ckp}")
    return controller, use_iter


class NFOController(nn.Module):
    def __init__(
            self,
            vocab_size,
            seq_length,
            padding_idx,
            start_idx,
            arity_table=None,
            tokenizer=None,
            op_arity=None,
            sentence_emb_size=None,
            embedding_size=512,
            lstm_layers=1,
            gat_layers=3,
            gat_heads=4,
            dropout=0.5,
            encoder_kind='gat',
            vae=False,
    ):
        super(NFOController, self).__init__()
        self.encoder_kind = encoder_kind
        self.vae = bool(vae)
        if encoder_kind == 'gat':
            if arity_table is None:
                if tokenizer is None or op_arity is None:
                    raise ValueError(
                        "NFOController(encoder_kind='gat') requires either "
                        "`arity_table` or both `tokenizer` and `op_arity` so the "
                        "GNN encoder can build trees from token sequences."
                    )
                arity_table = build_arity_table(tokenizer, op_arity)
            self.encoder = GATEncoder(
                vocab_size,
                seq_length,
                padding_idx,
                arity_table,
                sentence_emb_size=sentence_emb_size,
                embedding_size=embedding_size,
                gat_layers=gat_layers,
                gat_heads=gat_heads,
                dropout=dropout,
                vae=self.vae,
            )
        elif encoder_kind == 'lstm':
            self.encoder = LSTMEncoder(
                vocab_size,
                seq_length,
                padding_idx,
                sentence_emb_size=sentence_emb_size,
                embedding_size=embedding_size,
                lstm_layers=lstm_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"unknown encoder_kind {encoder_kind!r} (expected 'gat' or 'lstm')"
            )
        self.predictor = Predictor(
            embedding_size
        )
        self.decoder = Decoder(
            vocab_size,
            seq_length,
            padding_idx,
            start_idx,
            lstm_layers=lstm_layers,
            hidden_size=embedding_size
        )
        self.padding_idx = padding_idx
        self.mode = Mode.EPD

        self.flatten_parameters()

    def flatten_parameters(self):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()

    def train(self, mode=True):
        super(NFOController, self).train(mode)
        if self.mode == Mode.EP:
            self.decoder.eval()
        elif self.mode == Mode.ED:
            self.encoder.eval()

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, x, opt_eta=1e-3, score_threshold=None, langevin_tau=0.0):
        target_attn_mask = x != self.padding_idx
        if self.mode == Mode.ED:
            embedding, node_emb, node_mask = self.encode(x)
            embedding = embedding.detach()
            node_emb = node_emb.detach()
            node_mask = node_mask.detach()
            x, symbols = self.decode(embedding, node_emb, node_mask, target_attn_mask, target_var=x)
            if self.training:
                return x
            else:
                return x, symbols
        elif self.mode == Mode.EP:
            embedding, _, _ = self.encode(x)
            scores = self.predict(embedding)
            return scores
        elif self.mode == Mode.EPD:
            embedding, node_emb, node_mask = self.encode(x)
            scores = self.predict(embedding)
            x, symbols = self.decode(embedding, node_emb, node_mask, target_attn_mask, target_var=x)
            if self.training:
                return scores, x
            else:
                return scores, symbols
        else:
            embedding, node_emb, node_mask = self.infer(
                x, opt_eta, score_threshold, langevin_tau
            )
            x, symbols = self.decode(embedding, node_emb, node_mask, target_attn_mask)
            return x, symbols

    def encode(self, x):
        embedding, node_emb, node_mask = self.encoder(x)
        return embedding, node_emb, node_mask

    def predict(self, x):
        x = self.predictor(x)
        return x

    def decode(self, embedding, node_emb, node_mask, target_attn_mask, target_var=None):
        x, symbols = self.decoder(
            embedding, node_emb, node_mask, target_attn_mask, target_variable=target_var,
        )
        return x, symbols

    def infer(self, x, eta, score_threshold, langevin_tau=0.0):
        # Langevin INFER on per-node embeddings:
        # node_emb_{t+1} = node_emb_t + eta * grad(score) + sqrt(2*eta*tau) * noise
        # tau = 0 recovers vanilla DIFER INFER (deterministic gradient ascent).
        embedding, node_emb, node_mask = self.encode(x)
        score = self.predict(embedding)

        cur_score = score.detach().cpu().numpy()
        final_score = cur_score * score_threshold
        new_embeddings = embedding.detach()
        length = len(cur_score)
        noise_scale = float(np.sqrt(2.0 * eta * langevin_tau)) if langevin_tau > 0 else 0.0
        epoch = 0
        while np.sum(cur_score > final_score) < length * 0.5 and epoch < 50:
            opt_outputs = torch.autograd.grad(score, node_emb, torch.ones_like(score))[0]
            update = eta * opt_outputs
            if noise_scale > 0.0:
                update = update + noise_scale * torch.randn_like(node_emb)
            if self.encoder_kind == 'gat':
                # GAT pools with the node mask, so updates on padded slots are
                # silently dropped by extract_feature anyway — zero them here
                # too to keep the buffer clean and consistent across iters.
                update = update * node_mask.unsqueeze(-1).float()
            # LSTM path keeps the original DIFER behavior: update every
            # timestep, including PAD positions, since extract_feature sums
            # over the full sequence without masking.
            node_emb.data = node_emb.data + update
            new_embeddings = self.encoder.extract_feature(node_emb, node_mask)
            score = self.predict(new_embeddings)
            cur_score = score.detach().cpu().numpy()
            epoch += 1

        return new_embeddings, node_emb, node_mask


def main():
    from feat_tree import random_generate_tree
    from tokenizer import Tokenizer
    from collections import defaultdict
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
    sentences = [random_generate_tree(feats, ops, op_arity, op_order, type_dict, max_order=4).post_order()[0] for _ in range(10)]
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
        tokenizer=tokenizer,
        op_arity=op_arity,
    )
    nfo.set_mode(Mode.INFER)
    log_softmax, new_feat = nfo(torch.tensor(seqs, dtype=torch.long), score_threshold=1.3)
    log(new_feat)
    log(tokenizer.inverse_transform(new_feat))


if __name__ == '__main__':
    main()
