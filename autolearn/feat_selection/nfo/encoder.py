"""
GAT-based encoder for the NFO controller.

Replaces the previous LSTM(+LeNet) encoder. Operates on the feature tree as
a graph (reconstructed from the post-order token sequence at forward time)
and produces:

    sentence_emb : (B, sentence_emb_size)    unit-norm, fed to predictor
    node_emb     : (B, T_b, hidden)          per-node hidden states, padded
                                              to the max node count in the
                                              batch (T_b varies per batch)
    node_mask    : (B, T_b)                  True for real nodes

The downstream decoder attends over (node_emb, node_mask). LSBO/INFER/VI
keep their interface: perturb ``node_emb`` and call ``extract_feature`` to
recover a sentence embedding.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch

from autolearn.feat_selection.nfo.graph_utils import seqs_to_pyg_batch


class Encoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            seq_length,
            padding_idx,
            arity_table,
            sentence_emb_size=None,
            embedding_size=512,
            gat_layers=3,
            gat_heads=4,
            dropout=0.5,
            vae=False,
            # kept for backward-compat with the old __init__ kwargs; ignored
            lstm_layers=None,
    ):
        super(Encoder, self).__init__()
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.embedding_size = embedding_size
        self.sentence_emb_size = embedding_size if sentence_emb_size is None else sentence_emb_size
        self.vae = bool(vae)

        # arity_table is a numpy array, not a learnable tensor. Stash as a
        # plain attribute so .to(device) doesn't try to move it.
        self.arity_table = np.asarray(arity_table, dtype=np.int64)

        self.embedding = nn.Embedding(
            vocab_size + 1,
            embedding_size,
            padding_idx=padding_idx,
        )
        self.emb_dropout = nn.Dropout(dropout)

        # Stack of GATConv layers. Each intermediate layer uses ``gat_heads``
        # heads with embedding_size/gat_heads channels per head and concats
        # back to embedding_size. The last layer averages across heads
        # (concat=False) to stay at embedding_size.
        assert embedding_size % gat_heads == 0, (
            f"embedding_size ({embedding_size}) must be divisible by gat_heads ({gat_heads})"
        )
        per_head = embedding_size // gat_heads
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        for i in range(gat_layers):
            is_last = (i == gat_layers - 1)
            if is_last:
                conv = GATConv(
                    embedding_size, embedding_size,
                    heads=gat_heads, concat=False, dropout=dropout,
                )
            else:
                conv = GATConv(
                    embedding_size, per_head,
                    heads=gat_heads, concat=True, dropout=dropout,
                )
            self.gat_layers.append(conv)
            self.gat_norms.append(nn.LayerNorm(embedding_size))

        # Pooled-graph -> sentence_emb projection. Mean-pool over the
        # variable-length node set, then project to the sentence space.
        # In VAE mode we expose mu/logvar heads instead of a single
        # deterministic projection; ``extract_feature`` still returns the
        # deterministic mu so downstream consumers (predictor, GP, LSBO
        # ascent) work unchanged. The reparameterized sample z is exposed
        # via ``encode_with_kl`` for training.
        self.sentence_proj = nn.Linear(embedding_size, self.sentence_emb_size)
        if self.vae:
            self.logvar_head = nn.Linear(embedding_size, self.sentence_emb_size)

    def flatten_parameters(self):
        # no-op: kept for API compat with the old LSTM encoder
        return

    def _node_features(self, batch):
        """Run the GAT stack on a PyG Batch and return per-node hidden states."""
        h = self.embedding(batch.x)
        h = self.emb_dropout(h)
        edge_index = batch.edge_index
        for i, (conv, norm) in enumerate(zip(self.gat_layers, self.gat_norms)):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            if i < len(self.gat_layers) - 1:
                h_new = F.leaky_relu(h_new)
                # residual when shapes match (always, since all layers
                # output embedding_size)
                h = h + h_new
            else:
                h = h_new
        return h

    def forward(self, x, sentence_lens=None):
        """
        x:             (B, T) long token tensor (post-order, PAD-padded)
        sentence_lens: ignored — kept for backward compat with the old
                       LSTM encoder's signature. Node masks come from the
                       PyG batch.

        Returns:
            sentence_emb : (B, sentence_emb_size)  L2-normalized.
                           In VAE training mode this is the reparameterized
                           sample z (decoder/predictor see stochastic
                           inputs); otherwise it's the deterministic mu.
            node_emb     : (B, T_b, embedding_size) padded to per-batch
                                                    max node count
            node_mask    : (B, T_b) bool, True for real nodes

        Side effect: ``self.last_kl`` is set to the per-sample KL divergence
        against N(0, I) for VAE mode, or zeros otherwise. multi_train reads
        this after the forward to fold KL into the loss.
        """
        del sentence_lens  # unused
        device = x.device
        batch = seqs_to_pyg_batch(x, self.arity_table, self.padding_idx, device)
        h_nodes = self._node_features(batch)
        node_emb, node_mask = to_dense_batch(h_nodes, batch.batch)
        node_emb = F.normalize(node_emb, p=2, dim=-1)
        z, _mu, _logvar, kl = self.encode_with_kl(node_emb, node_mask)
        self.last_kl = kl
        return z, node_emb, node_mask

    def _pool(self, node_emb, node_mask):
        mask_f = node_mask.unsqueeze(-1).float()
        return (node_emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

    def extract_feature(self, node_emb, node_mask):
        """Masked mean-pool over node embeddings, project, L2-normalize.

        node_emb:  (B, T_b, H) — may be a perturbed version of the original
                   outputs (LSBO / INFER / crossover do this).
        node_mask: (B, T_b)    — required. Without it, padded slots would
                   silently pull the mean down.

        In VAE mode this still returns the deterministic mu (after L2-norm)
        so the predictor / GP / LSBO ascent see a fixed point. Use
        ``encode_with_kl`` if you want the stochastic z + KL term.
        """
        pooled = self._pool(node_emb, node_mask)
        mu = self.sentence_proj(pooled)
        return F.normalize(mu, p=2, dim=-1)

    def encode_with_kl(self, node_emb, node_mask):
        """VAE pass: returns (z, mu, logvar, kl_per_sample).

        z is the reparameterized sample (used for decoder/predictor at
        training time); mu is the deterministic projection (used at
        inference and by LSBO/INFER). kl is computed against N(0, I).
        Both z and mu are L2-normalized for downstream compatibility.

        When self.vae is False, falls back to (mu, mu, zeros, zeros) so
        callers can be agnostic to the mode.
        """
        pooled = self._pool(node_emb, node_mask)
        mu = self.sentence_proj(pooled)
        if not self.vae:
            mu_n = F.normalize(mu, p=2, dim=-1)
            zero = torch.zeros_like(mu_n)
            kl = torch.zeros(mu_n.size(0), device=mu_n.device)
            return mu_n, mu_n, zero, kl
        logvar = self.logvar_head(pooled).clamp(min=-8.0, max=8.0)
        std = torch.exp(0.5 * logvar)
        if self.training:
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        # KL(N(mu, sigma^2) || N(0, I)) per-sample, summed over dims.
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
        z = F.normalize(z, p=2, dim=-1)
        mu_n = F.normalize(mu, p=2, dim=-1)
        return z, mu_n, logvar, kl
