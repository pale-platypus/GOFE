"""
Amortized Variational Inference for Feature Engineering (Path 2).

Trains a conditional normalizing flow q_phi(z | c) to approximate the
Boltzmann distribution p*(z | c) ∝ exp(score_theta(z) / tau) over the NFO
encoder embedding space, conditioned on an anchor embedding c drawn from
the top-K of the current feature pool.

Reverse-KL objective (the log Z normalizer is constant in phi and dropped):
    L(phi) = E_{z ~ q_phi(.|c)}[ beta * log q_phi(z|c) - score_theta(z) / tau ]

At inference, samples from q_phi are decoded using each anchor's encoder
context (enc_out, hidden, attn_mask) — analogous to LSBO's anchor-conditioned
decoding, but with a learned amortized sampler instead of GP + EI ascent.

DIFER's INFER corresponds to the tau -> 0, beta -> 0 limit of this method
(deterministic gradient ascent on the predictor).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from autolearn.utils import log


class AffineCoupling(nn.Module):
    """RealNVP-style affine coupling layer with conditioning."""

    def __init__(self, dim, hidden, cond_dim, mask):
        super().__init__()
        self.register_buffer('mask', mask)
        in_dim = dim + cond_dim
        self.s_net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim), nn.Tanh(),
        )
        self.t_net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z, c):
        """Sampling direction: z (base) -> x. Returns (x, log_det(dx/dz))."""
        z_masked = z * self.mask
        h = torch.cat([z_masked, c], dim=-1)
        s = self.s_net(h) * (1 - self.mask)
        t = self.t_net(h) * (1 - self.mask)
        x = z_masked + (1 - self.mask) * (z * torch.exp(s) + t)
        log_det = s.sum(dim=-1)
        return x, log_det


class ConditionalFlow(nn.Module):
    """Stack of affine couplings parametrizing q_phi(z | c) over R^dim."""

    def __init__(self, dim, hidden, n_layers, cond_dim):
        super().__init__()
        self.dim = dim
        layers = []
        half = dim // 2
        for i in range(n_layers):
            mask = torch.zeros(dim)
            if i % 2 == 0:
                mask[:half] = 1.0
            else:
                mask[half:] = 1.0
            layers.append(AffineCoupling(dim, hidden, cond_dim, mask))
        self.layers = nn.ModuleList(layers)

    def sample(self, condition, n_per_cond):
        """
        condition: (B, cond_dim)
        Returns:
            x:     (B * n_per_cond, dim)  reparameterized samples from q_phi(.|c)
            log_q: (B * n_per_cond,)      log q_phi(x | c)
            c_exp: (B * n_per_cond, cond_dim) condition repeated for each sample
        """
        B = condition.size(0)
        device = condition.device
        c_exp = condition.unsqueeze(1).expand(-1, n_per_cond, -1).reshape(B * n_per_cond, -1)
        z = torch.randn(B * n_per_cond, self.dim, device=device)
        log_base = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * self.dim * math.log(2.0 * math.pi)
        x = z
        log_det = torch.zeros(B * n_per_cond, device=device)
        for layer in self.layers:
            x, ld = layer(x, c_exp)
            log_det = log_det + ld
        # change of variables: log q(x|c) = log p_base(z) - log |det dx/dz|
        log_q = log_base - log_det
        return x, log_q, c_exp


def _train_flow(
    flow, nfo, anchor_embs, *,
    tau, beta, n_iters, batch_size, n_per_anchor, lr,
):
    """Reverse-KL training of q_phi against exp(score/tau).

    anchor_embs: (N, dim) detached top-K embeddings from the pool.
    Predictor and encoder parameters are NOT updated (only flow is in the optimizer).
    """
    flow.train()
    nfo.eval()
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    N = anchor_embs.size(0)
    device = anchor_embs.device

    for step in range(n_iters):
        idx = torch.randint(0, N, (batch_size,), device=device)
        c = anchor_embs[idx]
        z, log_q, _ = flow.sample(c, n_per_anchor)
        # Predictor was trained on unit-sphere embeddings; renormalize before evaluating score.
        z_norm = F.normalize(z, p=2, dim=-1)
        score = nfo.predict(z_norm).squeeze(-1)  # (B*n_per,), bounded by tanh

        loss = (beta * log_q - score / tau).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5.0)
        opt.step()

        if (step + 1) % max(1, n_iters // 5) == 0 or step == 0:
            log(f"[VI] step {step + 1}/{n_iters} "
                f"loss={loss.item():.4f} "
                f"E[score]={score.mean().item():+.4f} "
                f"E[log_q]={log_q.mean().item():+.2f}")

    flow.eval()


def amortized_vi_search(
    nfo, tokenizer, device,
    pool_features, pool_scores, padding_idx,
    *, dim=512, hidden=256, n_layers=6,
    tau=0.1, beta=1e-2, n_iters=300, batch_size=32,
    n_per_anchor_train=4, n_per_anchor_inf=8,
    top_k_seed=16, lr=1e-3,
):
    """
    End-to-end: build flow, fit it to current pool, sample, decode.

    Returns a list of decoded feature strings (caller dedups via
    FeatureGenerator.append_new).
    """
    nfo.eval()

    seqs = tokenizer.transform(pool_features)
    seqs_t = torch.tensor(seqs, dtype=torch.long, device=device)
    target_attn_mask_pool = seqs_t != padding_idx
    with torch.no_grad():
        emb_pool, enc_out_pool, node_mask_pool = nfo.encoder(seqs_t)
    scores_t = torch.tensor(pool_scores, dtype=torch.float32, device=device)
    order = torch.argsort(scores_t, descending=True)
    K = min(top_k_seed, order.numel())
    anchor_idx = order[:K]
    anchor_emb = emb_pool[anchor_idx].detach()                # (K, dim) — also used as condition
    anchor_out = enc_out_pool[anchor_idx].detach()            # (K, T, hidden)
    anchor_node_mask = node_mask_pool[anchor_idx]             # (K, T)
    anchor_target_attn = target_attn_mask_pool[anchor_idx]    # (K, seq_len)
    anchor_scores = scores_t[anchor_idx]

    log(f"[VI] pool={len(pool_features)} K={K} anchors "
        f"(anchor score range [{anchor_scores.min().item():.4f}, {anchor_scores.max().item():.4f}]) "
        f"dim={dim} n_layers={n_layers} tau={tau} beta={beta}")

    flow = ConditionalFlow(dim=dim, hidden=hidden, n_layers=n_layers, cond_dim=dim).to(device)
    _train_flow(
        flow, nfo, anchor_emb,
        tau=tau, beta=beta,
        n_iters=n_iters, batch_size=batch_size,
        n_per_anchor=n_per_anchor_train, lr=lr,
    )

    # inference: sample n_per_anchor_inf candidates from each anchor's q_phi(.|c)
    with torch.no_grad():
        z, log_q_inf, _ = flow.sample(anchor_emb, n_per_anchor_inf)
        z_norm = F.normalize(z, p=2, dim=-1)
        pred = nfo.predict(z_norm).squeeze(-1)

        # Pair each sampled embedding with its anchor's decoding context.
        # anchor_out: (K, T, H) -> expand to (K*n_per, T, H)
        T, H = anchor_out.size(1), anchor_out.size(2)
        anchor_out_exp = (anchor_out.unsqueeze(1)
                          .expand(-1, n_per_anchor_inf, -1, -1)
                          .reshape(K * n_per_anchor_inf, T, H))
        anchor_node_mask_exp = (anchor_node_mask.unsqueeze(1)
                                .expand(-1, n_per_anchor_inf, -1)
                                .reshape(K * n_per_anchor_inf, -1))
        anchor_target_attn_exp = (anchor_target_attn.unsqueeze(1)
                                  .expand(-1, n_per_anchor_inf, -1)
                                  .reshape(K * n_per_anchor_inf, -1))
        _, sentences = nfo.decoder(
            z_norm, anchor_out_exp, anchor_node_mask_exp, anchor_target_attn_exp,
        )

    feats = tokenizer.inverse_transform(sentences)

    # diagnostics: how many decoded sequences are identical to their anchor's seq?
    anchor_seqs_np = seqs[anchor_idx.detach().cpu().numpy()]
    n_total = len(feats)
    n_identical = sum(
        1 for i in range(n_total)
        if (sentences[i] == anchor_seqs_np[i // n_per_anchor_inf]).all()
    )
    n_unique = len(set(feats))
    log(f"[VI] decoded {n_total} candidates: "
        f"{n_identical} identical to anchor, {n_unique} unique strings; "
        f"predicted scores [{pred.min().item():.4f}, {pred.max().item():.4f}], "
        f"log_q [{log_q_inf.min().item():.2f}, {log_q_inf.max().item():.2f}]")

    return feats
