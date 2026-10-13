import math

import torch
import torch.nn as nn

from autolearn.utils import log


SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


def _import_gpytorch():
    try:
        import gpytorch
    except ImportError as e:
        raise ImportError(
            "LSBO requires gpytorch. Install it with `pip install gpytorch` "
            "or set `use_lsbo: false` in the NFO config to use the legacy INFER path."
        ) from e
    return gpytorch


def _build_gp(train_x, train_y, gpytorch):
    class _ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, x, y, lik):
            super().__init__(x, y, lik)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=x.size(-1))
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(train_x.device)
    gp = _ExactGPModel(train_x, train_y, likelihood).to(train_x.device)
    return gp, likelihood


def _fit_gp(emb, scores, n_iter, lr, gpytorch):
    gp, likelihood = _build_gp(emb, scores, gpytorch)
    gp.train()
    likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
    optimizer = torch.optim.Adam(gp.parameters(), lr=lr)
    for _ in range(n_iter):
        optimizer.zero_grad()
        output = gp(emb)
        loss = -mll(output, scores)
        loss.backward()
        optimizer.step()
    gp.eval()
    likelihood.eval()
    return gp, likelihood


def _ei(mean, std, S_star, xi):
    std = std.clamp(min=1e-9)
    delta = mean - S_star - xi
    z = delta / std
    cdf = 0.5 * (1.0 + torch.erf(z / SQRT2))
    pdf = torch.exp(-0.5 * z * z) / SQRT2PI
    return delta * cdf + std * pdf


def _project_(delta, r_max):
    flat = delta.flatten(start_dim=1)
    norms = flat.norm(dim=1)
    scale = torch.clamp(r_max / norms.clamp(min=1e-12), max=1.0)
    delta.mul_(scale.view(-1, *([1] * (delta.dim() - 1))))


@torch.no_grad()
def _generate_starts(enc_out_pool, scores, top_k_seed, sigma_noise_grid, starts_per_combo):
    order = torch.argsort(scores, descending=True)
    top_k = min(top_k_seed, order.numel())
    anchor_indices = order[:top_k].tolist()
    starts = []
    for ai in anchor_indices:
        for sigma in sigma_noise_grid:
            for _ in range(starts_per_combo):
                delta = float(sigma) * torch.randn_like(enc_out_pool[ai])
                starts.append((ai, delta))
    return starts


def lsbo_search(
    nfo, tokenizer, device,
    pool_features, pool_scores, padding_idx,
    *, K=16, n_cands=32, n_steps=50, eta=1e-3, xi=0.01,
    r_max=2.0, top_k_seed=8, sigma_noise_grid=(0.05, 0.1, 0.2, 0.4),
    starts_per_combo=3, gp_train_iters=100, gp_lr=0.1, gp_max_train=512,
    s_star_quantile=0.95, gp_train_top_frac=0.5,
    rerank_round_trip=True,
):
    """
    Latent-space Bayesian optimization on the NFO encoder/decoder.

    Pipeline: encode pool -> fit RBF GP on (embedding, score) -> generate M
    perturbed starts around top-k pool anchors -> batched EI prescreen keeps
    top K -> gradient ascent on EI from each, with local Frobenius-norm trust
    region of radius r_max around the anchor -> decode top n_cands.

    Returns a list of decoded feature strings (caller dedups via
    FeatureGenerator.append_new).
    """
    gpytorch = _import_gpytorch()
    nfo.eval()

    seqs = tokenizer.transform(pool_features)
    seqs_t = torch.tensor(seqs, dtype=torch.long, device=device)
    target_attn_mask_pool = seqs_t != padding_idx
    with torch.no_grad():
        emb_pool, enc_out_pool, node_mask_pool = nfo.encoder(seqs_t)
    scores_t = torch.tensor(pool_scores, dtype=torch.float32, device=device)

    if scores_t.numel() > gp_max_train:
        n_top = max(1, int(round(gp_max_train * gp_train_top_frac)))
        n_rand = gp_max_train - n_top
        top_idx = torch.topk(scores_t, n_top).indices
        remaining_mask = torch.ones(scores_t.numel(), dtype=torch.bool, device=scores_t.device)
        remaining_mask[top_idx] = False
        remaining_idx = torch.nonzero(remaining_mask, as_tuple=False).flatten()
        if remaining_idx.numel() > 0 and n_rand > 0:
            n_rand = min(n_rand, remaining_idx.numel())
            perm = torch.randperm(remaining_idx.numel(), device=remaining_idx.device)[:n_rand]
            rand_idx = remaining_idx[perm]
            keep = torch.cat([top_idx, rand_idx])
        else:
            keep = top_idx
        emb_train = emb_pool[keep].contiguous()
        scores_train = scores_t[keep].contiguous()
        gp_top_count = top_idx.numel()
        gp_rand_count = keep.numel() - gp_top_count
    else:
        emb_train = emb_pool.contiguous()
        scores_train = scores_t.contiguous()
        gp_top_count = scores_t.numel()
        gp_rand_count = 0

    log(f"[LSBO] pool={len(pool_features)} gp_train={emb_train.size(0)} "
        f"(top={gp_top_count}, random={gp_rand_count}) "
        f"emb_dim={emb_pool.size(-1)} enc_out_shape={tuple(enc_out_pool.shape[1:])}")

    gp, likelihood = _fit_gp(emb_train, scores_train, gp_train_iters, gp_lr, gpytorch)
    # Use a quantile of the pool scores rather than the max as S*. Two reasons:
    # (1) when scores are MinMax-normalized in feat_tree.FeatureGenerator the pool
    #     max is exactly 1.0 by construction, so EI(S*=1.0) saturates to ~0; (2) EI
    #     against the median-good is a more achievable target than the global max.
    S_star = float(torch.quantile(scores_t, float(s_star_quantile)).item())
    pool_max = float(scores_t.max().item())
    log(f"[LSBO] GP fit done. S*={S_star:.4f} (q={s_star_quantile}, pool_max={pool_max:.4f}) xi={xi}")

    starts = _generate_starts(
        enc_out_pool, scores_t,
        top_k_seed=min(top_k_seed, len(pool_features)),
        sigma_noise_grid=sigma_noise_grid,
        starts_per_combo=starts_per_combo,
    )
    M = len(starts)
    if M == 0:
        log("[LSBO] no starts generated; pool empty?")
        return []

    anchor_idx_M = torch.tensor([s[0] for s in starts], dtype=torch.long, device=device)
    delta_M = torch.stack([s[1] for s in starts], dim=0)
    _project_(delta_M, r_max)

    with torch.no_grad():
        anchor_out_M = enc_out_pool[anchor_idx_M]
        anchor_mask_M = node_mask_pool[anchor_idx_M]
        new_emb_M = nfo.encoder.extract_feature(anchor_out_M + delta_M, anchor_mask_M)
        with gpytorch.settings.fast_pred_var():
            post = likelihood(gp(new_emb_M))
        ei_M = _ei(post.mean, post.variance.clamp(min=1e-12).sqrt(), S_star, xi)

    K = min(K, M)
    top_idx = torch.topk(ei_M, K).indices
    log(f"[LSBO] prescreen kept {K}/{M} starts; "
        f"top-K EI [{ei_M[top_idx].min().item():.4g}, {ei_M[top_idx].max().item():.4g}]")

    anchor_idx_K = anchor_idx_M[top_idx]
    delta_K = nn.Parameter(delta_M[top_idx].detach().clone())
    anchor_out_K = enc_out_pool[anchor_idx_K]
    anchor_mask_K = node_mask_pool[anchor_idx_K]
    optimizer = torch.optim.SGD([delta_K], lr=eta)

    for _ in range(n_steps):
        optimizer.zero_grad()
        new_emb_K = nfo.encoder.extract_feature(anchor_out_K + delta_K, anchor_mask_K)
        with gpytorch.settings.fast_pred_var():
            post = likelihood(gp(new_emb_K))
        std_K = post.variance.clamp(min=1e-12).sqrt()
        ei_K = _ei(post.mean, std_K, S_star, xi)
        loss = -ei_K.sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            _project_(delta_K.data, r_max)

    with torch.no_grad():
        new_out_K = anchor_out_K + delta_K
        new_emb_K = nfo.encoder.extract_feature(new_out_K, anchor_mask_K)
        with gpytorch.settings.fast_pred_var():
            post = likelihood(gp(new_emb_K))
        ei_final = _ei(post.mean, post.variance.clamp(min=1e-12).sqrt(), S_star, xi)

    n_cands = min(n_cands, K)
    final_top = torch.topk(ei_final, n_cands).indices
    log(f"[LSBO] post-ascent EI [{ei_final[final_top].min().item():.4g}, "
        f"{ei_final[final_top].max().item():.4g}]")

    selected_anchor = anchor_idx_K[final_top]
    selected_outputs = (anchor_out_K[final_top] + delta_K[final_top]).detach()
    selected_mask = anchor_mask_K[final_top]
    selected_emb = nfo.encoder.extract_feature(selected_outputs, selected_mask).detach()
    selected_target_attn = target_attn_mask_pool[selected_anchor]

    with torch.no_grad():
        _, sentences = nfo.decoder(
            selected_emb, selected_outputs, selected_mask, selected_target_attn,
        )
    feats = tokenizer.inverse_transform(sentences)

    # Diagnostic: how many decoded sequences differ from their anchor's seq?
    # If most are identical, the trust region is too tight for the decoder to flip tokens.
    anchor_seqs_np = seqs[selected_anchor.detach().cpu().numpy()]
    n_total = len(feats)
    n_identical = sum(
        1 for i in range(n_total)
        if (sentences[i] == anchor_seqs_np[i]).all()
    )
    n_unique = len(set(feats))
    log(f"[LSBO] decoded {n_total} candidates: "
        f"{n_identical} identical to anchor, "
        f"{n_unique} unique strings; "
        f"if most are identical, increase r_max")

    if rerank_round_trip and feats:
        feats = _round_trip_rerank(
            feats, nfo, tokenizer, device, padding_idx,
            gp, likelihood, gpytorch, S_star, xi,
        )
    return feats


def _round_trip_rerank(
    feats, nfo, tokenizer, device, padding_idx,
    gp, likelihood, gpytorch, S_star, xi,
):
    """Re-encode decoded feats and rerank by GP-predicted EI on the round-trip
    embedding, not on the post-ascent latent the decoder may not faithfully
    invert.

    Features that fail to tokenize / encode are kept but pushed to the back
    of the list (so the caller's append_new still sees them, just last).
    """
    import numpy as np

    encodable_idx, encodable_feats = [], []
    for i, f in enumerate(feats):
        if not isinstance(f, str) or not f:
            continue
        try:
            tokenizer.transform([f])
        except Exception:
            continue
        encodable_idx.append(i)
        encodable_feats.append(f)

    if not encodable_feats:
        log("[LSBO] round-trip rerank: 0/%d feats re-tokenizable; "
            "keeping original order" % len(feats))
        return feats

    try:
        seqs_rt = tokenizer.transform(encodable_feats)
        seqs_rt_t = torch.tensor(seqs_rt, dtype=torch.long, device=device)
        with torch.no_grad():
            emb_rt, _, _ = nfo.encoder(seqs_rt_t)
            with gpytorch.settings.fast_pred_var():
                post_rt = likelihood(gp(emb_rt))
            std_rt = post_rt.variance.clamp(min=1e-12).sqrt()
            ei_rt = _ei(post_rt.mean, std_rt, S_star, xi).detach().cpu().numpy()
            mean_rt = post_rt.mean.detach().cpu().numpy()
    except Exception as e:
        log(f"[LSBO] round-trip rerank: encode/GP failed ({e}); "
            f"keeping original order")
        return feats

    order = np.argsort(-ei_rt)  # descending
    reranked = [encodable_feats[j] for j in order]
    # Append the un-encodable feats at the back, preserving their original order.
    leftover = [feats[i] for i in range(len(feats)) if i not in set(encodable_idx)]
    reranked.extend(leftover)

    log(f"[LSBO] round-trip rerank: {len(encodable_feats)}/{len(feats)} "
        f"re-encoded; EI [{ei_rt[order].max():.4g} -> {ei_rt[order].min():.4g}], "
        f"GP mean [{mean_rt[order[0]]:.4f} -> {mean_rt[order[-1]]:.4f}]")
    return reranked
