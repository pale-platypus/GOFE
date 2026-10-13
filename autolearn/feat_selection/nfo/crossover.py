import torch

from autolearn.utils import log


def latent_crossover(
    nfo, tokenizer, device,
    pool_features, pool_scores, padding_idx,
    *, top_k_pairs=8, alphas=(0.25, 0.5, 0.75),
):
    """
    Latent-space crossover for symbolic feature expressions.

    Take the top `top_k_pairs` pool entries by score, generate every unordered
    pair, and for each pair interpolate the encoder_outputs at each alpha:

        encoder_outputs_cross = alpha * enc_out_i + (1 - alpha) * enc_out_j

    Then derive emb_cross via extract_feature, take the union of attention
    masks, and decode each interpolation deterministically. This produces
    candidate expressions that structurally combine two known good ones.

    Number of candidates returned: top_k_pairs * (top_k_pairs - 1) / 2 * len(alphas).
    Caller dedups via FeatureGenerator.append_new.
    """
    nfo.eval()

    seqs = tokenizer.transform(pool_features)
    seqs_t = torch.tensor(seqs, dtype=torch.long, device=device)
    target_attn_mask_pool = seqs_t != padding_idx
    with torch.no_grad():
        emb_pool, enc_out_pool, node_mask_pool = nfo.encoder(seqs_t)
    scores_t = torch.tensor(pool_scores, dtype=torch.float32, device=device)

    n = min(top_k_pairs, len(pool_features))
    if n < 2:
        log(f"[CROSSOVER] pool too small (n={n}); skipping")
        return []

    order = torch.argsort(scores_t, descending=True)[:n].tolist()

    pair_i, pair_j, alpha_list = [], [], []
    for a_pos in range(n):
        for b_pos in range(a_pos + 1, n):
            for alpha in alphas:
                pair_i.append(order[a_pos])
                pair_j.append(order[b_pos])
                alpha_list.append(float(alpha))

    if not pair_i:
        return []

    i_idx = torch.tensor(pair_i, dtype=torch.long, device=device)
    j_idx = torch.tensor(pair_j, dtype=torch.long, device=device)
    alpha_t = torch.tensor(alpha_list, dtype=torch.float32, device=device)

    out_i = enc_out_pool[i_idx]
    out_j = enc_out_pool[j_idx]
    node_mask_i = node_mask_pool[i_idx]
    node_mask_j = node_mask_pool[j_idx]
    target_attn_i = target_attn_mask_pool[i_idx]
    target_attn_j = target_attn_mask_pool[j_idx]

    alpha_o = alpha_t.view(-1, 1, 1)

    with torch.no_grad():
        out_cross = alpha_o * out_i + (1.0 - alpha_o) * out_j
        # Union of node masks: any position that's a real node in either
        # parent contributes to the interpolated graph.
        node_mask_cross = node_mask_i | node_mask_j
        # Union of target-token masks: decoder's output-loss mask covers
        # token positions valid in either parent.
        target_attn_cross = target_attn_i | target_attn_j
        emb_cross = nfo.encoder.extract_feature(out_cross, node_mask_cross)

        _, sentences = nfo.decoder(
            emb_cross, out_cross, node_mask_cross, target_attn_cross,
        )

    feats = tokenizer.inverse_transform(sentences)
    n_unique = len(set(feats))
    log(f"[CROSSOVER] pairs={n * (n - 1) // 2} alphas={len(alphas)} "
        f"-> {len(feats)} candidates ({n_unique} unique strings)")
    return feats
