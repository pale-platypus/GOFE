"""Held-out UTILITY matrix (plugin-metric + downstream-mismatch experiments).

For each method's train-selected top-k features (searched on <name>__tr only),
construct the same expressions on BOTH splits and measure all three
model-agnostic usefulness metrics (MI, |Pearson|, Fisher) per split. The test
split was never seen by any search or selection, so test-split numbers are free
of selection optimism.

Rows cover three method groups:
  plugin     — B-MAFE optimizing each pluggable metric directly (MI / Pearson /
               Fisher at lam=1). Expectation: the variant optimizing metric X
               tops column X (diagonal dominance).
  model-dep  — methods whose search optimizes a downstream model (OpenFE,
               AutoFeat, DIFER, E-AFE, and B-MAFE's own DS-RF / DS-LIN
               variants). Expectation: lower utility than the plugin rows, and
               DS-LIN (linear model on nonlinear data) lowest.
  agnostic   — Random search control.

OpenFE/AutoFeat don't emit portable expressions, so their utility is measured
on the library's own transform of each split (derived_columns.csv /
derived_test_columns.csv), same as fair_test_eval.py.

Usage:
    python scripts/plugin_metric_eval.py --datasets pima german --out plugin_matrix.csv
"""
from __future__ import annotations
import argparse
import os
import sys
import zlib
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from scripts.fair_test_eval import (get_env, read_features, construct_aligned,
                                    FAIR, DISPLAY, TOP_K)
from autolearn.feat_selection.nfo.metrics import (mutual_information, pearson,
                                                  fisher_score)

METRICS = ['MI', 'Pearson', 'Fisher']

# (display, kind, path template on {d}=<name>__tr, group)
METHODS = [
    ('B-MAFE(MI)',      'expr', 'nfo/RF/lsbo/mi_gat/lam1.0/{d}/selected_topk_lam1.0.json',                'plugin'),
    ('B-MAFE(Pearson)', 'expr', 'nfo/RF/lsbo/mi_gat_pearson/lam1.0/{d}/selected_topk_lam1.0.json',        'plugin'),
    ('B-MAFE(Fisher)',  'expr', 'nfo/RF/lsbo/mi_gat_fisher/lam1.0/{d}/selected_topk_lam1.0.json',         'plugin'),
    ('B-MAFE(DS-RF)',   'expr', 'nfo/RF/lsbo/mi_gat_downstream/lam1.0/{d}/selected_topk_lam1.0.json',     'model-dep'),
    ('B-MAFE(DS-LIN)',  'expr', 'nfo/RF/lsbo/mi_gat_downstream_lin/lam1.0/{d}/selected_topk_lam1.0.json', 'model-dep'),
    ('OpenFE',          'cols', 'nfo/RF/openfe/iter/{d}',                                                 'model-dep'),
    ('AutoFeat',        'cols', 'nfo/RF/autofeat/iter/{d}',                                               'model-dep'),
    ('DIFER',           'expr', 'nfo/RF/infer/iter_difer_orig/{d}/eval_manifest.json',                    'model-dep'),
    ('E-AFE',           'expr', 'nfo/RF/eafe/iter/{d}/eval_manifest.json',                                'model-dep'),
    ('Random',          'expr', 'nfo/RF/random/iter/{d}/eval_manifest.json',                              'agnostic'),
]


def _fill_median(v):
    v = np.asarray(v, np.float64).reshape(-1)
    v = np.where(np.isfinite(v), v, np.nan)
    if np.all(np.isnan(v)):
        return None
    return np.where(np.isnan(v), np.nanmedian(v), v)


def util_metrics(v, y, task):
    v = _fill_median(v)
    if v is None:
        return {m: float('nan') for m in METRICS}
    return {'MI': float(mutual_information(v, y, task)),
            'Pearson': float(pearson(v, y)),
            'Fisher': float(fisher_score(v, y))}


def method_columns(d, kind, tmpl, top_k=TOP_K):
    """Per-feature columns on (train, test) for a method, aligned across
    splits. Returns (cols_tr, cols_te, own_strs) — own_strs is None for
    'cols' methods (library columns carry no portable expression)."""
    path = os.path.join(FAIR, tmpl.format(d=d + '__tr'))
    if kind == 'cols':
        tr_csv = os.path.join(path, 'derived_columns.csv')
        te_csv = os.path.join(path, 'derived_test_columns.csv')
        if not (os.path.exists(tr_csv) and os.path.exists(te_csv)):
            return None
        dtr = pd.read_csv(tr_csv).to_numpy(dtype=np.float64)[:, :top_k]
        dte = pd.read_csv(te_csv).to_numpy(dtype=np.float64)[:, :top_k]
        k = min(dtr.shape[1], dte.shape[1])
        if k == 0:
            return None
        return [dtr[:, i] for i in range(k)], [dte[:, i] for i in range(k)], None
    feats = read_features(path, top_k=top_k)
    if feats is None:
        return None
    cols_tr, cols_te = construct_aligned(d, feats)
    return (cols_tr, cols_te, feats) if cols_tr else None


def random_pool(d, top_k=TOP_K):
    """[(feat_str, col_tr, col_te)] for the Random baseline's expressions —
    an unbiased uniform sample of valid random trees over the same grammar,
    used to pad short methods up to a common k."""
    path = os.path.join(FAIR, f'nfo/RF/random/iter/{d}__tr/eval_manifest.json')
    feats = read_features(path, top_k=top_k) or []
    pool = []
    for s in feats:
        ctr, cte = construct_aligned(d, [s])
        if ctr:
            pool.append((s, ctr[0], cte[0]))
    return pool


def pad_with_random(d, method, cols_tr, cols_te, own_strs, pool, top_k):
    """Pad a short method's column lists up to top_k with seeded random draws
    from the Random pool (no replacement; dedup by expression string where
    the method's own expressions are known). Returns n_padded."""
    need = top_k - len(cols_tr)
    if need <= 0 or not pool:
        return 0
    avail = [p for p in pool
             if own_strs is None or p[0] not in set(own_strs)]
    rng = np.random.default_rng(zlib.crc32(f'{d}|{method}'.encode()))
    take = rng.permutation(len(avail))[:need]
    for i in take:
        _, ctr, cte = avail[i]
        cols_tr.append(ctr)
        cols_te.append(cte)
    return len(take)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['pima', 'german'])
    p.add_argument('--out', default='plugin_matrix.csv')
    p.add_argument('--top_k', type=int, default=TOP_K)
    p.add_argument('--no_pad', action='store_true',
                   help='disable random padding: aggregate each method over '
                        'only its own returned features (legacy behaviour)')
    args = p.parse_args()

    rows = []
    for d in args.datasets:
        env_tr, env_te = get_env(d + '__tr'), get_env(d + '__te')
        task = env_tr.dataset.task
        splits = {'train': (env_tr, None), 'test': (env_te, None)}
        ys = {s: np.asarray(env.dataset.labels) for s, (env, _) in splits.items()}

        # base-feature anchor: best base feature per metric, per split
        base_best = {}
        for s, env in [('train', env_tr), ('test', env_te)]:
            X = env.dataset.instances.to_numpy(dtype=np.float64)
            per = [util_metrics(X[:, j], ys[s], task) for j in range(X.shape[1])]
            base_best[s] = {m: max(pm[m] for pm in per) for m in METRICS}
            rows.append({'dataset': d, 'method': 'Base (best feature)',
                         'group': 'base', 'split': s, 'n_features': X.shape[1],
                         **{f'best_{m}': base_best[s][m] for m in METRICS},
                         **{f'mean_{m}': float(np.nanmean([pm[m] for pm in per]))
                            for m in METRICS},
                         **{f'gain_{m}': 0.0 for m in METRICS}})

        pool = [] if args.no_pad else random_pool(d, top_k=args.top_k)
        print(f"\n== {DISPLAY.get(d, d)} [{task}] ==", flush=True)
        for nice, kind, tmpl, group in METHODS:
            got = method_columns(d, kind, tmpl, top_k=args.top_k)
            if got is None:
                print(f"  [miss] {nice}", flush=True)
                continue
            cols_tr, cols_te, own_strs = got
            n_own = len(cols_tr)
            n_pad = 0
            if not args.no_pad and nice != 'Random':
                n_pad = pad_with_random(d, nice, cols_tr, cols_te, own_strs,
                                        pool, args.top_k)
            for s, cols in [('train', cols_tr), ('test', cols_te)]:
                per = [util_metrics(c, ys[s], task) for c in cols]
                best = {m: float(np.nanmax([pm[m] for pm in per])) for m in METRICS}
                mean = {m: float(np.nanmean([pm[m] for pm in per])) for m in METRICS}
                gain = {m: (best[m] - base_best[s][m]) / abs(base_best[s][m])
                        if base_best[s][m] else float('nan') for m in METRICS}
                rows.append({'dataset': d, 'method': nice, 'group': group,
                             'split': s, 'n_features': len(cols),
                             'n_own': n_own, 'n_padded': n_pad,
                             **{f'best_{m}': best[m] for m in METRICS},
                             **{f'mean_{m}': mean[m] for m in METRICS},
                             **{f'gain_{m}': gain[m] for m in METRICS}})
            te = [r for r in rows if r['dataset'] == d and r['method'] == nice
                  and r['split'] == 'test'][0]
            print(f"  {nice:16} n={te['n_own']:2d}+{te['n_padded']:2d}  test gains: "
                  + '  '.join(f"{m}={te[f'gain_{m}']:+.3f}" for m in METRICS),
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    for d in args.datasets:
        for s in ('test', 'train'):
            sub = df[(df.dataset == d) & (df.split == s)]
            if sub.empty:
                continue
            print(f"\n===== {DISPLAY.get(d, d)} — {s.upper()} split: "
                  f"relative gain of best top-k feature vs best base feature =====")
            piv = sub.set_index('method')[[f'gain_{m}' for m in METRICS]]
            piv.columns = METRICS
            print(piv.round(3).to_string())
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
