"""FAIR held-out-test evaluation.

For each method, take the feature expressions it selected *on the train split*
(<name>__tr — the only data its search/selection ever saw), construct those same
expressions on BOTH train and test base features, fit each downstream model on
TRAIN [base + derived], and score on the held-out TEST [base + derived].

This removes the selection-optimism / leakage that plagued the CV-on-full-data
numbers: the test rows were never seen during feature generation or selection,
and every method is judged the same way on the same untouched test set.

Pipeline models (LIN/KNN) fit their imputer+scaler on TRAIN and apply to TEST,
so there is no preprocessing leakage either.

Usage:
    python scripts/fair_test_eval.py --out fair_results.csv
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from autolearn.feat_selection.env import SklearnEnv
from autolearn.feat_selection.nfo.feat_tree import generate_tree_from_str
from autolearn.feat_selection.eval_quality import _make_model, _PIPELINE_MODELS
from autolearn.metrics import r2_score
from sklearn.metrics import f1_score

FAIR = 'outputs_fair'
MODELS = ['RF', 'LGB', 'XGB', 'LIN', 'KNN']
TOP_K = 16

DATASETS = ['pima', 'german', 'winequality-red', 'spambase', 'amazon',
            'credit_default', 'housing', 'airfoil', '620', '637']

# (display, kind, path-template). kind:
#   'expr' -> path is a manifest json with portable postfix expressions
#             ('features' or 'feature_strs'); construct on train+test envs.
#   'cols' -> path is the manifest DIR; load derived_columns.csv (train) and
#             derived_test_columns.csv (test) — the library's own transform of
#             the held-out test (OpenFE/AutoFeat don't emit portable expressions).
#  {d} -> '<name>__tr'
METHODS = [
    ('OpenFE',         'cols', 'nfo/RF/openfe/iter/{d}'),
    ('AutoFeat',       'cols', 'nfo/RF/autofeat/iter/{d}'),
    ('DIFER',          'expr', 'nfo/RF/infer/iter_difer_orig/{d}/eval_manifest.json'),
    ('E-AFE',          'expr', 'nfo/RF/eafe/iter/{d}/eval_manifest.json'),
    ('Random',         'expr', 'nfo/RF/random/iter/{d}/eval_manifest.json'),
    ('B-MAFE(MI)',     'expr', 'nfo/RF/lsbo/mi_gat/lam1.0/{d}/selected_topk_lam1.0.json'),
    ('B-MAFE(DS)',     'expr', 'nfo/RF/lsbo/mi_gat_downstream/lam1.0/{d}/selected_topk_lam1.0.json'),
    ('B-MAFE(DS-LIN)', 'expr', 'nfo/RF/lsbo/mi_gat_downstream_lin/lam1.0/{d}/selected_topk_lam1.0.json'),
]

DISPLAY = {'pima': 'Pima', 'german': 'German Credit',
           'winequality-red': 'Wine Quality (Red)', 'spambase': 'Spambase',
           'amazon': 'Amazon Employee', 'credit_default': 'Credit Default',
           'housing': 'Housing Boston', 'airfoil': 'Airfoil',
           '620': 'OpenML_620', '637': 'OpenML_637'}

_ENV_CACHE = {}


def get_env(name):
    if name not in _ENV_CACHE:
        _ENV_CACHE[name] = SklearnEnv(name, 4, parallel=False, score_mode='cv')
    return _ENV_CACHE[name]


def read_features(path, top_k=TOP_K):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        j = json.load(f)
    # B-MAFE/DIFER use 'features'; E-AFE/Random use 'feature_strs' (both postfix).
    feats = j.get('features') or j.get('feature_strs') or []
    return [s for s in feats if s][:top_k]


def load_cols_matrices(name, mdir, top_k=TOP_K):
    """For 'cols' baselines (OpenFE/AutoFeat): base + library-transformed derived
    columns, train from derived_columns.csv and test from derived_test_columns.csv
    (both produced by the library's own transform; test never seen in fitting)."""
    env_tr, env_te = get_env(name + '__tr'), get_env(name + '__te')
    btr = env_tr.dataset.instances.copy(); btr.columns = btr.columns.astype(str)
    bte = env_te.dataset.instances.copy(); bte.columns = bte.columns.astype(str)
    Xtr = btr.to_numpy(dtype=np.float64); Xte = bte.to_numpy(dtype=np.float64)
    tr_csv = os.path.join(mdir, 'derived_columns.csv')
    te_csv = os.path.join(mdir, 'derived_test_columns.csv')
    n = 0
    if os.path.exists(tr_csv) and os.path.exists(te_csv):
        dtr = pd.read_csv(tr_csv).to_numpy(dtype=np.float64)[:, :top_k]
        dte = pd.read_csv(te_csv).to_numpy(dtype=np.float64)[:, :top_k]
        # align column count (defensive) and row counts
        k = min(dtr.shape[1], dte.shape[1])
        if dtr.shape[0] == Xtr.shape[0] and dte.shape[0] == Xte.shape[0] and k > 0:
            Xtr = np.concatenate([Xtr, dtr[:, :k]], axis=1)
            Xte = np.concatenate([Xte, dte[:, :k]], axis=1)
            n = k
    return Xtr, Xte, n


def construct_aligned(name, feat_strs):
    """Construct the SAME expressions on train and test; keep only the ones that
    build successfully (and with the right length) on BOTH, so the train/test
    column sets are identical and aligned."""
    env_tr, env_te = get_env(name + '__tr'), get_env(name + '__te')
    n_tr = env_tr.dataset.instances.shape[0]
    n_te = env_te.dataset.instances.shape[0]
    tr_trees = generate_tree_from_str(feat_strs, env_tr)
    te_trees = generate_tree_from_str(feat_strs, env_te)
    cols_tr, cols_te = [], []
    for s, tt, te in zip(feat_strs, tr_trees, te_trees):
        if tt is None or te is None:
            continue
        try:
            ctr = np.asarray(env_tr.construct_feature(tt), dtype=np.float64).reshape(-1)
            cte = np.asarray(env_te.construct_feature(te), dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if ctr.shape[0] != n_tr or cte.shape[0] != n_te:
            continue
        cols_tr.append(ctr)
        cols_te.append(cte)
    return cols_tr, cols_te


def matrices(name, feat_strs):
    env_tr, env_te = get_env(name + '__tr'), get_env(name + '__te')
    btr = env_tr.dataset.instances.copy(); btr.columns = btr.columns.astype(str)
    bte = env_te.dataset.instances.copy(); bte.columns = bte.columns.astype(str)
    Xtr = btr.to_numpy(dtype=np.float64); Xte = bte.to_numpy(dtype=np.float64)
    cols_tr, cols_te = construct_aligned(name, feat_strs)
    if cols_tr:
        Xtr = np.concatenate([Xtr] + [c.reshape(-1, 1) for c in cols_tr], axis=1)
        Xte = np.concatenate([Xte] + [c.reshape(-1, 1) for c in cols_te], axis=1)
    return Xtr, Xte, len(cols_tr)


def _fit_score(name, Xtr, Xte, n_kept):
    env_tr, env_te = get_env(name + '__tr'), get_env(name + '__te')
    task = env_tr.dataset.task
    y_tr = np.asarray(env_tr.dataset.labels)
    y_te = np.asarray(env_te.dataset.labels)
    out = {'n_features': n_kept}
    for mdl in MODELS:
        xtr = Xtr.copy(); xte = Xte.copy()
        xtr[~np.isfinite(xtr)] = np.nan; xte[~np.isfinite(xte)] = np.nan
        if mdl not in _PIPELINE_MODELS:
            xtr[np.isnan(xtr)] = -1.0; xte[np.isnan(xte)] = -1.0
        try:
            m = _make_model(task, mdl)
            m.fit(xtr, y_tr)
            pred = m.predict(xte)
            s = r2_score(y_te, pred) if task == 'regression' \
                else f1_score(y_te, pred, average='micro')
        except Exception as e:
            s = float('nan')
        out[mdl] = float(s)
    out['avg'] = float(np.nanmean([out[m] for m in MODELS]))
    return out


def score_on_test(name, feat_strs):
    """Expr methods: build [base + constructed derived] for train/test, fit+score."""
    Xtr, Xte, n_kept = matrices(name, feat_strs)
    return _fit_score(name, Xtr, Xte, n_kept)


def base_scores(name):
    return score_on_test(name, [])  # no derived features = base anchor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='fair_results.csv')
    p.add_argument('--datasets', nargs='+', default=DATASETS)
    args = p.parse_args()

    rows = []
    for d in args.datasets:
        base = base_scores(d)
        rows.append({'Dataset': DISPLAY.get(d, d), 'Method': 'Base (no FE)',
                     **{k: base[k] for k in ['n_features'] + MODELS + ['avg']}})
        for nice, kind, tmpl in METHODS:
            path = f'{FAIR}/{tmpl.format(d=d + "__tr")}'
            if kind == 'cols':
                Xtr, Xte, n = load_cols_matrices(d, path)
                if n == 0:
                    print(f'  [miss] {d:8} {nice:16} (no derived_test_columns at {path})', flush=True)
                    continue
                sc = _fit_score(d, Xtr, Xte, n)
            else:
                feats = read_features(path)
                if feats is None:
                    print(f'  [miss] {d:8} {nice:16} (no manifest at {path})', flush=True)
                    continue
                sc = score_on_test(d, feats)
            rows.append({'Dataset': DISPLAY.get(d, d), 'Method': nice,
                         **{k: sc[k] for k in ['n_features'] + MODELS + ['avg']}})
            print(f'  {d:8} {nice:16} n={sc["n_features"]:2d} '
                  f'test-avg={sc["avg"]:.3f}', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
