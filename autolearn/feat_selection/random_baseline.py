"""Uniform random-expression pool generator (padding pool; not a compared method).

Samples ``n_features`` random feature trees from the same vocabulary and depth
budget used by DIFER's initial random pool, then emits an ``eval_manifest.json``
(plus ``derived_columns.csv``) that ``autolearn.feat_selection.eval_quality``
can ingest. Because the manifest carries ``feature_strs`` and uses
DIFER-format post-order strings, Kext / DBF are scored on the exact tree
(matches the VI / DIFER explainability path).

Usage:
    python -m autolearn.feat_selection.random_baseline \\
        --data pima --n_features 16 \\
        --feat_pool outputs/nfo/RF/random/iter/pima
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

sys.path.extend(["."])

from autolearn.feat_selection.env import SklearnEnv
from autolearn.feat_selection.nfo.feat_tree import (
    generate_tree_from_str,
    is_valid_feat_str,
    random_generate_tree,
)
from autolearn.utils import log


def _sample_unique(env, n_target, max_order, *, max_attempts_per_feat=20):
    feats = env.features
    ops = env.ops
    op_arity = env.op_arity
    op_order = env.op_order
    type_dict = env.type_dict
    max_length = env.max_length

    sampled = []
    seen = set()
    attempts = 0
    cap = max(n_target * max_attempts_per_feat, 200)
    while len(sampled) < n_target and attempts < cap:
        attempts += 1
        tree = random_generate_tree(
            feats, ops, op_arity, op_order, type_dict,
            max_order=max_order, max_length=max_length,
        )
        feat_str = tree.post_order()[0]
        canon, valid = is_valid_feat_str(feat_str, env)
        if not valid or canon in seen:
            continue
        seen.add(canon)
        sampled.append(canon)
    return sampled, attempts


def _materialize(env, feat_strs):
    """Return (derived_df, kept_strs) — drops features that fail to construct."""
    if not feat_strs:
        return pd.DataFrame(), []
    trees = generate_tree_from_str(feat_strs, env)
    cols, kept = {}, []
    for s, t in zip(feat_strs, trees):
        if t is None:
            continue
        try:
            col = env.construct_feature(t)
            arr = np.asarray(col, dtype=np.float64).reshape(-1)
            cols[f"new_{len(kept)}"] = arr
            kept.append(s)
        except Exception as e:
            log(f"[Random] dropping unbuildable {s!r}: {type(e).__name__}: {e}")
    return pd.DataFrame(cols), kept


def run_random(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    env = SklearnEnv(args.data, args.max_order, parallel=False, score_mode='cv')
    log(f"[Random] dataset={args.data} task={env.dataset.task} "
        f"n_base={len(env.features)} target_n_features={args.n_features}")

    t0 = time.time()
    sampled, attempts = _sample_unique(env, args.n_features, args.max_order)
    log(f"[Random] sampled {len(sampled)} unique valid expressions "
        f"in {attempts} attempts ({time.time() - t0:.1f}s)")

    derived, kept_strs = _materialize(env, sampled)
    log(f"[Random] materialized {derived.shape[1]} columns")

    os.makedirs(args.feat_pool, exist_ok=True)
    cols_path = os.path.join(args.feat_pool, 'derived_columns.csv')
    derived.to_csv(cols_path, index=False)
    manifest = {
        'dataset': args.data,
        'method': 'random',
        'lam': None,
        'kind': 'columns',
        'columns_path': 'derived_columns.csv',
        'n_features': int(derived.shape[1]),
        'feature_strs': kept_strs,
        'seed': args.seed,
        'max_order': args.max_order,
    }
    manifest_path = os.path.join(args.feat_pool, 'eval_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    log(f"[Random] wrote eval_manifest to {manifest_path} "
        f"(n_features={derived.shape[1]})")
    return manifest


def parse_args():
    p = argparse.ArgumentParser(description='Random expressions baseline')
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max_order', type=int, default=4)
    p.add_argument('--n_features', type=int, default=16,
                   help='target number of random expressions; should match eval --top_k')
    p.add_argument('--feat_pool', type=str, required=True,
                   help='directory to write eval_manifest.json + derived_columns.csv')
    return p.parse_args()


if __name__ == '__main__':
    run_random(parse_args())
