"""Create row-subsampled copies of a dataset for scalability experiments.

Writes data/<base>_n<N>.csv and data/<base>_n<N>.json for each N. Reuses the
parent dataset's metadata (task, time_budget, meta dict). Stratified sampling
when the parent is classification, simple random when regression.

Usage:
    python scripts/make_subsamples.py amazon --sizes 500 1000 2000 4000 8000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make_one(base, n, seed=0):
    src_csv = f'data/{base}.csv'
    src_json = f'data/{base}.json'
    if not os.path.exists(src_csv) or not os.path.exists(src_json):
        print(f'  SKIP {base}: missing {src_csv} or {src_json}')
        return None
    df = pd.read_csv(src_csv, header=None)
    with open(src_json) as f:
        meta = json.load(f)
    if n >= len(df):
        print(f'  SKIP n={n}: parent has only {len(df)} rows')
        return None
    rng = np.random.default_rng(seed)
    # Stratified for classification, simple random for regression.
    if meta.get('task') == 'classification':
        y = df.iloc[:, -1]
        sample_idx = []
        for cls, grp in df.groupby(df.columns[-1]):
            target = max(1, int(round(n * len(grp) / len(df))))
            if target >= len(grp):
                sample_idx.extend(grp.index.tolist())
            else:
                sample_idx.extend(rng.choice(grp.index.values, size=target, replace=False).tolist())
        # If rounding over/undershot, pad/trim randomly from full df
        if len(sample_idx) < n:
            remaining = list(set(df.index) - set(sample_idx))
            pad = rng.choice(remaining, size=n - len(sample_idx), replace=False)
            sample_idx.extend(pad.tolist())
        elif len(sample_idx) > n:
            sample_idx = rng.choice(sample_idx, size=n, replace=False).tolist()
        sub = df.loc[sorted(sample_idx)].reset_index(drop=True)
    else:
        sub = df.sample(n=n, random_state=seed).reset_index(drop=True)

    out_base = f'{base}_n{n}'
    sub.to_csv(f'data/{out_base}.csv', header=False, index=False)
    with open(f'data/{out_base}.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  wrote data/{out_base}.csv ({len(sub)} rows)')
    return out_base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base', help='base dataset name (e.g. amazon)')
    ap.add_argument('--sizes', type=int, nargs='+', required=True)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    print(f'Creating subsamples of {args.base}: {args.sizes}')
    for n in args.sizes:
        make_one(args.base, n, seed=args.seed)


if __name__ == '__main__':
    main()
