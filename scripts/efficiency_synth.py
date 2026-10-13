"""Generate a synthetic dataset for the efficiency-sweep experiments.

Wraps sklearn.make_classification with sensible defaults and writes the
two files the SklearnEnv data loader expects:
  data/<name>.csv   — N rows, n+1 columns (last = label), no header (header=None)
  data/<name>.json  — {"0": "num", ..., str(n-1): "num", "task": "C"}

Usage:
  python scripts/efficiency_synth.py --name synth_N_5000 --N 5000 --n 20 [--seed 0]

The generated dataset is binary classification, with ceil(n/2) informative
features and 0 redundant features (so the search space has signal to find).
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True, help='dataset name (used for data/<name>.{csv,json})')
    p.add_argument('--N', type=int, required=True, help='number of rows')
    p.add_argument('--n', type=int, required=True, help='number of feature columns')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n_classes', type=int, default=2)
    args = p.parse_args()

    n_informative = max(2, args.n // 2)
    # make_classification requires n_informative + n_redundant + n_repeated <= n_features.
    X, y = make_classification(
        n_samples=args.N,
        n_features=args.n,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=args.n_classes,
        n_clusters_per_class=2 if args.n_classes == 2 else 1,
        random_state=args.seed,
    )

    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame(X)
    df['label'] = y
    csv_path = f'data/{args.name}.csv'
    df.to_csv(csv_path, header=False, index=False)

    meta = {str(i): 'num' for i in range(args.n)}
    meta['task'] = 'C'
    json_path = f'data/{args.name}.json'
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'wrote {csv_path} ({args.N} rows × {args.n+1} cols)')
    print(f'wrote {json_path} ({args.n} num features, classification)')


if __name__ == '__main__':
    main()
