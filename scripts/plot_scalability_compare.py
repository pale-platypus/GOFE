"""Side-by-side scalability comparison across base datasets of different d.

Reads each base's timing.csv from outputs_num/quality_compare/scalability/
and produces one figure with two panels:
  Left:  raw wall-clock vs N, all (method, budget, dataset) combos
  Right: OpenFE / B-MAFE speedup ratio vs N for each dataset

This visualizes how B-MAFE's relative advantage grows with d.

Usage:
    python scripts/plot_scalability_compare.py [bases ...]
        [--out plots/scalability/compare.png]

Defaults to amazon, spambase.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def load_timing(base):
    p = f'outputs_num/quality_compare/scalability/{base}/timing.csv'
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df['budget_int'] = pd.to_numeric(df['budget'], errors='coerce').astype('Int64')
    return df


def _d_of(base):
    """Read number of base features from data/<base>.csv."""
    try:
        with open(f'data/{base}.csv') as f:
            first = f.readline()
        return first.count(',')   # d
    except FileNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bases', nargs='*', default=['amazon', 'german', 'spambase'])
    ap.add_argument('--out', default='plots/scalability/compare.png')
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    cmap = {'amazon': 'C0', 'spambase': 'C1', 'german': 'C2', 'pima': 'C3'}

    for base in args.bases:
        df = load_timing(base)
        if df is None:
            print(f'  SKIP {base}: no timing.csv')
            continue
        d = _d_of(base)
        label_suffix = f'  (d={d})' if d else ''

        # Left panel: raw time vs N — show B-MAFE(B=256), OpenFE, AutoFeat
        sub = df[df['budget_int'] == 256].sort_values('n_rows')
        if not sub.empty:
            axes[0].plot(sub['n_rows'], sub['seconds'],
                         marker='o', linewidth=2, markersize=7,
                         color=cmap.get(base, None), linestyle='-',
                         label=f'{base} B-MAFE(B=256){label_suffix}')
        ofe = df[df['method'] == 'OpenFE'].sort_values('n_rows')
        if not ofe.empty:
            axes[0].plot(ofe['n_rows'], ofe['seconds'],
                         marker='s', linewidth=2.2, markersize=8,
                         color=cmap.get(base, None), linestyle='--',
                         label=f'{base} OpenFE{label_suffix}')
        af = df[df['method'] == 'AutoFeat'].sort_values('n_rows')
        if not af.empty:
            axes[0].plot(af['n_rows'], af['seconds'],
                         marker='^', linewidth=2.2, markersize=8,
                         color=cmap.get(base, None), linestyle=':',
                         label=f'{base} AutoFeat{label_suffix}')

        # Right panel: ratio OpenFE / B-MAFE(B=256)  and  AutoFeat / B-MAFE
        sub_b256 = df[df['budget_int'] == 256][['n_rows', 'seconds']].rename(
            columns={'seconds': 'bmafe_seconds'})
        if not sub_b256.empty:
            ofe_n = df[df['method'] == 'OpenFE'][['n_rows', 'seconds']].rename(
                columns={'seconds': 'baseline_seconds'})
            af_n = df[df['method'] == 'AutoFeat'][['n_rows', 'seconds']].rename(
                columns={'seconds': 'baseline_seconds'})
            for src, name, ls in [(ofe_n, 'OpenFE', '--'), (af_n, 'AutoFeat', ':')]:
                merged = sub_b256.merge(src, on='n_rows').sort_values('n_rows')
                if not merged.empty:
                    ratio = merged['baseline_seconds'] / merged['bmafe_seconds']
                    axes[1].plot(merged['n_rows'], ratio,
                                 marker='o', linewidth=2.0, markersize=8,
                                 color=cmap.get(base, None), linestyle=ls,
                                 label=f'{name} / B-MAFE on {base}{label_suffix}')

    axes[0].set_xlabel('rows N')
    axes[0].set_ylabel('seconds (search wall-clock)')
    axes[0].set_title('Wall-clock vs N')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(left=0)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc='best', fontsize=8)

    axes[1].set_xlabel('rows N')
    axes[1].set_ylabel('baseline time / B-MAFE(B=256) time')
    axes[1].set_title('B-MAFE speedup over baselines')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc='best', fontsize=9)

    fig.suptitle(
        'Scalability comparison across base-feature dimensionality d',
        fontsize=12,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
