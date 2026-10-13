"""Surrogate model quality (paper Sec. IV-E).

Reads outputs_num/surrogate/pred_true.csv (written by iter_train.py when run
with SURR_DUMP set) and plots predicted vs. true surrogate score on the
held-out VALID split, one panel per (dataset, lambda). At lambda=1 the surrogate
target is pure usefulness U; at lambda=0 it is pure explainability E=(1-Kext_norm).
Reports R^2, Spearman rho, and RMSE per panel.

Usage:  python scripts/plot_surrogate.py
            [--csv outputs_num/surrogate/pred_true.csv]
            [--split valid] [--out plots/surrogate/surrogate_quality.png]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


DATASETS = {
    'pima': 'Pima',
    'german': 'German Credit',
    'housing': 'Housing Boston',
    'spambase': 'Spambase',
}




def target_name(lam):
    if abs(lam - 1.0) < 1e-9:
        return 'Usefulness  $U(f)$'
    if abs(lam) < 1e-9:
        return 'Explainability  $E(f)$'
    return f'Composite  $S(f;\\lambda={lam:g})$'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='outputs_num/surrogate/pred_true.csv')
    ap.add_argument('--split', default='valid', choices=['valid', 'train', 'all'])
    ap.add_argument('--out', default='plots/surrogate/surrogate_quality.png')
    args = ap.parse_args()

    # bump every font to match the Pareto figure
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
    })

    df = pd.read_csv(args.csv)
    if args.split != 'all':
        df = df[df['split'] == args.split]

    groups = sorted(df.groupby(['dataset', 'lam']).groups.keys(),
                    key=lambda t: (t[0], -t[1]))
    n = len(groups)
    ncol = 2 if n <= 4 else 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.4 * nrow),
                             squeeze=False)

    stats = []
    for i, (ds, lam) in enumerate(groups):
        ax = axes[i // ncol][i % ncol]
        g = df[(df['dataset'] == ds) & (df['lam'] == lam)]
        t = g['true'].to_numpy()
        p = g['pred'].to_numpy()
        r2 = r2_score(t, p) if len(t) > 2 else np.nan
        rho = spearmanr(t, p).correlation if len(t) > 2 else np.nan
        rmse = float(np.sqrt(np.mean((t - p) ** 2)))
        stats.append(dict(dataset=ds, lam=lam, target=target_name(lam),
                          n=len(t), r2=r2, spearman=rho, rmse=rmse))

        ax.scatter(t, p, s=18, alpha=0.45, color='#1f77b4', edgecolors='none')
        lo = min(t.min(), p.min())
        hi = max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.6)
        ax.set_title(f'{DATASETS[ds]}', fontsize=18)
        # x-label on bottom row, y-label on left column only
        if (i // ncol) == (nrow - 1) or (i + ncol) >= n:
            ax.set_xlabel('true score')
        if (i % ncol) == 0:
            ax.set_ylabel('predicted score')
        ax.text(0.05, 0.95,
                f'$R^2$={r2:.3f}\n$\\rho$={rho:.3f}\nRMSE={rmse:.3f}\n{len(t)} points',
                transform=ax.transAxes, va='top', fontsize=13,
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        ax.grid(True, alpha=0.3)

    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')

    # fig.suptitle(f'Surrogate prediction quality (held-out {args.split} split)',
    #              fontsize=13)
    fig.tight_layout(h_pad=1.0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    fig.savefig(args.out.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)

    sdf = pd.DataFrame(stats)
    metrics_path = args.out.replace('.png', '').replace(
        'surrogate_quality', 'surrogate_metrics') + '.csv'
    sdf.to_csv(metrics_path, index=False)
    print(sdf.to_string(index=False))
    print(f'\nwrote -> {args.out}')


if __name__ == '__main__':
    main()
