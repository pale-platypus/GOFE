"""Pareto frontier of feature USEFULNESS vs COMPLEXITY (the headline figure
for a bi-objective AutoFE paper).

For each dataset we pool every evaluated feature (across the lam grid dirs) per
method from outputs_num/quality_compare/<ds>/lam*/per_feature.csv, then plot:

    x = complexity  (Kext, lower = better / more interpretable)
    y = usefulness  (MI,   higher = better)

The Pareto-optimal direction is the UPPER-LEFT corner (high MI, low Kext).
B-MAFE (method 'lsbo') traces a frontier by varying lambda; baselines appear as
scatter. We draw B-MAFE's non-dominated frontier as a step line and quantify how
many baseline features it dominates.

Usage:
    python scripts/plot_pareto.py [datasets ...]
        [--use mi|pearson|fisher] [--comp kext|dbf]
        [--outdir plots/pareto]
Defaults: spambase winequality-red pima
    (the three highest-MI classification datasets in Table II)
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QC = 'outputs_num/quality_compare'

# display name + style; order controls legend/z-order. 'lsbo' = proposed.
METHODS = [
    ('base_features', 'Base features', 'o', '#888888'),
    # ('random',        'Random',        'v', '#bcbd22'),
    ('autofeat',      'AutoFeat',      's', '#17becf'),
    ('openfe',        'OpenFE',        'D', '#2ca02c'),
    ('eafe',          'E-AFE',         'P', '#9467bd'),
    ('infer',         'DIFER',         'X', '#d62728'),
    ('lsbo',          'GOFE (ours)', '*', '#1f77b4'),
]
DISPLAY = {m: d for m, d, _, _ in METHODS}
MARKER = {m: mk for m, _, mk, _ in METHODS}
COLOR = {m: c for m, _, _, c in METHODS}


def load_pool(ds, use, comp):
    """One row per (method, distinct feat_str); raw use/comp values."""
    files = sorted(glob.glob(os.path.join(QC, ds, 'lam*', 'per_feature.csv')))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.rename(columns={use: 'use', comp: 'comp'})
    df = df.dropna(subset=['use', 'comp'])
    df = df[df['comp'] > 0]
    # dedupe identical expressions within a method (lam dirs duplicate baselines)
    df = df.sort_values('use', ascending=False).drop_duplicates(
        subset=['method', 'feat_str'])
    return df


def pareto_front(use, comp):
    """Indices of non-dominated points: maximize use, minimize comp."""
    order = np.argsort(comp)               # ascending complexity
    best_use = -np.inf
    keep = []
    for idx in order:
        if use[idx] > best_use + 1e-12:
            keep.append(idx)
            best_use = use[idx]
    return keep


def dominates_count(front_use, front_comp, b_use, b_comp):
    """How many baseline points are dominated by the frontier
    (some frontier pt has use>= and comp<= with at least one strict)."""
    dominated = 0
    for u, c in zip(b_use, b_comp):
        hit = np.any((front_use >= u) & (front_comp <= c) &
                     ((front_use > u) | (front_comp < c)))
        dominated += int(hit)
    return dominated

DATASETS = {
    'pima': 'Pima',
    'german': 'German Credit',
    'housing': 'Housing Boston',
    'spambase': 'Spambase',
    'winequality-red': 'Wine Quality (Red)',
}

def panel(ax, ds, use, comp, use_label, comp_label, show_xlabel=True,
          show_ylabel=True):
    df = load_pool(ds, use, comp)
    if df is None or df.empty:
        ax.set_title(f'{ds} (no data)')
        return None
    # scatter every method
    for m, _, _, _ in METHODS:
        g = df[df['method'] == m]
        if g.empty:
            continue
        is_ours = (m == 'lsbo')
        ax.scatter(g['comp'], g['use'],
                   s=120 if is_ours else 42,
                   marker=MARKER[m], c=COLOR[m],
                   edgecolors='black' if is_ours else 'none',
                   linewidths=0.6 if is_ours else 0,
                   alpha=0.95 if is_ours else 0.55,
                   zorder=5 if is_ours else 3,
                   label=DISPLAY[m])
    # B-MAFE frontier
    ours = df[df['method'] == 'lsbo']
    stat = None
    if not ours.empty:
        u = ours['use'].to_numpy()
        c = ours['comp'].to_numpy()
        keep = pareto_front(u, c)
        fc = c[keep]; fu = u[keep]
        srt = np.argsort(fc)
        ax.step(fc[srt], fu[srt], where='post', color=COLOR['lsbo'],
                lw=2.0, alpha=0.9, zorder=4)
        # Dominance over the union of the four COMPETING baselines only.
        # NOTE: an exclusion list ('not lsbo, not base_features') silently swept
        # in B-MAFE's own ablation variants (lsbo_pearson_mdl, lsbo_fisher_mdl)
        # and `random`, so the reported percentage counted B-MAFE dominating
        # itself. Use an explicit allow-list of the baselines the paper names.
        BASELINE_METHODS = ('openfe', 'autofeat', 'infer', 'eafe')
        base = df[df['method'].isin(BASELINE_METHODS)]
        ndom = dominates_count(fu, fc, base['use'].to_numpy(),
                               base['comp'].to_numpy())
        nb = len(base)
        stat = dict(dataset=ds, n_baseline=nb, n_dominated=ndom,
                    pct_dominated=100.0 * ndom / nb if nb else np.nan,
                    frontier_size=len(keep))
    ax.set_title(DATASETS[ds], fontsize=18)
    if show_xlabel:
        ax.set_xlabel(f'{comp_label} (↓)')
    if show_ylabel:
        ax.set_ylabel(f'{use_label} (↑)')
    ax.grid(True, alpha=0.3)
    # annotate ideal corner
    # ax.annotate('ideal', xy=(0.02, 0.97), xycoords='axes fraction', fontsize=8, color='gray', va='top')
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*',
                    default=['spambase', 'winequality-red', 'pima'])
    ap.add_argument('--use', default='mi', choices=['mi', 'pearson', 'fisher'])
    ap.add_argument('--comp', default='kext', choices=['kext', 'dbf'])
    ap.add_argument('--outdir', default='plots/pareto')
    ap.add_argument('--ncol', type=int, default=None,
                    help='columns in the combined panel (default: n if <=3 '
                         'datasets (single row), 2 if 4, else 4)')
    args = ap.parse_args()

    # bump every font by 6 units relative to matplotlib defaults
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
    })

    use_label = {'mi': 'Mutual information, MI', 'pearson': '|Pearson|',
                 'fisher': 'Fisher score'}[args.use]
    comp_label = {'kext': 'Description complexity, $K$',
                  'dbf': 'Distinct base features DBF'}[args.comp]

    os.makedirs(args.outdir, exist_ok=True)
    stats = []

    # combined multi-panel
    n = len(args.datasets)
    # <=3 datasets -> single row (avoids an empty cell in the 3-panel default)
    ncol = args.ncol if args.ncol else (n if n <= 3 else (2 if n == 4 else 4))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.4 * nrow),
                             squeeze=False)
    for i, ds in enumerate(args.datasets):
        ax = axes[i // ncol][i % ncol]
        # only label x on the bottom row of each column, y on the left column
        bottom = (i // ncol) == (nrow - 1) or (i + ncol) >= n
        left = (i % ncol) == 0
        s = panel(ax, ds, args.use, args.comp, use_label, comp_label,
                  show_xlabel=bottom, show_ylabel=left)
        if s:
            stats.append(s)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    # single shared legend. On a single-row figure the panel is wide but short,
    # so a 2-row legend box would hang down over the axes titles -> lay the
    # entries out in one row instead.
    handles, labels = axes[0][0].get_legend_handles_labels()
    leg_ncol = len(handles) if nrow == 1 else 3
    leg = fig.legend(handles, labels, loc='upper center', ncol=leg_ncol,
                     fontsize=16, frameon=True, bbox_to_anchor=(0.5, 1.05))
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.0)
    base = os.path.join(args.outdir, f'pareto_{args.use}_{args.comp}')
    fig.savefig(base + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(base + '.pdf', bbox_inches='tight')
    plt.close(fig)

    # per-dataset standalone
    for ds in args.datasets:
        f, a = plt.subplots(figsize=(10, 5))
        s = panel(a, ds, args.use, args.comp, use_label, comp_label)
        a.legend(fontsize=14, loc='lower right')
        f.tight_layout()
        f.savefig(os.path.join(args.outdir, f'{ds}_{args.use}_{args.comp}.png'),
                  dpi=150, bbox_inches='tight')
        plt.close(f)

    if stats:
        sdf = pd.DataFrame(stats)
        sdf.to_csv(os.path.join(args.outdir,
                                f'dominance_{args.use}_{args.comp}.csv'),
                   index=False)
        print(sdf.to_string(index=False))
    print(f'\nwrote -> {base}.png/.pdf  (+ per-dataset)')


if __name__ == '__main__':
    main()
