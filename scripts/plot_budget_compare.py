"""Overlay budget-ablation curves for multiple datasets on one figure.

Reads each <dataset>'s budget_ablation/tables.csv (produced by
scripts/run_budget_ablation.sh) and renders four side-by-side panels
(Usefulness Δ / Complexity Δ / Total Score / Wall-clock) with one line
per dataset. Linear axes throughout.

Usage:
    python scripts/plot_budget_compare.py [datasets ...]
        [--lam 0.5] [--out plots/budget_ablation/compare.png]

Defaults to Pima, German, Spambase if no datasets are passed.
"""
from __future__ import annotations

import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _abl_dir(d):
    return f'outputs_num/quality_compare/budget_ablation/{d}'


def _budget_from_label(s):
    m = re.match(r'lsbo_budget(\d+)', str(s))
    return int(m.group(1)) if m else None


def curve(dataset, lam, stat='best'):
    abl = _abl_dir(dataset)
    tbl = pd.read_csv(os.path.join(abl, 'tables.csv'))
    pf = pd.read_csv(os.path.join(abl, 'per_feature.csv'))
    timing = pd.read_csv(os.path.join(abl, 'timing.csv'))

    base = tbl[tbl['method'] == 'base_features']
    # best-vs-best or mean-vs-mean: the base reference must match the
    # statistic used for the engineered top-k
    mi_col = 'mi_mean' if stat == 'mean' else 'mi_max'
    mi_base = float(base[mi_col].iloc[0]) if not base.empty else float('nan')
    kext_base = float(base['kext_mean'].iloc[0]) if not base.empty else float('nan')

    rows = []
    for m, g in pf.groupby('method'):
        budget = _budget_from_label(m)
        if budget is None:
            continue
        g = g.dropna(subset=['mi']).sort_values(['mi', 'feat_idx'],
                                                 ascending=[False, True])
        if g.empty:
            continue
        if stat == 'mean':
            # mean over the stored top-k (16) rather than the single best
            mi_v = float(g['mi'].mean())
            kext_v = float(g['kext'].mean())
            tot_v = float(g[f'composite_lam{lam}'].mean()) \
                if f'composite_lam{lam}' in g else np.nan
        else:
            best = g.iloc[0]
            mi_v = float(best['mi'])
            kext_v = float(best['kext']) if pd.notna(best.get('kext')) \
                else np.nan
            tot_v = float(best.get(f'composite_lam{lam}', np.nan))
        rows.append({
            'budget': budget,
            'use_delta': (mi_v - mi_base) / mi_base
                         if (pd.notna(mi_base) and mi_base > 0) else np.nan,
            'comp_delta': (kext_v - kext_base) / kext_base
                          if (pd.notna(kext_v) and pd.notna(kext_base)
                              and kext_base > 0) else np.nan,
            'total': tot_v,
        })
    df = pd.DataFrame(rows).sort_values('budget').reset_index(drop=True)
    df = df.merge(timing, on='budget', how='left')
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*',
                    default=['pima', 'spambase'])
    ap.add_argument('--lam', type=float, default=0.5)
    ap.add_argument('--stat', choices=['best', 'mean'], default='best',
                    help="statistic over the stored top-k: single best "
                         "feature ('best') or mean over all top-k ('mean')")
    ap.add_argument('--out_png', default='plots/budget_ablation/budget_compare.png')
    ap.add_argument('--out_pdf', default='plots/budget_ablation/budget_compare.pdf')
    args = ap.parse_args()

    # bump every font to match the Pareto figure
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
    })

    fig, axes2d = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes2d.ravel()
    titles = [
        'Usefulness Δ vs budget',
        'Complexity Δ vs budget',
        'Bi-objective score vs budget',
        'Running time vs budget',
    ]
    pre = 'mean ' if args.stat == 'mean' else ''
    ylabels = [
        f'Mutual information, {pre}MI Δ (↑)',
        f'Description complexity, {pre}K Δ (↓)',
        f'{pre.capitalize()}bi-objective score (↑)' if pre
        else 'Bi-objective score (↑)',
        'GOFE search time (s)',
    ]
    cols = ['use_delta', 'comp_delta', 'total', 'seconds']
    colors = {'pima': 'C0', 'german': 'C1', 'spambase': 'C2'}

    n_features_per_ds = {}
    n_rows_per_ds = {}
    for ds in args.datasets:
        try:
            with open(f'data/{ds}.csv') as f:
                first = f.readline()
            n_features_per_ds[ds] = first.count(',')  # d
            n_rows_per_ds[ds] = sum(1 for _ in open(f'data/{ds}.csv'))
        except FileNotFoundError:
            n_features_per_ds[ds] = '?'
            n_rows_per_ds[ds] = '?'

    for ds in args.datasets:
        try:
            df = curve(ds, args.lam, stat=args.stat)
        except Exception as e:
            print(f'  SKIP {ds}: {type(e).__name__}: {e}')
            continue
        label = f'{ds}  (N={n_rows_per_ds[ds]}, n={n_features_per_ds[ds]})'
        c = colors.get(ds, None)
        for i, col in enumerate(cols):
            axes[i].plot(df['budget'], df[col], marker='o', linewidth=2,
                         markersize=7, label=label, color=c)

    for i, ax in enumerate(axes):
        if i >= 2:                      # x-label on bottom row only
            ax.set_xlabel('Training budget, B')
        ax.set_ylabel(ylabels[i])
        # ax.set_title(titles[i], fontsize=18)
        ax.grid(True, alpha=0.3)
        if i < 2:
            ax.axhline(0, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        ax.set_xlim(left=0)

    # Bi-objective panel: lock y-range as requested (best-stat figure only;
    # the mean curves sit lower, so let them auto-scale).
    if args.stat == 'best':
        axes[2].set_ylim(0.2, 0.85)

    # single shared, black-edged legend across the top
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='upper center',
                     ncol=len(labels), fontsize=16, frameon=True,
                     bbox_to_anchor=(0.5, 1.01))
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(1.0)

    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.0)
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig.savefig(args.out_png, dpi=130, bbox_inches='tight')
    fig.savefig(args.out_pdf, dpi=130, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
