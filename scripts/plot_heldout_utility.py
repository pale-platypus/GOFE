# -*- coding: utf-8 -*-
"""Held-out plugin-metric figure for Section IV.

2x4 small multiples: rows = {mean MI, mean |Pearson|} on the untouched test
split, cols = datasets. Horizontal bars per method (padded k=16 policy),
colored by method group; dashed vertical line = the single best base feature
(the conservative anchor). Only the best bar per panel is direct-labeled;
exact values live in the companion table.

Usage:
    python scripts/plot_heldout_utility.py
Outputs: plots/heldout/heldout_utility.{pdf,png}
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = 'outputs_fair/plugin_matrix_4ds.csv'
OUT = 'plots/heldout/heldout_utility'

DS = ['pima', 'german', 'winequality-red', 'airfoil']
DS_NICE = {'pima': 'Pima', 'german': 'German Credit',
           'winequality-red': 'Wine (Red)', 'airfoil': 'Airfoil'}
# (csv name, display, group) top-to-bottom; None = visual gap between groups
ROWS = [
    ('B-MAFE(MI)', 'GOFE(MI)', 'agn'),
    ('B-MAFE(Pearson)', r'GOFE($\rho$)', 'agn'),
    None,
    ('OpenFE', 'OpenFE', 'loop'),
    ('AutoFeat', 'AutoFeat', 'loop'),
    ('DIFER', 'DIFER', 'loop'),
    ('E-AFE', 'E-AFE', 'loop'),
    None,
    ('Random', 'Random', 'ctl'),
]
COLOR = {'agn': '#2a78d6', 'loop': '#eb6834', 'ctl': '#8a8880'}
GROUP_LABEL = {'agn': 'Model-agnostic (ours)', 'loop': 'Model-in-the-loop',
               'ctl': 'Random control'}
METRICS = [('mean_MI', 'best_MI', 'Mean MI of top-16 (test)'),
           ('mean_Pearson', 'best_Pearson', r'Mean $|\rho|$ of top-16 (test)')]
INK, MUTED = '#0b0b0b', '#52514e'


def main():
    te = pd.read_csv(CSV)
    te = te[te.split == 'test']

    plt.rcParams.update({
        'font.size': 8, 'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
        'xtick.color': MUTED, 'ytick.color': INK, 'text.color': INK,
        'axes.labelcolor': INK,
    })
    fig, axes = plt.subplots(2, 4, figsize=(10.2, 4.6), sharey=True)

    ypos, labels, gaps = [], [], []
    y = 0.0
    for r in ROWS:
        if r is None:
            gaps.append(y)
            y += 0.55
            continue
        ypos.append(y)
        labels.append(r[1])
        y += 1.0

    for i, (mcol, bcol, mlabel) in enumerate(METRICS):
        for j, d in enumerate(DS):
            ax = axes[i, j]
            sub = te[te.dataset == d].set_index('method')
            anchor = float(sub.loc['Base (best feature)', bcol])
            vals, cols = [], []
            for r in ROWS:
                if r is None:
                    continue
                name, _, grp = r
                v = (float(sub.loc[name, mcol])
                     if name in sub.index else np.nan)
                vals.append(v)
                cols.append(COLOR[grp])
            vals = np.array(vals)
            ok = np.isfinite(vals)
            ax.barh(np.array(ypos)[ok], vals[ok], height=0.62,
                    color=np.array(cols)[ok], zorder=3)
            for yy in np.array(ypos)[~ok]:
                ax.text(0.004, yy, '--', va='center', ha='left',
                        fontsize=7, color=MUTED, zorder=3)
            ax.axvline(anchor, color=INK, lw=0.9, ls=(0, (4, 2.5)), zorder=4)
            # direct-label the best bar only (selective labeling)
            b = int(np.nanargmax(vals))
            ax.text(vals[b] * 1.02, ypos[b], f'{vals[b]:.2f}',
                    va='center', ha='left', fontsize=7, color=INK, zorder=5,
                    bbox=dict(facecolor='white', edgecolor='none',
                              pad=0.6, alpha=0.85))
            hi = max(np.nanmax(vals), anchor)
            ax.set_xlim(0, hi * 1.18)
            ax.invert_yaxis() if i == 0 and j == 0 else None
            ax.grid(axis='x', color='#e4e3de', lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            for s in ('top', 'right', 'left'):
                ax.spines[s].set_visible(False)
            ax.tick_params(axis='y', length=0)
            ax.tick_params(axis='x', labelsize=7)
            if i == 0:
                ax.set_title(DS_NICE[d], fontsize=9, pad=4)
            if j == 0:
                ax.set_yticks(ypos)
                ax.set_yticklabels(labels, fontsize=7.5)
                ax.set_ylabel(mlabel, fontsize=8)
    # shared y direction: barh + shared axes -> set once
    axes[0, 0].set_ylim(y - 0.45, -0.55)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[g])
               for g in ('agn', 'loop', 'ctl')]
    handles.append(plt.Line2D([0], [0], color=INK, lw=0.9,
                              ls=(0, (4, 2.5))))
    fig.legend(handles,
               [GROUP_LABEL['agn'], GROUP_LABEL['loop'], GROUP_LABEL['ctl'],
                'best single base feature'],
               ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               frameon=False, fontsize=8, handlelength=1.4,
               handleheight=0.9, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + '.pdf', bbox_inches='tight')
    fig.savefig(OUT + '.png', dpi=200, bbox_inches='tight')
    print('wrote', OUT + '.{pdf,png}')


if __name__ == '__main__':
    main()
