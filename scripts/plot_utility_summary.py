"""Four-panel method summary across utilities (companion to Table 2).

Panels: (a) mean relative MI gain over the ten datasets, (b) mean relative
description-complexity gap (lower = better), (c) subgroup purpose: mean
held-out d_m over the four datasets, (d) outlier purpose (alpha = 15%):
mean held-out d_m. Each panel draws a dashed horizontal line at the best
baseline's value. E-AFE's subgroup/outlier means cover the three datasets
it produced features on (no Spambase; Table 4).

Usage: python scripts/plot_utility_summary.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

BASELINES = ['OpenFE', 'AutoFeat', 'DIFER', 'E-AFE']
OURS_BLUE, LOOP_ORANGE = '#2a78d6', '#eb6834'


def topk_means():
    df = pd.read_csv(
        'outputs_num/quality_compare/topk_tables/topk_padded16_lam0.5.csv')
    name = {'B-MAFE(MI)': 'GOFE'}
    df['M'] = df['Method'].map(lambda m: name.get(m, m))
    df = df[df['M'].isin(BASELINES + ['GOFE'])]
    g = df.groupby('M')
    return g['MI_delta_mean'].mean(), g['K_delta_mean'].mean()


def purpose_means():
    sub = pd.read_csv('outputs_fair/purpose_subgroup_op5_avg.csv')
    sub['M'] = sub['method'].replace({'B-MAFE(subgroup)': 'GOFE'})
    out = pd.read_csv('outputs_fair/purpose_ifout.csv')
    out = out[np.isclose(out['rule'], 0.15)]
    out['M'] = out['method'].replace({'B-MAFE(ifout)': 'GOFE'})
    base_sub = float(sub[sub['method'] == 'Base (best feature)']['d_mean'].mean())
    base_out = float(out[out['method'] == 'Base (best feature)']['d_mean'].mean())
    sub = sub[sub['M'].isin(BASELINES + ['GOFE'])]
    out = out[out['M'].isin(BASELINES + ['GOFE'])]
    return (sub.groupby('M')['d_mean'].mean(),
            out.groupby('M')['d_mean'].mean(), base_sub, base_out)


def main():
    mi, kx = topk_means()
    dsub, dout, base_sub, base_out = purpose_means()

    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    })
    fig, axes2d = plt.subplots(2, 2, figsize=(3.6, 3.5))
    axes = axes2d.ravel()
    panels = [
        (mi,   r'(a) Informativeness  $\overline{\Delta U}_{\mathrm{MI}}$ ($\uparrow$)',  True,  0.0),
        (kx,   r'(b) Complexity  $\overline{\Delta C}_{\mathrm{desc}}$ ($\downarrow$)',   False, 0.0),
        (dsub, r'(c) Subgroup  $d_m$ ($\uparrow$)',                                       True,  base_sub),
        (dout, r'(d) Outlier  $d_m$ ($\uparrow$)',                                        True,  base_out),
    ]
    order = BASELINES + ['GOFE']
    for ax, (vals, title, higher, base) in zip(axes, panels):
        v = [float(vals[m]) for m in order]
        colors = [LOOP_ORANGE] * 4 + [OURS_BLUE]
        ax.bar(range(5), v, color=colors, width=0.62)
        # best base feature: zero gap by definition in (a)/(b); Table 4's
        # reference row averaged over datasets in (c)/(d)
        ax.axhline(base, color='k', linestyle='--', linewidth=0.9)
        if base <= min(v):
            ax.set_ylim(bottom=base - 0.07 * (max(v) - base))
        ax.set_title(title, pad=3)
        ax.set_xticks(range(5))
        ax.set_xticklabels(order, rotation=40, ha='right')
        ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    # shared legend entry for the reference line
    axes[0].plot([], [], 'k--', linewidth=0.9, label='best base feature')
    fig.legend(loc='lower center', ncol=1, frameon=False,
               bbox_to_anchor=(0.5, -0.025), fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=3.2)
    os.makedirs('plots/utility_summary', exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'plots/utility_summary/utility_summary.{ext}',
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    for lbl, s in zip('abcd', [mi, kx, dsub, dout]):
        print(f'({lbl})', {m: round(float(s[m]), 3) for m in order})
    print('wrote plots/utility_summary/utility_summary.pdf')


if __name__ == '__main__':
    main()
