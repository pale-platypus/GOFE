"""Summarize the maximum-expression-depth (Dmax) ablation.

Reads outputs_num/quality_compare/depth_ablation/<ds>/{tables.csv,timing.csv}
produced by scripts/run_maxorder_ablation.sh and emits:
  * a stdout summary table (MI of the best selected feature, mean MI, mean K,
    mean DBF, wall-clock) per dataset x Dmax,
  * paper-ready LaTeX,
  * a 2-panel figure (usefulness vs Dmax, complexity vs Dmax).

All numbers are relative to each dataset's own base-feature anchor where a
delta is reported, matching the convention used in the main results table.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = 'outputs_num/quality_compare/depth_ablation'
DEPTHS = [2, 3, 4, 5]
DATASETS = [('pima', 'Pima'), ('housing', 'Housing Boston'), ('spambase', 'Spambase')]
# Okabe-Ito, CVD-safe
COLORS = {'Pima': '#0072B2', 'Housing Boston': '#D55E00', 'Spambase': '#009E73'}


def load(ds):
    t = pd.read_csv(os.path.join(ROOT, ds, 'tables.csv'))
    base = t[t['method'] == 'base_features']
    mi_base = float(base['mi_max'].iloc[0])
    k_base = float(base['kext_mean'].iloc[0])
    rows = []
    for d in DEPTHS:
        r = t[t['method'] == f'lsbo_depth{d}']
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append(dict(depth=d,
                         mi_max=float(r['mi_max']),
                         mi_mean=float(r['mi_mean']),
                         kext=float(r['kext_mean']),
                         dbf=float(r['dbf_mean']),
                         mi_delta=(float(r['mi_max']) - mi_base) / mi_base,
                         k_delta=(float(r['kext_mean']) - k_base) / k_base))
    df = pd.DataFrame(rows)
    tim = os.path.join(ROOT, ds, 'timing.csv')
    if os.path.exists(tim):
        tt = pd.read_csv(tim).drop_duplicates('depth')
        df = df.merge(tt, on='depth', how='left')
    return df, mi_base, k_base


def main():
    all_df = {}
    print('=' * 78)
    print('MAXIMUM EXPRESSION DEPTH (Dmax) ABLATION  -- B-MAFE(MI), lambda=0.5')
    print('=' * 78)
    for ds, disp in DATASETS:
        df, mi_base, k_base = load(ds)
        all_df[disp] = df
        print(f'\n--- {disp}  (base: best-base MI={mi_base:.4f}, K_base={k_base:.3f}) ---')
        show = df.copy()
        show['mi_delta'] = show['mi_delta'].map(lambda v: f'{v:+.3f}')
        show['k_delta'] = show['k_delta'].map(lambda v: f'{v:+.3f}')
        print(show.to_string(index=False,
                             float_format=lambda v: f'{v:.4f}'))

    # ---------------- LaTeX ----------------
    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Effect of the maximum expression depth $D_{\max}$ on '
                 r'\VarFEAT{} (MI usefulness, $\lambda=0.5$, single run of three '
                 r'search iterations per cell). MI is the mutual information of the '
                 r'best selected feature; $K$ and DBF are means over the selected '
                 r'top-$k$. Usefulness saturates by $D_{\max}=3$--$4$ while '
                 r'complexity grows monotonically, motivating $D_{\max}=4$ as the '
                 r'default operating point.}')
    lines.append(r'\label{tab:depth-ablation}')
    lines.append(r'\begin{tabular}{llcccc}')
    lines.append(r'\toprule')
    lines.append(r'Dataset & $D_{\max}$ & MI & mean $K$ & mean DBF & time (s) \\')
    lines.append(r'\midrule')
    for i, (ds, disp) in enumerate(DATASETS):
        df = all_df[disp]
        for j, r in df.iterrows():
            name = disp if j == 0 else ''
            secs = '' if 'seconds' not in df.columns or pd.isna(r.get('seconds')) \
                else f"{int(r['seconds'])}"
            lines.append(f"{name} & {int(r['depth'])} & {r['mi_max']:.3f} & "
                         f"{r['kext']:.2f} & {r['dbf']:.2f} & {secs} \\\\")
        if i < len(DATASETS) - 1:
            lines.append(r'\midrule')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    tex = '\n'.join(lines)
    os.makedirs('outputs_num/quality_compare/depth_ablation', exist_ok=True)
    with open('outputs_num/quality_compare/depth_ablation/depth_table.tex', 'w') as f:
        f.write(tex + '\n')
    print('\n' + '=' * 78)
    print('LaTeX  ->  outputs_num/quality_compare/depth_ablation/depth_table.tex')
    print('=' * 78)
    print(tex)

    # ---------------- Figure ----------------
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.5))
    for disp, df in all_df.items():
        c = COLORS[disp]
        ax[0].plot(df['depth'], df['mi_max'], marker='o', lw=2, ms=6, color=c, label=disp)
        ax[1].plot(df['depth'], df['kext'], marker='s', lw=2, ms=6, color=c, label=disp)
    ax[0].set_xlabel(r'maximum expression depth  $D_{\max}$', fontsize=10)
    ax[0].set_ylabel('mutual information  MI  ' + r'$\uparrow$', fontsize=10)
    ax[1].set_xlabel(r'maximum expression depth  $D_{\max}$', fontsize=10)
    ax[1].set_ylabel('mean description complexity  K  ' + r'$\downarrow$', fontsize=10)
    for a in ax:
        a.set_xticks(DEPTHS)
        a.grid(True, lw=0.4, color='#EEEEEE')
        a.set_axisbelow(True)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    ax[0].legend(fontsize=8.5, frameon=False)
    fig.tight_layout(pad=0.4)
    os.makedirs('plots/depth_ablation', exist_ok=True)
    fig.savefig('plots/depth_ablation/depth_ablation.png', dpi=300,
                bbox_inches='tight', pad_inches=0.02)
    fig.savefig('plots/depth_ablation/depth_ablation.pdf',
                bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('\nwrote plots/depth_ablation/depth_ablation.{png,pdf}')


if __name__ == '__main__':
    main()
