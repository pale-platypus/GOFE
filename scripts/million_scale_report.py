# -*- coding: utf-8 -*-
"""Report the million-scale runtime sweep: figure + LaTeX table.

Reads outputs_num/scal_million/timing.csv (written by run_million_scale.sh),
emits a 5-panel runtime figure (one panel per varied parameter; the center
configuration appears in every panel as the shared point) and a compact
LaTeX table. Runtime only — no quality metrics.

Usage:
    python scripts/million_scale_report.py
Outputs:
    plots/scal_million/million_scale.{pdf,png}
    paper_drafts/table_million_scale.tex
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = 'outputs_num/scal_million/timing.csv'
FIG = 'plots/scal_million/million_scale'
TEX = 'paper_drafts/table_million_scale.tex'

CENTER = dict(N=1_000_000, B=256, D=4, nops=9, k=16)
AXES = [
    ('N', 'N', r'rows $N$', True),
    ('B', 'B', r'budget $B$', False),
    ('D', 'D', r'depth $D_{\max}$', False),
    ('O', 'nops', r'vocabulary $|\mathcal{O}|$', False),
    ('K', 'k', r'top-$k$', False),
]
BLUE, INK, MUTED = '#2a78d6', '#0b0b0b', '#52514e'


def load():
    df = pd.read_csv(CSV)
    df = df[df['exit'] == 0]
    center = df[df.axis == 'center']
    return df, (float(center.seconds.iloc[0]) if not center.empty else np.nan)


def axis_points(df, axis, valcol, center_secs):
    sub = df[df.axis == axis][[valcol, 'seconds']].astype(float)
    pts = dict(zip(sub[valcol], sub.seconds))
    if np.isfinite(center_secs):
        pts[float(CENTER[valcol])] = center_secs
    xs = sorted(pts)
    return np.array(xs), np.array([pts[x] for x in xs])


def main():
    df, center_secs = load()
    plt.rcParams.update({
        'font.size': 8, 'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
        'axes.labelcolor': INK,
    })
    fig, axs = plt.subplots(1, 5, figsize=(10.2, 2.1))
    for ax, (axis, valcol, label, logx) in zip(axs, AXES):
        xs, ys = axis_points(df, axis, valcol, center_secs)
        if len(xs) == 0:
            ax.set_visible(False)
            continue
        ax.plot(xs, ys / 60.0, '-o', color=BLUE, lw=1.6, ms=4, zorder=3)
        ci = np.where(xs == float(CENTER[valcol]))[0]
        if ci.size:
            ax.plot(xs[ci], ys[ci] / 60.0, 'o', color=INK, ms=5, zorder=4)
        if logx:
            ax.set_xscale('log')
        ax.set_xlabel(label, fontsize=8)
        if ax is axs[0]:
            ax.set_ylabel('search time (min)', fontsize=8)
        ax.grid(color='#e4e3de', lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG + '.pdf', bbox_inches='tight')
    fig.savefig(FIG + '.png', dpi=200, bbox_inches='tight')
    print('wrote', FIG + '.{pdf,png}')

    # ---- LaTeX table: one block per axis --------------------------------
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Search wall-clock (minutes) on million-scale synthetic '
        r'data ($n{=}20$ base features) as one parameter varies around the '
        r'center configuration $N{=}10^{6}$, $B{=}256$, $D_{\max}{=}4$, '
        r'$|\mathcal{O}|{=}9$, $k{=}16$ (three search iterations, '
        r'\VarFEAT{}(MI), $\lambda{=}1$). Center cell marked $^{*}$.}',
        r'\label{tab:million-scale}',
        r'\begin{tabular}{lrrrrr}',
        r'\toprule',
    ]
    for axis, valcol, label, _ in AXES:
        xs, ys = axis_points(df, axis, valcol, center_secs)
        if len(xs) == 0:
            continue
        vals, times = [], []
        for x, y in zip(xs, ys):
            star = '$^{*}$' if x == float(CENTER[valcol]) else ''
            vals.append((f'{x/1000:.0f}k' if valcol == 'N' and x >= 1000
                         else f'{x:.0f}') + star)
            times.append(f'{y/60:.1f}')
        pad = 5 - len(vals)
        lines.append(label + ' & ' + ' & '.join(vals + [''] * pad) + r' \\')
        lines.append(r'\quad time (min) & ' + ' & '.join(times + [''] * pad)
                     + r' \\')
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines += [r'\end{tabular}', r'\end{table}']
    with open(TEX, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote', TEX)


if __name__ == '__main__':
    main()
