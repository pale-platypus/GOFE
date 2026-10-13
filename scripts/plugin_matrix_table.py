"""Emit the LaTeX plugin-metric matrix table from plugin_metric_eval.py's CSV.

Layout (comment 1: separate model-agnostic from model-in-the-loop methods):
rows = methods in two labeled groups + Random control + base anchor;
columns = dataset x {MI, |Pearson|} mean-over-top-k on the HELD-OUT test split.
Fisher is computed in the CSV but excluded here (univariate Fisher ranks
features identically to |Pearson|; the paper presents MI + Pearson only).

Best per column bold, second-best underlined (base anchor excluded from
ranking -- it is the reference, not a competitor).

Usage:
    python scripts/plugin_matrix_table.py --csv outputs_fair/plugin_matrix_4ds.csv \
        --out paper_drafts/table_plugin_matrix.tex
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

METRICS = ['MI', 'Pearson']
METRIC_TEX = {'MI': r'MI', 'Pearson': r'$|\rho|$'}
DISPLAY = {'pima': 'Pima', 'german': 'German', 'winequality-red': 'Wine Red',
           'airfoil': 'Airfoil'}

GROUPS = [
    (r'\emph{Model-agnostic (ours)}', ['B-MAFE(MI)', 'B-MAFE(Pearson)']),
    (r'\emph{Model-in-the-loop}', ['OpenFE', 'AutoFeat', 'DIFER', 'E-AFE',
                                   'B-MAFE(DS-RF)', 'B-MAFE(DS-LIN)']),
    (r'\emph{Control}', ['Random']),
]
ANCHOR = 'Base (best feature)'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--split', default='test', choices=['test', 'train'])
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df.split == args.split]
    datasets = [d for d in ['pima', 'german', 'winequality-red', 'airfoil']
                if d in set(df.dataset)]

    def cell_value(method, d, m):
        row = df[(df.dataset == d) & (df.method == method)]
        if row.empty:
            return None
        return float(row.iloc[0][f'mean_{m}'])

    # rank competitors (anchor excluded) per column for bold/underline
    competitors = [m for _, ms in GROUPS for m in ms]
    ranks = {}
    for d in datasets:
        for m in METRICS:
            vals = {meth: cell_value(meth, d, m) for meth in competitors}
            vals = {k: v for k, v in vals.items() if v is not None and np.isfinite(v)}
            order = sorted(vals, key=vals.get, reverse=True)
            ranks[(d, m)] = (order[0] if order else None,
                             order[1] if len(order) > 1 else None)

    def fmt(method, d, m):
        v = cell_value(method, d, m)
        if v is None or not np.isfinite(v):
            return '--'
        s = f'{v:.3f}' if m == 'MI' or v < 10 else f'{v:.1f}'
        best, second = ranks.get((d, m), (None, None))
        if method == best:
            return r'\textbf{' + s + '}'
        if method == second:
            return r'\underline{' + s + '}'
        return s

    ncols = len(datasets) * len(METRICS)
    colspec = 'l' + ('cc' * len(datasets))
    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        r'\caption{Plugin-metric matrix under the leakage-free protocol: mean '
        r'usefulness of the padded top-16 selected by each method (\VarFEAT{} '
        r'variants at $\lambda{=}1$) on the untouched test split, after '
        r'searching on the train split only. Best per column \textbf{bold}, '
        r'second-best \underline{underlined}.}',
        r'\label{tab:plugin-matrix}',
        r'\begin{tabular}{' + colspec + '}',
        r'\toprule',
        ' & ' + ' & '.join(r'\multicolumn{2}{c}{' + DISPLAY.get(d, d) + '}'
                           for d in datasets) + r' \\',
        ' & ' + ' & '.join(METRIC_TEX[m] for _ in datasets for m in METRICS)
        + r' \\',
        r'\midrule',
    ]
    for label, methods in GROUPS:
        lines.append(r'\multicolumn{' + str(ncols + 1) + r'}{l}{' + label + r'} \\')
        for meth in methods:
            cells = [fmt(meth, d, m) for d in datasets for m in METRICS]
            name = meth.replace('B-MAFE', r'\VarFEAT{}')
            lines.append(r'\quad ' + name + ' & ' + ' & '.join(cells) + r' \\')
        lines.append(r'\midrule')
    def anchor_value(d, m):
        row = df[(df.dataset == d) & (df.method == ANCHOR)]
        if row.empty:
            return None
        return float(row.iloc[0][f'best_{m}'])

    cells = []
    for d in datasets:
        for m in METRICS:
            v = anchor_value(d, m)
            cells.append('--' if v is None else
                         (f'{v:.3f}' if m == 'MI' or v < 10 else f'{v:.1f}'))
    lines.append('Best base feature & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
