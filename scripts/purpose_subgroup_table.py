"""Emit the subgroup purpose-matrix LaTeX table from purpose_eval's CSV.

Per dataset: held-out Cohen's d of the best selected feature (d_b), mean over
the padded top-16 (d_m), and membership recovery (balanced accuracy of the
train-fitted threshold on the best-train-d feature, applied to test A/B
rows). Anchor row: the raw base features.

Usage:
    python scripts/purpose_subgroup_table.py \
        --csv outputs_fair/purpose_subgroup.csv \
        --out paper_drafts/table_purpose_subgroup.tex
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

DS = ['pima', 'german', 'winequality-red', 'spambase']
DS_TEX = {'pima': 'Pima', 'german': 'German', 'winequality-red': 'Wine Red',
          'spambase': 'Spambase'}
GROUPS = [
    (r'\emph{Purpose-plugged (ours)}', ['B-MAFE(subgroup)']),
    (r'\emph{Model-agnostic, generic}', ['B-MAFE(MI)', 'B-MAFE(Pearson)']),
    (r'\emph{Model-in-the-loop}', ['OpenFE', 'AutoFeat', 'DIFER', 'E-AFE']),
    (r'\emph{Control}', ['Random']),
]
ANCHOR = 'Base (best feature)'
COLS = ['d_best', 'd_mean', 'recov']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    df = pd.read_csv(args.csv)

    def cell(method, d, col):
        r = df[(df.dataset == d) & (df.method == method)]
        return float(r.iloc[0][col]) if not r.empty else float('nan')

    competitors = [m for _, ms in GROUPS for m in ms]
    ranks = {}
    for d in DS:
        for col in COLS:
            vals = {m: cell(m, d, col) for m in competitors}
            vals = {k: v for k, v in vals.items() if np.isfinite(v)}
            order = sorted(vals, key=vals.get, reverse=True)
            ranks[(d, col)] = (order[0] if order else None,
                               order[1] if len(order) > 1 else None)

    def fmt(m, d, col):
        v = cell(m, d, col)
        if not np.isfinite(v):
            return '--'
        s = f'{v:.2f}' if col != 'recov' else f'{v:.3f}'
        best, second = ranks[(d, col)]
        if m == best:
            return r'\textbf{' + s + '}'
        if m == second:
            return r'\underline{' + s + '}'
        return s

    ncols = len(DS) * 3
    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        r'\caption{Subgroup purpose matrix, held-out: standardized distance '
        r"(Cohen's $d$) between the frozen rule-defined subgroups in each "
        r'feature, on the untouched test rows. $d_b$: best selected feature; '
        r'$d_m$: mean over the padded top-16; Rec.: balanced accuracy of the '
        r'train-fitted threshold recovering subgroup membership on test. '
        r'Best per column \textbf{bold}, second-best \underline{underlined}.}',
        r'\label{tab:purpose-subgroup}',
        r'\setlength{\tabcolsep}{3.5pt}',
        r'\begin{tabular}{l' + 'ccc' * len(DS) + '}',
        r'\toprule',
        ' & ' + ' & '.join(r'\multicolumn{3}{c}{' + DS_TEX[d] + '}'
                           for d in DS) + r' \\',
        ' & ' + ' & '.join([r'$d_b$', r'$d_m$', r'Rec.'] * len(DS)) + r' \\',
        r'\midrule',
    ]
    for label, methods in GROUPS:
        lines.append(r'\multicolumn{' + str(ncols + 1) + r'}{l}{' + label + r'} \\')
        for m in methods:
            name = m.replace('B-MAFE', r'\VarFEAT{}')
            cells = [fmt(m, d, c) for d in DS for c in COLS]
            lines.append(r'\quad ' + name + ' & ' + ' & '.join(cells) + r' \\')
        lines.append(r'\midrule')
    cells = []
    for d in DS:
        for c in COLS:
            v = cell(ANCHOR, d, c)
            cells.append('--' if not np.isfinite(v)
                         else (f'{v:.2f}' if c != 'recov' else f'{v:.3f}'))
    lines.append(r'Base features & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
