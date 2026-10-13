"""Aggregate the |O| operator-vocabulary ablation across datasets and seeds
into one LaTeX table (companion to scripts/depth_table.py).

Scans outputs_num/quality_compare/ops_ablation/<dataset>[_s<seed>]/ for
tables.csv (rows method=lsbo_ops<N>) and timing.csv, aggregates per
(dataset, |O|) over seeds as mean +- sd, and reports MI of the selected top-k,
mean description complexity K, mean DBF, and wall-clock.

Usage:
    python scripts/ops_ablation_table.py --out paper_drafts/table_ops_ablation.tex
"""
from __future__ import annotations
import argparse
import glob
import os
import re
import numpy as np
import pandas as pd

ROOT = 'outputs_num/quality_compare/ops_ablation'
DISPLAY = {'pima': 'Pima', 'german': 'German Credit', 'housing': 'Housing Boston',
           'winequality-red': 'Wine Quality (Red)', 'spambase': 'Spambase'}


def collect():
    recs = []
    for run_dir in sorted(glob.glob(os.path.join(ROOT, '*'))):
        if not os.path.isdir(run_dir):
            continue
        base = os.path.basename(run_dir)
        m = re.fullmatch(r'(.+?)(?:_s(\d+))?', base)
        dataset, seed = m.group(1), int(m.group(2) or 0)
        tab = os.path.join(run_dir, 'tables.csv')
        if not os.path.exists(tab):
            continue
        df = pd.read_csv(tab)
        timing = {}
        tcsv = os.path.join(run_dir, 'timing.csv')
        if os.path.exists(tcsv):
            t = pd.read_csv(tcsv)
            timing = dict(zip(t.iloc[:, 0].astype(int), t.iloc[:, 1].astype(float)))
        for _, row in df.iterrows():
            mm = re.fullmatch(r'lsbo_ops(\d+)', str(row['method']))
            if not mm:
                continue
            n_ops = int(mm.group(1))
            recs.append({'dataset': dataset, 'seed': seed, 'n_ops': n_ops,
                         'mi_max': row.get('mi_max', np.nan),
                         'mi_mean': row.get('mi_mean', np.nan),
                         'kext_mean': row.get('kext_mean', np.nan),
                         'dbf_mean': row.get('dbf_mean', np.nan),
                         'seconds': timing.get(n_ops, np.nan)})
    return pd.DataFrame(recs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    args = p.parse_args()

    df = collect()
    if df.empty:
        raise SystemExit(f'no ops_ablation results found under {ROOT}')
    g = df.groupby(['dataset', 'n_ops'])
    agg = g.agg(seeds=('seed', 'nunique'),
                mi=('mi_max', 'mean'), mi_sd=('mi_max', 'std'),
                k=('kext_mean', 'mean'), k_sd=('kext_mean', 'std'),
                dbf=('dbf_mean', 'mean'),
                sec=('seconds', 'mean')).reset_index()

    def pm(v, sd):
        if np.isnan(v):
            return '--'
        if np.isnan(sd) or sd == 0:
            return f'{v:.3f}' if abs(v) < 10 else f'{v:.1f}'
        return (f'{v:.3f}' if abs(v) < 10 else f'{v:.1f}') + \
               r' {\scriptsize$\pm$' + (f'{sd:.3f}' if sd < 10 else f'{sd:.1f}') + '}'

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Effect of the operator-vocabulary size $|\mathcal{O}|$ '
        r'(\VarFEAT{}(MI), $\lambda{=}0.5$, three iterations per cell, mean '
        r'over seeds). MI: best selected feature; $K$ and DBF: means over the '
        r'selected top-$k$, with $K$ on the full-vocabulary basis so rows are '
        r'comparable.}',
        r'\label{tab:ops-ablation}',
        r'\begin{tabular}{llcccc}',
        r'\toprule',
        r'Dataset & $|\mathcal{O}|$ & MI & mean $K$ & mean DBF & time (s) \\',
        r'\midrule',
    ]
    for i, dataset in enumerate(sorted(agg.dataset.unique())):
        sub = agg[agg.dataset == dataset].sort_values('n_ops')
        if i:
            lines.append(r'\midrule')
        for j, (_, r) in enumerate(sub.iterrows()):
            name = DISPLAY.get(dataset, dataset) if j == 0 else ''
            lines.append(f"{name} & {int(r.n_ops)} & {pm(r.mi, r.mi_sd)} & "
                         f"{pm(r.k, r.k_sd)} & {r.dbf:.2f} & {r.sec:.0f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    n_runs = df.groupby('dataset').seed.nunique().to_dict()
    print(f'wrote {args.out}  (seeds per dataset: {n_runs})')


if __name__ == '__main__':
    main()
