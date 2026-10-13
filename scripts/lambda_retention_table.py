"""Emit the lambda-retention LaTeX table (comment 4: the lambda knob spans the
complexity spectrum while retaining most usefulness; baselines are fixed points).

Reads the macro lambda sweep (mean over datasets of the selected top-k's mean
K / DBF / MI at each lambda) and reports, per lambda: K, the reduction in K
relative to lambda=1, MI, and the fraction of lambda=1 MI retained.

Usage:
    python scripts/lambda_retention_table.py \
        --csv outputs_num/quality_compare/lambda_sweep/lambda_sweep_macro.csv \
        --out paper_drafts/table_lambda_retention.tex
"""
from __future__ import annotations
import argparse
import os
import pandas as pd

LAMS = [0.0, 0.2, 0.5, 0.8, 1.0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    df = pd.read_csv(args.csv).set_index('lam')
    ref = df.loc[1.0]

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Usefulness retention across the $\lambda$ spectrum '
        r'(macro-mean over the ten datasets of the selected top-$k$): at '
        r'$\lambda{=}0.5$, $K$ is halved while '
        f"{df.loc[0.5, 'macro_mean_MI'] / ref['macro_mean_MI'] * 100:.0f}\\% "
        r'of the $\lambda{=}1$ MI is retained.}',
        r'\label{tab:lambda-retention}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'$\lambda$ & mean $K$ (bits) & $K$ vs. $\lambda{=}1$ & mean MI & '
        r'MI retained \\',
        r'\midrule',
    ]
    for lam in LAMS:
        row = df.loc[lam]
        k_red = (row['macro_mean_K'] / ref['macro_mean_K'] - 1) * 100
        mi_ret = row['macro_mean_MI'] / ref['macro_mean_MI'] * 100
        lines.append(
            f"{lam:.1f} & {row['macro_mean_K']:.1f} & "
            + (r'--' if lam == 1.0 else f'{k_red:+.0f}\\%') + ' & '
            + f"{row['macro_mean_MI']:.3f} & "
            + (r'100\%' if lam == 1.0 else f'{mi_ret:.0f}\\%') + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
