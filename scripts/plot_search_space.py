"""Combinatorial size of the B-MAFE feature search space vs |X| (Eq. 12).

Implements Theorem 1 / Eq. 12. With m1 unary and m2 binary operators, n base
features, and a maximum expression depth Dmax (the maximum NUMBER OF OPERATOR
tokens, i.e. internal nodes), the size of the expression space is

    |F_Dmax| = sum_{D=0}^{Dmax} sum_{b=0}^{D} N(b, D-b) * m2^b * m1^(D-b) * n^(b+1)

    N(b, u) = (2b+u)! / (b! u! (b+1)!)          # rooted ordered tree shapes (Eq. 13)

A term with b binary and u=D-b unary internal nodes has b+1 leaves: N(b,u) counts
the tree shapes, m2^b / m1^u label the operator nodes, and n^(b+1) labels the
leaves with base features. Dmax bounds the number of operators (b+u <= Dmax),
NOT the tree height.

For the default B-MAFE configuration (num-ops-only vocabulary):
  * Binary (m2=5): {+, -, *, / (division), mod}
  * Unary  (m1=4): {log, sqrt|.|, minmax, 1/x (reciprocal)}
  * Maximum expression depth (operators): Dmax = 4 (configurable via --max_order)

The count grows fast in Dmax (Dmax=4 is ~10^10 at n=20 and ~10^13 at n=57), so
we plot log10(|F_Dmax|) on a *linear* y-axis with integer ticks (each tick =
one decade), avoiding matplotlib's log-axis. The x-axis (number of base
features) is also linear.

Usage:
    python scripts/plot_search_space.py
                                  [--max_order 4]
                                  [--n_base_min 1] [--n_base_max 60]
                                  [--out_dir plots/search_space]
"""
from __future__ import annotations

import argparse
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Actual num-ops-only vocabulary used in the experiments (search_space.py):
#   binary (m2=5): +, -, *, / (division), mod
#   unary  (m1=4): log, sqrt|.|, minmax, 1/x (reciprocal)
OPERATORS = ['+', '-', '*', '/', 'mod',
             'log', 'sqrt_abs', 'minmax', 'recip']
N_BINARY = 5
N_UNARY = 4


def _tree_shapes(b: int, u: int) -> int:
    """Eq. 13: number of rooted ordered tree shapes with b binary internal
    nodes, u unary internal nodes, and b+1 leaves (cycle-lemma count)."""
    return math.factorial(2 * b + u) // (
        math.factorial(b) * math.factorial(u) * math.factorial(b + 1))


def tree_count(n_base: int, depth: int, n_unary=N_UNARY, n_binary=N_BINARY) -> float:
    """Size of the expression space |F_Dmax| (Eq. 12): number of grammatically
    valid prefix expressions with at most ``depth`` operator tokens on
    ``n_base`` features, given ``n_binary`` binary and ``n_unary`` unary
    operators. Returns a float (the exact integer overflows plotting needs)."""
    total = 0
    for D in range(depth + 1):
        for b in range(D + 1):
            u = D - b
            total += (_tree_shapes(b, u) * (n_binary ** b)
                      * (n_unary ** u) * (n_base ** (b + 1)))
    return float(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_order', type=int, default=4)
    ap.add_argument('--n_base_min', type=int, default=1)
    ap.add_argument('--n_base_max', type=int, default=60)
    ap.add_argument('--out_dir', default='plots/search_space')
    args = ap.parse_args()

    # bump every font to match the other figures
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
    })

    os.makedirs(args.out_dir, exist_ok=True)
    ns = list(range(args.n_base_min, args.n_base_max + 1))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    depths = list(range(1, args.max_order + 1))
    for d in depths:
        ys = [math.log10(tree_count(n, d)) for n in ns]
        # markers every 5th n to avoid clutter; line stays smooth
        ax.plot(ns, ys, marker='o', markersize=7, markevery=5, linewidth=2,
                label=r'$D_{\mathrm{max}}$ = ' + f'{d}')

    ax.set_xlabel(r'Number of base features  $n$')
    ax.set_ylabel(r'$\log_{10}$(number of possible expressions)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', frameon=True)
    ax.set_xlim(args.n_base_min, args.n_base_max)
    # headroom so the legend box (upper-left) clears the curves
    ax.set_ylim(bottom=0, top=14)

    fig.tight_layout()
    out_png = os.path.join(args.out_dir, 'search_space_vs_n_base.png')
    out_pdf = os.path.join(args.out_dir, 'search_space_vs_n_base.pdf')
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    fig.savefig(out_pdf, dpi=130, bbox_inches='tight')
    plt.close(fig)

    # Also dump a tidy CSV
    csv_path = os.path.join(args.out_dir, 'search_space_counts.csv')
    with open(csv_path, 'w') as f:
        header = ['n_base'] + [f'log10_A_d{d}' for d in depths] + [f'A_d{d}' for d in depths]
        f.write(','.join(header) + '\n')
        for n in ns:
            row = [str(n)]
            row += [f'{math.log10(tree_count(n, d)):.3f}' for d in depths]
            row += [f'{tree_count(n, d):.3e}' for d in depths]
            f.write(','.join(row) + '\n')
    print(f'wrote {csv_path}')

    # Small table for the paper caption (selected points)
    print()
    print('Sample counts:')
    print(f'  {"n":>4}  ' + '  '.join(f'A(d={d})' for d in depths))
    for n in [5, 10, 20, 30, 50, 60]:
        cells = [f'{tree_count(n, d):.2e}' for d in depths]
        print(f'  {n:>4}  ' + '  '.join(f'{c:>10}' for c in cells))


if __name__ == '__main__':
    main()
