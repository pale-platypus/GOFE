"""Compute the size of the B-MAFE expression search space.

A feature tree is either a leaf (1 of ``n`` base features) or an operator
node whose children are themselves feature trees of strictly lower depth.
Because the codebase serializes trees to post-order strings without
canonicalizing commutative ops (x+y and y+x are stored as distinct
features), every choice produces a distinct string. So

    g(0) = n
    g(d) = n + u·g(d-1) + b·g(d-1)²

where
    n = number of base features
    u = number of UNARY  operators in the op set
    b = number of BINARY operators in the op set
    d = maximum tree depth (= ``max_order`` in the codebase)

Use ``expression_space_size(n, max_order, u, b)`` to compute g(d) directly.
Use ``op_counts(env)`` to derive (u, b) from a constructed SklearnEnv.

CLI usage:
  python scripts/expression_space.py --n 20 --max_order 4
  python scripts/expression_space.py --table        # show a grid for the paper
"""
from __future__ import annotations
import argparse


def expression_space_size(n: int, max_order: int, n_unary: int, n_binary: int) -> int:
    """Number of distinct post-order feature trees with depth at most max_order."""
    g = n
    for _ in range(max_order):
        g = n + n_unary * g + n_binary * g * g
    return g


def op_counts(env) -> tuple[int, int]:
    """Return (n_unary, n_binary) for a constructed SklearnEnv.

    Walks env.op_info entries; each is ``(name, arity, has_order, fn, type)``.
    """
    n_unary = sum(1 for op in env.op_info if op[1] == 1)
    n_binary = sum(1 for op in env.op_info if op[1] == 2)
    return n_unary, n_binary


def sample_fraction(n_samples: int, n: int, max_order: int,
                    n_unary: int, n_binary: int) -> float:
    """Fraction of the search space n_samples represents."""
    g = expression_space_size(n, max_order, n_unary, n_binary)
    return n_samples / g if g > 0 else float('nan')


def derived_random_set_size(n: int, max_order: int,
                            c: float = 10.0,
                            n_unary: int = 4, n_binary: int = 5,
                            min_size: int = 32) -> int:
    """Log-scale random_set_size that grows with the expression space.

    Returns ``round(c * log2(g(n, max_order)))`` floored at ``min_size``. The
    exact 5-10% rule is mathematically infeasible above max_order=2 (g(20,3) =
    2.4e15 samples), so we use a log proxy: each depth step roughly doubles
    the budget, which keeps cells tractable while still letting the axis
    surface random-pool effects.

    With c=10 the midpoint (n=20, max_order=4) yields ~1045 samples, matching
    the codebase's historical default of 1024.
    """
    import math
    g = expression_space_size(n, max_order, n_unary, n_binary)
    if g <= 1:
        return min_size
    size = round(c * math.log2(g))
    return max(min_size, size)


def _fmt_pct(x: float) -> str:
    if x >= 1.0:
        return '>100% (full)'
    if x >= 0.01:
        return f'{100*x:6.2f} %'
    if x >= 1e-15:
        return f'{100*x:8.2e} %'
    return f'{100*x:8.2e} %'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, help='# base features')
    p.add_argument('--max_order', type=int, help='max tree depth')
    p.add_argument('--u', type=int, default=4, help='# unary ops (default 4 for num-only)')
    p.add_argument('--b', type=int, default=5, help='# binary ops (default 5 for num-only)')
    p.add_argument('--samples', type=int, default=1024,
                   help='# samples to report as a fraction of the space')
    p.add_argument('--table', action='store_true',
                   help='print a full (n, max_order) table instead of one cell')
    p.add_argument('--target_frac', type=float, nargs='*',
                   default=[0.05, 0.10, 0.20],
                   help='target sample fractions to print sample counts for')
    args = p.parse_args()

    if args.table:
        ns = [5, 10, 20, 50, 100]
        ds = [1, 2, 3, 4]
        print(f'Expression space size g(d) for {args.u} unary + {args.b} binary ops')
        print()
        print(f'{"n":>5} | ' + ' | '.join(f'd={d:<24}' for d in ds))
        print('-' * (8 + 28 * len(ds)))
        for n in ns:
            cells = []
            for d in ds:
                g = expression_space_size(n, d, args.u, args.b)
                cells.append(f'{g:24,.0f}' if g < 1e15 else f'{g:24.3e}')
            print(f'{n:>5} | ' + ' | '.join(cells))
        print()
        print(f'Sample counts to hit target fractions of g(d) at max_order=4:')
        for n in ns:
            g = expression_space_size(n, 4, args.u, args.b)
            samples = [f * g for f in args.target_frac]
            sample_strs = [f'{f*100:.0f}%={s:9.2e}' for f, s in zip(args.target_frac, samples)]
            print(f'  n={n:>3}  g(4)={g:9.3e}   ' + '  '.join(sample_strs))
        return

    if args.n is None or args.max_order is None:
        p.error('must supply --n and --max_order (or use --table)')

    g = expression_space_size(args.n, args.max_order, args.u, args.b)
    frac = args.samples / g if g > 0 else float('nan')
    print(f'expression_space_size(n={args.n}, max_order={args.max_order}, '
          f'u={args.u}, b={args.b}) = {g}')
    print(f'  {args.samples} samples = {_fmt_pct(frac)} of the space')
    for f in args.target_frac:
        target = f * g
        print(f'  to sample {100*f:>5.1f} %: {target:.3e} samples needed')


if __name__ == '__main__':
    main()
