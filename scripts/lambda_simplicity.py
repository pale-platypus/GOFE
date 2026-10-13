"""How does lambda play out? We want to see how much simpler features get.

Quantifies HOW the *selected* feature set changes as the trade-off parameter
lambda is swept, for all 10 datasets of the paper's Table II.

Key observation that makes this cheap: the composite score

    S(f; lambda) = lambda * MI_norm(f) + (1 - lambda) * (1 - Kext_norm(f))

depends on the pool only through (MI, Kext), and BOTH Kext and DBF are pure
functions of the postfix string + the (|X|, |O|) vocabulary sizes.  The raw MI
of every evaluated candidate is already stored in the search manifest
``outputs_num/nfo/RF/lsbo/mi_gat/lam0.5/<dataset>/{N}.json``.  So lambda can be
swept OFFLINE at arbitrary resolution over the *same fixed pool* with no
re-running of the search.  This is a **selection-only** re-ranking (see the
caveat printed at the end of the run).

Outputs
-------
  outputs_num/quality_compare/lambda_sweep/<dataset>.csv     per-dataset sweep
  outputs_num/quality_compare/lambda_sweep/lambda_sweep_all.csv      (long)
  outputs_num/quality_compare/lambda_sweep/lambda_sweep_summary.csv  (per-ds stats)
  outputs_num/quality_compare/lambda_sweep/qualitative_top1.csv      (readable)
  outputs_num/quality_compare/lambda_sweep/qualitative_top1.tex      (paper-ready)
  plots/lambda_sweep/lambda_simplicity.pdf / .png (300 dpi)

Usage
-----
  python scripts/lambda_simplicity.py [--tree outputs_num] [--topk 16]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from scripts.du_common import get_env, load_pool, to_infix, to_mathtext
from scripts.du_casestudy import NAMES
from autolearn.feat_selection.nfo.feat_tree import generate_tree_from_str
from autolearn.feat_selection.nfo.metrics import composite_from_pool, dbf as dbf_fn, kext as kext_fn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The 10 datasets of Table II, in the paper's row order
# (mirrors scripts/build_paper_table.ROWS).
DATASETS = [
    ('Pima',               'pima'),
    ('German Credit',      'german'),
    ('Wine Quality (Red)', 'winequality-red'),
    ('Spambase',           'spambase'),
    ('Amazon Employee',    'amazon'),
    ('Credit Default',     'credit_default'),
    ('Housing Boston',     'housing'),
    ('Airfoil',            'airfoil'),
    ('OpenML 620',         '620'),
    ('OpenML 637',         '637'),
]

LAM_GRID = [round(0.1 * i, 1) for i in range(11)]     # 0.0 .. 1.0
QUAL_LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]               # qualitative table
QUAL_DATASETS = ['pima', 'housing', 'airfoil']        # readable feature names

CSV_ROOT = 'outputs_num/quality_compare/lambda_sweep'
PLOT_ROOT = 'plots/lambda_sweep'

# Colorblind-safe hues (dataviz reference palette, light mode: slots
# blue/orange/aqua/violet/red).  10 series exceed the all-pairs colour cap, so
# identity is carried by a COMPOSITE encoding: each dataset gets a unique
# (colour, linestyle, marker) triple and same-coloured pairs always differ in
# both linestyle and marker.
HUES = ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7', '#e34948']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', 'h']
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'


def style_for(i):
    return dict(color=HUES[i % len(HUES)],
                linestyle='-' if i < len(HUES) else (0, (5, 2)),
                marker=MARKERS[i % len(MARKERS)])


# ---------------------------------------------------------------------------
# Pool scoring
# ---------------------------------------------------------------------------

def pool_stats(tree_root, dataset):
    """Return (feats, MI, K, DBF, LEN) arrays for every parseable pooled feature."""
    pool, path = load_pool(tree_root, dataset)
    if not pool:
        raise SystemExit(f'[lambda_simplicity] empty pool for {dataset} under {tree_root}')
    env = get_env(dataset)
    n_base, n_ops = len(env.features), len(env.ops)

    feats, mi, kk, dd, ll = [], [], [], [], []
    n_bad = 0
    for f, v in pool.items():
        try:
            trees = generate_tree_from_str([f], env)
            t = trees[0] if trees else None
            if t is None:
                n_bad += 1
                continue
            k = float(kext_fn(t, n_base, n_ops))
            d = float(dbf_fn(t))
        except Exception:
            n_bad += 1
            continue
        if not np.isfinite(k) or not np.isfinite(v):
            n_bad += 1
            continue
        feats.append(f)
        mi.append(float(v))
        kk.append(k)
        dd.append(d)
        ll.append(float(len(f.split(','))))
    return (feats, np.array(mi), np.array(kk), np.array(dd), np.array(ll),
            path, n_bad, n_base, n_ops)


def rank(MI, K, lam):
    """Pool ranking by composite S(f; lambda), ties broken by higher MI.

    The tie-break matters only at lambda = 0, where every K-minimal feature
    scores exactly 1.0; without it the "simplest" pick would be whichever
    K-minimal feature happened to be enumerated first. np.lexsort is stable,
    so pool order is the final (deterministic) tie-break.
    """
    comp = composite_from_pool(MI, K, lam)
    return np.lexsort((-np.asarray(MI, float), -comp))


def sweep_dataset(MI, K, DBF, LEN, topk, lams=LAM_GRID):
    """For each lambda: top-k by composite, then mean K / DBF / MI / token length."""
    rows = []
    for lam in lams:
        order = rank(MI, K, lam)[:topk]
        rows.append(dict(lam=lam,
                         n_selected=int(len(order)),
                         mean_K=float(K[order].mean()),
                         mean_DBF=float(DBF[order].mean()),
                         mean_MI=float(MI[order].mean()),
                         mean_len=float(LEN[order].mean()),
                         max_K=float(K[order].max()),
                         min_K=float(K[order].min())))
    return rows


def top1(feats, MI, K, lam):
    i = int(rank(MI, K, lam)[0])
    return feats[i], float(K[i]), float(MI[i])


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(results, out_pdf, out_png):
    plt.rcParams.update({
        'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
        'axes.edgecolor': TEXT_SECONDARY, 'axes.linewidth': 0.8,
        'xtick.color': TEXT_SECONDARY, 'ytick.color': TEXT_SECONDARY,
        'text.color': TEXT_PRIMARY, 'axes.labelcolor': TEXT_PRIMARY,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.9), sharex=True)
    lams = np.array(LAM_GRID)

    # Labels are split across two lines: at >=9 pt a single-line 40-character
    # rotated ylabel is taller than the panel and gets clipped.
    panels = [
        (axes[0, 0], 'mean_K',   'mean description\ncomplexity $K$ (bits)',    False, '(a)'),
        (axes[0, 1], 'mean_DBF', 'mean distinct base\nfeatures (DBF)',         False, '(b)'),
        (axes[1, 0], 'mean_K',   'mean description complexity $K$,\n'
                                 r'normalized to $\lambda=1$',                 True,  '(c)'),
        (axes[1, 1], 'mean_DBF', 'mean distinct base features (DBF),\n'
                                 r'normalized to $\lambda=1$',                 True,  '(d)'),
    ]

    handles, labels = [], []
    for ax, key, ylab, norm, tag in panels:
        for i, (disp, ds) in enumerate(DATASETS):
            y = np.array([r[key] for r in results[ds]['sweep']], float)
            if norm:
                ref = y[-1]
                y = y / ref if abs(ref) > 1e-12 else y
            st = style_for(i)
            ln, = ax.plot(lams, y, linewidth=1.6, markersize=4.2,
                          markeredgewidth=0.0, alpha=0.95, **st)
            if ax is axes[0, 0]:
                handles.append(ln)
                labels.append(disp)
        ax.set_ylabel(ylab, labelpad=3)
        ax.grid(True, linewidth=0.4, alpha=0.35, color=TEXT_SECONDARY)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        # park the panel tag in whichever corner the curves leave empty
        tx, ty, ha, va = (0.985, 0.03, 'right', 'bottom') if norm \
            else (0.015, 0.965, 'left', 'top')
        ax.text(tx, ty, tag, transform=ax.transAxes, ha=ha, va=va,
                fontsize=9, fontweight='bold', color=TEXT_PRIMARY)
        if norm:
            ax.axhline(1.0, color=TEXT_SECONDARY, linewidth=0.7,
                       linestyle=':', alpha=0.7, zorder=0)
    for ax in axes[1, :]:
        ax.set_xlabel(r'trade-off parameter $\lambda$')
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.015), handlelength=2.4, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

def _tex_escape(s):
    for a, b in (('\\', r'\textbackslash{}'), ('_', r'\_'), ('%', r'\%'),
                 ('&', r'\&'), ('#', r'\#'), ('$', r'\$')):
        s = s.replace(a, b)
    return s


def macro_latex(macro):
    """Compact aggregate table: what lambda buys, averaged over the 10 datasets."""
    k1, m1 = macro[-1][1], macro[-1][3]
    lines = [
        r'% Generated by scripts/lambda_simplicity.py -- do not edit by hand.',
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Effect of the trade-off parameter $\lambda$ on the top-16 selected'
        r' features, averaged over the 10 datasets of the main comparison. Lowering'
        r' $\lambda$ buys large, monotone reductions in description complexity for a'
        f' comparatively small loss of mutual information: at $\\lambda=0.5$ the'
        f' selected features cost {100*macro[5][1]/k1:.1f}\\% of the $\\lambda=1$'
        f' complexity while retaining {100*macro[5][3]/m1:.1f}\\% of its MI.}}',
        r'\label{tab:lambda_sweep}',
        r'\small',
        r'\begin{tabular}{cccccc}',
        r'\toprule',
        r'$\lambda$ & mean $K$ & \%\,of $K_{\lambda=1}$ & mean DBF & mean MI'
        r' & \%\,MI retained \\',
        r'\midrule',
    ]
    for lam, mk, md, mm, _ml in macro:
        lines.append(f'{lam:.1f} & {mk:.2f} & {100*mk/k1:.1f} & {md:.2f} & '
                     f'{mm:.4f} & {100*mm/m1:.1f} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def qualitative_latex(qual_rows):
    disp = {d: n for n, d in DATASETS}
    lines = [
        r'% Generated by scripts/lambda_simplicity.py -- do not edit by hand.',
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Top-1 B-MAFE feature selected from the \emph{same} evaluated pool as the'
        r' trade-off parameter $\lambda$ is swept. Lowering $\lambda$ moves the selection from'
        r' high-MI compound expressions to short, few-variable ones: the expressions get'
        r' visibly simpler while retaining a usable share of the mutual information.}',
        r'\label{tab:lambda_qualitative}',
        r'\small',
        r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{llp{0.44\columnwidth}rr}',
        r'\toprule',
        r'Dataset & $\lambda$ & Top-1 selected feature & $K$ (bits) & MI \\',
        r'\midrule',
    ]
    for j, ds in enumerate(QUAL_DATASETS):
        rows = [r for r in qual_rows if r['dataset'] == ds]
        for i, r in enumerate(rows):
            name = _tex_escape(disp.get(ds, ds)) if i == 0 else ''
            lines.append(f"{name} & {r['lam']:.2f} & ${r['mathtext']}$ & "
                         f"{r['K']:.1f} & {r['MI']:.3f} \\\\")
        lines.append(r'\midrule' if j < len(QUAL_DATASETS) - 1 else r'\bottomrule')
    lines += [r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tree', default='outputs_num')
    ap.add_argument('--topk', type=int, default=16)
    ap.add_argument('--csv_root', default=CSV_ROOT)
    ap.add_argument('--plot_root', default=PLOT_ROOT)
    args = ap.parse_args()

    os.makedirs(args.csv_root, exist_ok=True)
    FIELDS = ['lam', 'n_selected', 'mean_K', 'mean_DBF', 'mean_MI', 'mean_len',
              'min_K', 'max_K']

    results = {}
    print(f'=== lambda sweep (tree={args.tree}, top-k={args.topk}, '
          f'grid={LAM_GRID[0]}..{LAM_GRID[-1]} step 0.1) ===')
    for disp, ds in DATASETS:
        feats, MI, K, DBF, LEN, path, n_bad, n_base, n_ops = pool_stats(args.tree, ds)
        sw = sweep_dataset(MI, K, DBF, LEN, args.topk)
        rho = spearmanr(LAM_GRID, [r['mean_K'] for r in sw]).correlation
        kvals = [r['mean_K'] for r in sw]
        dvals = [r['mean_DBF'] for r in sw]
        mono_k = all(kvals[i + 1] >= kvals[i] - 1e-9 for i in range(len(kvals) - 1))
        strict_k = all(kvals[i + 1] > kvals[i] + 1e-9 for i in range(len(kvals) - 1))
        mono_d = all(dvals[i + 1] >= dvals[i] - 1e-9 for i in range(len(dvals) - 1))
        results[ds] = dict(disp=disp, sweep=sw, rho=float(rho), mono_k=mono_k,
                           strict_k=strict_k, mono_d=mono_d, n_pool=len(feats),
                           n_bad=n_bad, path=path,
                           feats=feats, MI=MI, K=K, n_base=n_base, n_ops=n_ops)

        with open(os.path.join(args.csv_root, f'{ds}.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for r in sw:
                w.writerow({k: r[k] for k in FIELDS})

        k0, k1 = sw[0]['mean_K'], sw[-1]['mean_K']
        d0, d1 = sw[0]['mean_DBF'], sw[-1]['mean_DBF']
        l0, l1 = sw[0]['mean_len'], sw[-1]['mean_len']
        m0, m1 = sw[0]['mean_MI'], sw[-1]['mean_MI']
        print(f'  {disp:20s} pool={len(feats):5d} (skipped {n_bad:3d})  '
              f'K {k0:6.2f}->{k1:6.2f} ({k1/max(k0,1e-9):5.2f}x)  '
              f'DBF {d0:4.2f}->{d1:4.2f}  len {l0:4.2f}->{l1:5.2f}  '
              f'MI {m0:.4f}->{m1:.4f}  rho(lam,K)={rho:+.3f}  '
              f'mono_K={"Y" if mono_k else "n"}')

    # ---- aggregate long CSV -------------------------------------------------
    agg = os.path.join(args.csv_root, 'lambda_sweep_all.csv')
    with open(agg, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'display'] + FIELDS)
        for disp, ds in DATASETS:
            for r in results[ds]['sweep']:
                w.writerow([ds, disp] + [r[k] for k in FIELDS])

    # ---- aggregate mean-over-datasets row ----------------------------------
    print('\n=== mean over the 10 datasets ===')
    print(f"  {'lambda':>6} {'mean K':>9} {'mean DBF':>9} {'mean MI':>9} {'mean len':>9}")
    macro = []
    for li, lam in enumerate(LAM_GRID):
        mk = float(np.mean([results[d]['sweep'][li]['mean_K'] for _, d in DATASETS]))
        md = float(np.mean([results[d]['sweep'][li]['mean_DBF'] for _, d in DATASETS]))
        mm = float(np.mean([results[d]['sweep'][li]['mean_MI'] for _, d in DATASETS]))
        ml = float(np.mean([results[d]['sweep'][li]['mean_len'] for _, d in DATASETS]))
        macro.append((lam, mk, md, mm, ml))
        print(f'  {lam:6.1f} {mk:9.3f} {md:9.3f} {mm:9.4f} {ml:9.3f}')

    # ---- per-dataset summary CSV -------------------------------------------
    summ = os.path.join(args.csv_root, 'lambda_sweep_summary.csv')
    with open(summ, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'display', 'pool_size', 'n_base', 'n_ops',
                    'K_lam0', 'K_lam1', 'K_ratio', 'DBF_lam0', 'DBF_lam1',
                    'len_lam0', 'len_lam1', 'MI_lam0', 'MI_lam1',
                    'spearman_lam_K', 'monotone_K', 'strict_monotone_K', 'monotone_DBF'])
        for disp, ds in DATASETS:
            R = results[ds]; sw = R['sweep']
            w.writerow([ds, disp, R['n_pool'], R['n_base'], R['n_ops'],
                        f"{sw[0]['mean_K']:.4f}", f"{sw[-1]['mean_K']:.4f}",
                        f"{sw[-1]['mean_K']/max(sw[0]['mean_K'],1e-9):.4f}",
                        f"{sw[0]['mean_DBF']:.4f}", f"{sw[-1]['mean_DBF']:.4f}",
                        f"{sw[0]['mean_len']:.4f}", f"{sw[-1]['mean_len']:.4f}",
                        f"{sw[0]['mean_MI']:.6f}", f"{sw[-1]['mean_MI']:.6f}",
                        f"{R['rho']:.4f}", int(R['mono_k']), int(R['strict_k']),
                        int(R['mono_d'])])
    with open(os.path.join(args.csv_root, 'lambda_sweep_macro.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['lam', 'macro_mean_K', 'macro_mean_DBF', 'macro_mean_MI', 'macro_mean_len'])
        for row in macro:
            w.writerow([f'{row[0]:.1f}'] + [f'{v:.6f}' for v in row[1:]])

    n_mono = sum(results[d]['mono_k'] for _, d in DATASETS)
    n_strict = sum(results[d]['strict_k'] for _, d in DATASETS)
    n_mono_d = sum(results[d]['mono_d'] for _, d in DATASETS)
    rhos = [results[d]['rho'] for _, d in DATASETS]
    print(f'\n  Spearman(lambda, mean K): min {min(rhos):+.3f}  median '
          f'{float(np.median(rhos)):+.3f}  max {max(rhos):+.3f}  '
          f'(= +1.000 on {sum(r > 0.9999 for r in rhos)}/10)')
    print(f'  mean K non-decreasing in lambda on {n_mono}/10 datasets '
          f'(strictly increasing on {n_strict}/10)')
    print(f'  mean DBF non-decreasing in lambda on {n_mono_d}/10 datasets')

    # ---- qualitative table --------------------------------------------------
    print('\n=== qualitative: top-1 selected feature vs lambda ===')
    qual_rows = []
    for ds in QUAL_DATASETS:
        R = results[ds]
        names = NAMES[ds]
        print(f'\n  --- {R["disp"]} ---')
        print(f'  {"lambda":>6}  {"K":>6}  {"MI":>7}   top-1 expression')
        for lam in QUAL_LAMS:
            f, k, mi = top1(R['feats'], R['MI'], R['K'], lam)
            infix = to_infix(f, names)
            qual_rows.append(dict(dataset=ds, display=R['disp'], lam=lam,
                                  postfix=f, infix=infix,
                                  mathtext=to_mathtext(f, names),
                                  K=k, MI=mi, n_tokens=len(f.split(','))))
            print(f'  {lam:6.2f}  {k:6.2f}  {mi:7.4f}   {infix}')

    with open(os.path.join(args.csv_root, 'qualitative_top1.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['dataset', 'display', 'lam', 'K', 'MI',
                                           'n_tokens', 'infix', 'postfix'])
        w.writeheader()
        for r in qual_rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    tex = qualitative_latex(qual_rows)
    tex_path = os.path.join(args.csv_root, 'qualitative_top1.tex')
    with open(tex_path, 'w') as fh:
        fh.write(tex + '\n')
    mtex = macro_latex(macro)
    mtex_path = os.path.join(args.csv_root, 'lambda_sweep_macro.tex')
    with open(mtex_path, 'w') as fh:
        fh.write(mtex + '\n')
    print('\n=== LaTeX: qualitative table (also written to ' + tex_path + ') ===')
    print(tex)
    print('\n=== LaTeX: aggregate table (also written to ' + mtex_path + ') ===')
    print(mtex)

    # ---- figure -------------------------------------------------------------
    out_pdf = os.path.join(args.plot_root, 'lambda_simplicity.pdf')
    out_png = os.path.join(args.plot_root, 'lambda_simplicity.png')
    make_figure(results, out_pdf, out_png)
    print(f'\n[saved] {agg}')
    print(f'[saved] {summ}')
    print(f'[saved] {out_pdf}')
    print(f'[saved] {out_png}')
    print('\nCAVEAT: only lambda in {0.0, 0.5, 1.0} were ever SEARCHED. This sweep '
          'is a selection-only re-ranking of one fixed evaluated pool (the '
          'lambda=0.5 run), so it isolates the effect of lambda on SELECTION, '
          'not on exploration.')


if __name__ == '__main__':
    main()
