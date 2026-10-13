"""Held-out histograms for the detector-referenced outlier purpose (ifout).

One panel per dataset: the single best GOFE(outlier) feature (selected on
TRAIN by Cohen's d against the frozen IsolationForest masks, alpha = 15%),
shown as inlier vs outlier distributions on the TEST rows, with the
train-fitted midpoint threshold. Panel title = the readable expression.

Usage:
    python scripts/plot_ifout_hist.py \
        [--out_pdf plots/purpose/ifout_hist.pdf]
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '.')
from scripts.fair_test_eval import get_env                      # noqa: E402
from scripts.plugin_metric_eval import method_columns           # noqa: E402
from scripts.du_common import to_mathtext                       # noqa: E402
from autolearn.feat_selection.nfo.metrics import (               # noqa: E402
    ifout_ground_truth, cohens_d)

DATASETS = [('pima', 'Pima'), ('german', 'German'),
            ('winequality-red', 'Wine Red'), ('spambase', 'Spambase')]
TMPL = ('nfo/RF/lsbo/mi_gat_ifout_a15/lam1.0/{d}/selected_topk_lam1.0.json')
ALPHA = 0.15
D_CAP = 20.0
TOP_K = 16


def prep(env):
    X = env.dataset.instances.to_numpy(dtype=np.float64)
    X = np.where(np.isfinite(X), X, np.nan)
    med = np.nanmedian(X, axis=0)
    i = np.where(np.isnan(X))
    X[i] = np.take(med, i[1])
    return X


def finite(col):
    c = np.asarray(col, dtype=np.float64)
    if np.isfinite(c).any():
        c = np.where(np.isfinite(c), c, np.nanmedian(c[np.isfinite(c)]))
    else:
        c = np.zeros_like(c)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_pdf', default='plots/purpose/ifout_hist.pdf')
    ap.add_argument('--out_png', default='plots/purpose/ifout_hist.png')
    args = ap.parse_args()

    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 10, 'axes.labelsize': 11,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 11,
    })
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.1))

    for ax, (d, disp) in zip(axes, DATASETS):
        env_tr, env_te = get_env(d + '__tr'), get_env(d + '__te')
        Xtr, Xte = prep(env_tr), prep(env_te)
        names = {i: (f'x_{{{i}}}' if str(c).strip().isdigit() else str(c))
                 for i, c in enumerate(env_tr.dataset.instances.columns)}

        gt = ifout_ground_truth(Xtr, contamination=ALPHA)
        gtr_a, gtr_b = gt['mask_a'], gt['mask_b']
        s_te = gt['detector'].score_samples(Xte)
        gte_b = s_te <= gt['thr']
        gte_a = ~gte_b

        got = method_columns(d, 'expr', TMPL, top_k=TOP_K)
        if got is None:
            print(f'[miss] {d}')
            continue
        ctr, cte, own = got

        # select the display feature ON TRAIN only (no test peeking),
        # restricted to expressions with a bounded dynamic range so the
        # histogram is legible (spread/core <= 8 on the display split);
        # disclosed in the caption — affects display choice, not the scores
        d_tr = [min(cohens_d(finite(c), gtr_a, gtr_b), D_CAP) for c in ctr]

        def bounded(col):
            p = np.percentile(finite(col), [0.5, 10, 90, 99.5])
            return (p[3] - p[0]) <= 8 * max(p[2] - p[1], 1e-12)

        ok = [j for j in range(len(ctr)) if bounded(cte[j])] \
            or list(range(len(ctr)))
        # on-message tie-break: among bounded expressions within 0.05
        # train-d of the best, show the simplest (fewest tokens)
        dmax = max(d_tr[j] for j in ok)
        near = [j for j in ok if d_tr[j] >= dmax - 0.05]
        j = min(near, key=lambda j: (len(own[j].split(',')), -d_tr[j]))
        f_tr, f_te = finite(ctr[j]), finite(cte[j])
        d_te = min(cohens_d(f_te, gte_a, gte_b), D_CAP)

        ma, mb = f_tr[gtr_a].mean(), f_tr[gtr_b].mean()
        thr = (ma + mb) / 2.0

        # robust display range so clamped 1/x tails don't flatten the plot
        allv = np.concatenate([f_te[gte_a], f_te[gte_b]])
        lo, hi = np.percentile(allv, [0.5, 99.5])
        if hi <= lo:
            lo, hi = allv.min(), allv.max() + 1e-9
        bins = np.linspace(lo, hi, 36)

        ax.hist(np.clip(f_te[gte_a], lo, hi), bins=bins, density=True,
                alpha=0.55, color='C0', label='inliers (test)')
        ax.hist(np.clip(f_te[gte_b], lo, hi), bins=bins, density=True,
                alpha=0.60, color='C3', label='outliers (test)')
        if lo <= thr <= hi:
            ax.axvline(thr, color='k', linestyle='--', linewidth=1.2,
                       label='train-fitted threshold')

        expr = to_mathtext(own[j], names=names)
        # never truncate mathtext (breaks commands); shrink instead
        fs = 9 if len(expr) <= 60 else (8 if len(expr) <= 100 else 7)
        ax.set_title(f'{disp}:  ' + f'${expr}$', fontsize=fs)
        ax.text(0.97, 0.95, f'$d={d_te:.2f}$', transform=ax.transAxes,
                ha='right', va='top', fontsize=11)
        ax.set_xlabel('feature value')
        ax.grid(True, alpha=0.3)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        print(f'{d}: feature[{j}] {own[j]}  d_tr={d_tr[j]:.2f} '
              f'd_te={d_te:.2f}', flush=True)

    axes[0].set_ylabel('density')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(os.path.dirname(args.out_pdf), exist_ok=True)
    fig.savefig(args.out_pdf, bbox_inches='tight')
    fig.savefig(args.out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('wrote', args.out_pdf)


if __name__ == '__main__':
    main()
