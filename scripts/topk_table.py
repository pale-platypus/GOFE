"""Recompute paper Table II with proper TOP-K aggregation (not top-1 argmax-MI).

Why this script exists
----------------------
``scripts/build_paper_table.py`` reports, for each (dataset, method) cell, the
metrics of a SINGLE feature: the argmax-MI feature among the method's evaluated
features. Two problems:

  1. A max over a set is not a summary of the set. It rewards a method for one
     lucky feature and says nothing about the other k-1 it also asks the user
     to accept.
  2. k is UNEQUAL across methods (AutoFeat returns 3-16 features, E-AFE 5-16,
     everyone else 16). A max-based statistic mechanically favours whoever
     evaluated MORE features, because max over a larger sample of the same
     distribution is larger in expectation.

This script fixes both:
  (a) NATIVE-k   : every method aggregated over its own available features
                   (capped at 16).
  (b) MATCHED-k  : k* = min over methods of #features contributed on that
                   dataset; every method aggregated over exactly its top-k*
                   features. Removes the unequal-k bias entirely.
  (c) PADDED-k16 : every method aggregated over exactly 16 features; methods
                   that contributed fewer are padded with features drawn at
                   random (fixed seed, no replacement, deduplicated by
                   feat_str) from the Random baseline's pool — an unbiased
                   uniform sample of valid expressions from the same grammar.
                   Semantics: "you asked for 16; whatever the method did not
                   supply is chance." Methods with zero features stay '--'
                   (pure filler would misreport Random's numbers as theirs).

Definitions
-----------------------------------------
  mi_base   = max mi over the 'base_features' rows (best single raw column)
  kext_base = kext of a bare base leaf = log2(n_base)   [uniform across bases]

  For a method's selected set F = its top-k features ranked by mi descending:
    MI_delta_mean = mean_{f in F} (mi(f)   - mi_base)  / mi_base
    K_delta_mean  = mean_{f in F} (kext(f) - kext_base)/ kext_base
    S_mean        = mean_{f in F} composite_lam<L>(f)
    MI_delta_kth  = (mi of the k-th / worst-ranked selected feature - mi_base)/mi_base
    S_kth         = composite of the k-th ranked feature when ranked BY composite

Lambda handling (fairness)
--------------------------
Only B-MAFE's feature SET depends on lambda; the baselines' sets are
lambda-independent (read from lam0.5). Therefore "best lambda per cell" tunes
B-MAFE and nobody else. PRIMARY reported number for B-MAFE is FIXED lam=0.5.
argmax-lambda is reported separately so the difference is visible.

Usage
-----
  python scripts/topk_table.py [--tree outputs_num]
Writes CSVs to outputs_num/quality_compare/topk_tables/ and prints everything.
"""
from __future__ import annotations

import os
import sys
import argparse
import functools

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants
# Dataset list + display names: copied verbatim from build_paper_table.ROWS so
# the two tables are line-for-line comparable.
ROWS = [
    ('Pima',              'pima',             'UCI',    'C'),
    ('German Credit',     'german',           'UCI',    'C'),
    ('Wine Quality (Red)','winequality-red',  'UCI',    'C'),
    ('Spambase',          'spambase',         'UCI',    'C'),
    ('Amazon Employee',   'amazon',           'UCI',    'C'),
    ('Credit Default',    'credit_default',   'UCI',    'R'),
    ('Housing Boston',    'housing',          'UCI',    'R'),
    ('Airfoil',           'airfoil',          'UCI',    'R'),
    ('Openml_620',        '620',              'OpenML', 'R'),
    ('Openml_637',        '637',              'OpenML', 'R'),
]

# (display, raw key in per_feature.csv, is_lam_dependent, training_metric)
METHODS_BY_TREE = {
    'outputs_num': [
        ('OpenFE',     'openfe',           False, 'mi'),
        ('AutoFeat',   'autofeat',         False, 'mi'),
        ('DIFER',      'infer',            False, 'mi'),
        ('E-AFE',      'eafe',             False, 'mi'),
        ('Random',     'random',           False, 'mi'),
        ('B-MAFE(MI)', 'lsbo',             True,  'mi'),
        ('B-MAFE(P)',  'lsbo_pearson_mdl', True,  'pearson'),
        ('B-MAFE(F)',  'lsbo_fisher_mdl',  True,  'fisher'),
    ],
    'outputs_numcat': [
        ('OpenFE',     'openfe',      False, 'mi'),
        ('AutoFeat',   'autofeat',    False, 'mi'),
        ('DIFER',      'infer',       False, 'mi'),
        ('E-AFE',      'eafe',        False, 'mi'),
        ('Random',     'random',      False, 'mi'),
        ('B-MAFE(MI)', 'lsbo',        True,  'mi'),
        ('B-MAFE(F)',  'lsbo_fisher', True,  'fisher'),
    ],
}

# Methods that appear in the paper's Table II proper (5 columns).
PAPER_METHODS = ['OpenFE', 'AutoFeat', 'DIFER', 'E-AFE', 'B-MAFE(MI)']

LAMS = [0.0, 0.5, 1.0]
BASE_LAM = 0.5          # lambda-independent methods are read from this pool
PRIMARY_LAM = 0.5       # B-MAFE primary (fixed) lambda
K_CAP = 16              # hard cap on native k

BOOTSTRAP_N = 1000
BOOTSTRAP_ALPHA = 0.10  # 90% CI, same as build_paper_table

TREE = 'outputs_num'
METHODS = METHODS_BY_TREE[TREE]

_COMPOSITE_PREFIX = {'mi': 'composite',
                     'pearson': 'composite_pearson',
                     'fisher': 'composite_fisher'}


def comp_col(training_metric, lam):
    return f'{_COMPOSITE_PREFIX.get(training_metric, "composite")}_lam{lam}'


# ---------------------------------------------------------------- IO layer
@functools.lru_cache(maxsize=512)
def per_feature_for(short, lam):
    p = f'{TREE}/quality_compare/{short}/lam{lam}/per_feature.csv'
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


@functools.lru_cache(maxsize=512)
def method_features(short, raw_m, lam):
    """All rows of per_feature.csv for (dataset, method, lam), sorted by mi
    descending with ties broken by smallest feat_idx. NaN-mi rows dropped."""
    df = per_feature_for(short, lam)
    if df is None:
        return None
    s = df[df['method'] == raw_m]
    if s.empty:
        return None
    s = s.dropna(subset=['mi'])
    if s.empty:
        return None
    return s.sort_values(['mi', 'feat_idx'],
                         ascending=[False, True]).reset_index(drop=True)


def padded_features(short, raw_m, lam):
    """method_features padded to K_CAP with rows drawn from the Random pool.

    Draw order is a fixed-seed permutation (crc32 of dataset|method), without
    replacement, skipping feat_strs the method already selected. Returns
    (padded_df sorted by mi desc, n_own, n_padded); (None, 0, 0) if the method
    contributed nothing.
    """
    import zlib
    own = method_features(short, raw_m, lam)
    if own is None:
        return None, 0, 0
    n_own = min(len(own), K_CAP)
    own = own.iloc[:n_own]
    if n_own >= K_CAP or raw_m == 'random':
        return own, n_own, 0
    pool = method_features(short, 'random', BASE_LAM)
    if pool is None:
        return own, n_own, 0
    have = set(own['feat_str'].astype(str))
    pool = pool[~pool['feat_str'].astype(str).isin(have)].reset_index(drop=True)
    rng = np.random.default_rng(zlib.crc32(f'{short}|{raw_m}'.encode()))
    order = rng.permutation(len(pool))[:K_CAP - n_own]
    pad = pool.iloc[order]
    out = pd.concat([own, pad], ignore_index=True)
    out = out.sort_values(['mi', 'feat_idx'],
                          ascending=[False, True]).reset_index(drop=True)
    return out, n_own, len(pad)


def base_anchors(short):
    """(mi_base, kext_base). mi_base = max mi over base_features rows;
    kext_base = kext of a bare leaf = log2(n_base) (uniform across base rows,
    so max == mean == min; we take the median for robustness)."""
    for lam in (BASE_LAM, 0.0, 1.0):
        b = method_features(short, 'base_features', lam)
        if b is None:
            continue
        mi_base = float(b['mi'].max())
        kext_base = float(np.median(b['kext'].to_numpy(dtype=np.float64)))
        return mi_base, kext_base
    return float('nan'), float('nan')


# ------------------------------------------------------------- aggregation
def topk_stats(feat_df, k, mi_base, kext_base, ccol):
    """Deterministic top-k statistics for one (method, lam) cell.

    feat_df must already be sorted by mi desc. Returns a dict of the five
    quantities plus the raw means used to build them.
    """
    out = dict(k_used=0, n_avail=0,
               mi_delta_mean=np.nan, k_delta_mean=np.nan, s_mean=np.nan,
               mi_delta_kth=np.nan, s_kth=np.nan,
               mi_mean_raw=np.nan, kext_mean_raw=np.nan)
    if feat_df is None or feat_df.empty or k <= 0:
        return out
    n = len(feat_df)
    kk = int(min(k, n))
    sel = feat_df.iloc[:kk]
    out['k_used'] = kk
    out['n_avail'] = n

    mi = sel['mi'].to_numpy(dtype=np.float64)
    kext = sel['kext'].to_numpy(dtype=np.float64)
    out['mi_mean_raw'] = float(np.nanmean(mi))
    out['kext_mean_raw'] = float(np.nanmean(kext))
    if np.isfinite(mi_base) and mi_base > 0:
        out['mi_delta_mean'] = (out['mi_mean_raw'] - mi_base) / mi_base
        out['mi_delta_kth'] = (float(mi[-1]) - mi_base) / mi_base
    if np.isfinite(kext_base) and kext_base > 0:
        out['k_delta_mean'] = (out['kext_mean_raw'] - kext_base) / kext_base

    if ccol in feat_df.columns:
        out['s_mean'] = float(np.nanmean(sel[ccol].to_numpy(dtype=np.float64)))
        # S_kth: rank the FULL set by composite, take the k-th (worst of top-k).
        cs = np.sort(feat_df[ccol].to_numpy(dtype=np.float64))[::-1]
        cs = cs[np.isfinite(cs)]
        if cs.size:
            out['s_kth'] = float(cs[min(kk, cs.size) - 1])
    return out


def bootstrap_topk(feat_df, k, mi_base, kext_base, ccol,
                   n_boot=BOOTSTRAP_N, alpha=BOOTSTRAP_ALPHA, seed=0):
    """Percentile CIs on the top-k statistics.

    Uncertainty model (the top-k analogue of build_paper_table's bootstrap):
    resample the method's n evaluated features with replacement, re-rank the
    resample by mi, take its top-k, and recompute the statistics. Answers
    "how much would this cell move if the search had returned a slightly
    different feature set of the same size?".
    """
    if feat_df is None or len(feat_df) < 2 or k <= 0:
        return {}
    n = len(feat_df)
    kk = int(min(k, n))
    mi = feat_df['mi'].to_numpy(dtype=np.float64)
    kext = feat_df['kext'].to_numpy(dtype=np.float64)
    comp = (feat_df[ccol].to_numpy(dtype=np.float64)
            if ccol in feat_df.columns else np.full(n, np.nan))
    rng = np.random.default_rng(seed)

    mi_m = np.empty(n_boot); ke_m = np.empty(n_boot)
    s_m = np.empty(n_boot); mi_k = np.empty(n_boot); s_k = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        order = np.argsort(-mi[idx], kind='stable')[:kk]
        sel = idx[order]
        mi_m[b] = np.nanmean(mi[sel])
        ke_m[b] = np.nanmean(kext[sel])
        s_m[b] = np.nanmean(comp[sel])
        mi_k[b] = mi[sel][-1]
        cs = np.sort(comp[idx])[::-1]
        s_k[b] = cs[kk - 1]

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def _pct(a):
        f = a[np.isfinite(a)]
        if f.size == 0:
            return (np.nan, np.nan)
        return (float(np.percentile(f, lo_q)), float(np.percentile(f, hi_q)))

    def _ratio(pair, base):
        if not np.isfinite(base) or base <= 0:
            return (np.nan, np.nan)
        return ((pair[0] - base) / base, (pair[1] - base) / base)

    return {
        'mi_delta_mean': _ratio(_pct(mi_m), mi_base),
        'k_delta_mean': _ratio(_pct(ke_m), kext_base),
        's_mean': _pct(s_m),
        'mi_delta_kth': _ratio(_pct(mi_k), mi_base),
        's_kth': _pct(s_k),
    }


def cell(short, raw_m, is_lam_dep, tmetric, k_mode, lam_mode):
    """Compute one (dataset, method) cell.

    k_mode  : 'native' -> min(n_avail, K_CAP);  int -> that exact k;
              'padded16' -> exactly K_CAP, short sets padded with seeded
              random draws from the Random pool (see padded_features).
    lam_mode: 'fixed'     -> PRIMARY_LAM for lam-dependent methods;
              'argmax_s'  -> lambda maximizing S_mean;
              'argmax_mi' -> lambda maximizing MI_delta_mean (the criterion
                             build_paper_table.py uses, LAM_PICK='argmax_mi_max').
    Lambda-independent methods always read BASE_LAM.
    """
    mi_base, kext_base = base_anchors(short)
    lams = (LAMS if (is_lam_dep and lam_mode.startswith('argmax'))
            else ([PRIMARY_LAM] if is_lam_dep else [BASE_LAM]))
    crit = 's_mean' if lam_mode != 'argmax_mi' else 'mi_delta_mean'

    best = None
    for lam in lams:
        n_own = n_pad = None
        if k_mode == 'padded16':
            fdf, n_own, n_pad = padded_features(short, raw_m, lam)
            k = K_CAP
        else:
            fdf = method_features(short, raw_m, lam)
        if fdf is None:
            continue
        ccol = comp_col(tmetric, lam)
        if k_mode == 'native':
            k = min(len(fdf), K_CAP)
        elif k_mode != 'padded16':
            k = int(k_mode)
        st = topk_stats(fdf, k, mi_base, kext_base, ccol)
        if n_own is not None:
            st['n_own'], st['n_padded'] = n_own, n_pad
        st['lam_used'] = lam
        st['_fdf'] = fdf
        st['_ccol'] = ccol
        st['_k'] = k
        if best is None or (np.isfinite(st[crit]) and
                            (not np.isfinite(best[crit]) or
                             st[crit] > best[crit])):
            best = st
    if best is None:
        return None
    ci = bootstrap_topk(best['_fdf'], best['_k'], mi_base, kext_base,
                        best['_ccol'])
    best['ci'] = ci
    best['mi_base'] = mi_base
    best['kext_base'] = kext_base
    return best


# --------------------------------------------------------- old top-1 logic
def old_top1(short, raw_m, is_lam_dep, tmetric):
    """Replicate build_paper_table.py's current cell exactly:
    LAM_PICK='argmax_mi_max' -> lambda maximizing the max-mi feature's mi;
    all three metrics come from that single argmax-MI feature."""
    mi_base, kext_base = base_anchors(short)
    lams = LAMS if is_lam_dep else [BASE_LAM]
    best_lam, best_mi, best_row = None, -np.inf, None
    for lam in lams:
        fdf = method_features(short, raw_m, lam)
        if fdf is None:
            continue
        r = fdf.iloc[0]                      # already sorted by mi desc
        if float(r['mi']) > best_mi:
            best_lam, best_mi, best_row = lam, float(r['mi']), r
    if best_row is None:
        return None
    ccol = comp_col(tmetric, best_lam)
    return {
        'lam_used': best_lam,
        'mi_delta_top1': ((best_mi - mi_base) / mi_base
                          if np.isfinite(mi_base) and mi_base > 0 else np.nan),
        'k_delta_top1': ((float(best_row['kext']) - kext_base) / kext_base
                         if np.isfinite(kext_base) and kext_base > 0 else np.nan),
        's_top1': (float(best_row[ccol]) if ccol in best_row.index
                   and pd.notna(best_row[ccol]) else np.nan),
    }


# ------------------------------------------------------------- formatting
def fmt_pm(v, ci, signed=True, nd=3):
    if v is None or not np.isfinite(v):
        return '--'
    vs = (f'{v:+.{nd}f}' if signed else f'{v:.{nd}f}')
    if ci is None or not np.isfinite(ci[0]) or not np.isfinite(ci[1]):
        return vs
    hw = (ci[1] - ci[0]) / 2.0
    return f'{vs} +/- {hw:.{nd}f}'


def latex_pm(v, ci, signed=True, nd=3):
    if v is None or not np.isfinite(v):
        return '--'
    vs = (f'{v:+.{nd}f}' if signed else f'{v:.{nd}f}')
    if ci is None or not np.isfinite(ci[0]) or not np.isfinite(ci[1]):
        return vs
    hw = (ci[1] - ci[0]) / 2.0
    return f'{vs} $\\pm$ {hw:.{nd}f}'


def rank_marks(values, higher_better):
    """Return list of '' / 'best' / 'second' for a metric row."""
    order = [(i, v) for i, v in enumerate(values) if v is not None and np.isfinite(v)]
    if not order:
        return [''] * len(values)
    order.sort(key=lambda kv: kv[1], reverse=higher_better)
    marks = [''] * len(values)
    marks[order[0][0]] = 'best'
    if len(order) > 1:
        marks[order[1][0]] = 'second'
    return marks


# ------------------------------------------------------------------ build
def build_table(k_mode, lam_mode, methods, kstars=None):
    """Returns a long DataFrame with one row per (dataset, method)."""
    recs = []
    for nice, short, src, task in ROWS:
        k = kstars[short] if (k_mode == 'matched' and kstars) else k_mode
        for disp, raw, is_lam, tm in methods:
            c = cell(short, raw, is_lam, tm, k, lam_mode)
            if c is None:
                recs.append(dict(Dataset=nice, Dataset_key=short, Method=disp,
                                 Method_key=raw))
                continue
            ci = c['ci']
            recs.append(dict(
                Dataset=nice, Dataset_key=short, Method=disp, Method_key=raw,
                lam_used=c['lam_used'], k_used=c['k_used'], n_avail=c['n_avail'],
                n_own=c.get('n_own', np.nan), n_padded=c.get('n_padded', np.nan),
                mi_base=c['mi_base'], kext_base=c['kext_base'],
                MI_delta_mean=c['mi_delta_mean'],
                MI_delta_mean_lo=ci.get('mi_delta_mean', (np.nan, np.nan))[0],
                MI_delta_mean_hi=ci.get('mi_delta_mean', (np.nan, np.nan))[1],
                K_delta_mean=c['k_delta_mean'],
                K_delta_mean_lo=ci.get('k_delta_mean', (np.nan, np.nan))[0],
                K_delta_mean_hi=ci.get('k_delta_mean', (np.nan, np.nan))[1],
                S_mean=c['s_mean'],
                S_mean_lo=ci.get('s_mean', (np.nan, np.nan))[0],
                S_mean_hi=ci.get('s_mean', (np.nan, np.nan))[1],
                MI_delta_kth=c['mi_delta_kth'],
                MI_delta_kth_lo=ci.get('mi_delta_kth', (np.nan, np.nan))[0],
                MI_delta_kth_hi=ci.get('mi_delta_kth', (np.nan, np.nan))[1],
                S_kth=c['s_kth'],
                S_kth_lo=ci.get('s_kth', (np.nan, np.nan))[0],
                S_kth_hi=ci.get('s_kth', (np.nan, np.nan))[1],
                mi_mean_raw=c['mi_mean_raw'], kext_mean_raw=c['kext_mean_raw'],
            ))
    return pd.DataFrame(recs)


def compute_kstars(methods):
    """k* per dataset = min over methods of #features contributed (cap 16).
    Also returns which methods are k-limited (attain the min)."""
    ks, limiters, counts = {}, {}, {}
    for nice, short, src, task in ROWS:
        per = {}
        for disp, raw, is_lam, tm in methods:
            lam = PRIMARY_LAM if is_lam else BASE_LAM
            f = method_features(short, raw, lam)
            per[disp] = 0 if f is None else min(len(f), K_CAP)
        avail = {m: v for m, v in per.items() if v > 0}
        kmin = min(avail.values()) if avail else 0
        ks[short] = kmin
        limiters[short] = [m for m, v in avail.items() if v == kmin]
        counts[short] = per
    return ks, limiters, counts


METRIC_ROWS = [
    ('MI delta (mean top-k)', 'MI_delta_mean', True,  True),
    ('K delta (mean top-k)',  'K_delta_mean',  False, True),
    ('S (mean top-k)',        'S_mean',        True,  False),
]


def render_ascii(df, methods, title, kstars=None):
    """Print a three-rows-per-dataset ASCII table."""
    disp_names = [m[0] for m in methods]
    w, dw = 20, 26
    lines = []
    lines.append('=' * (dw + 24 + w * len(disp_names)))
    lines.append(title)
    lines.append('=' * (dw + 24 + w * len(disp_names)))
    hdr = f'{"Dataset":<{dw}}{"Metric":<24}' + ''.join(f'{d:>{w}}' for d in disp_names)
    lines.append(hdr)
    lines.append('-' * len(hdr))
    for nice, short, src, task in ROWS:
        sub = df[df.Dataset_key == short].set_index('Method')
        ktag = f' (k*={kstars[short]})' if kstars else ''
        first = True
        for label, col, higher, signed in METRIC_ROWS:
            vals = [sub[col].get(d, np.nan) if d in sub.index else np.nan
                    for d in disp_names]
            cis = []
            for d in disp_names:
                if d in sub.index:
                    cis.append((sub[f'{col}_lo'].get(d, np.nan),
                                sub[f'{col}_hi'].get(d, np.nan)))
                else:
                    cis.append((np.nan, np.nan))
            marks = rank_marks(vals, higher)
            cells = []
            for v, ci, mk in zip(vals, cis, marks):
                s = fmt_pm(v, ci, signed=signed)
                if mk == 'best':
                    s = '*' + s
                elif mk == 'second':
                    s = '~' + s
                cells.append(f'{s:>{w}}')
            name = (nice + ktag) if first else ''
            lines.append(f'{name:<{dw}}{label:<24}' + ''.join(cells))
            first = False
        lines.append('')
    # Average row
    lines.append(f'{"AVERAGE":<{dw}}{"":<24}')
    for label, col, higher, signed in METRIC_ROWS:
        vals, hws = [], []
        for d in disp_names:
            v = df[df.Method == d][col].to_numpy(dtype=np.float64)
            v = v[np.isfinite(v)]
            vals.append(float(v.mean()) if v.size else np.nan)
            if v.size >= 2:
                sem = float(v.std(ddof=1) / np.sqrt(v.size))
                hws.append(_tcrit(v.size) * sem)
            else:
                hws.append(np.nan)
        marks = rank_marks(vals, higher)
        cells = []
        for v, hw, mk in zip(vals, hws, marks):
            s = ('--' if not np.isfinite(v)
                 else (f'{v:+.3f}' if signed else f'{v:.3f}')
                 + ('' if not np.isfinite(hw) else f' +/- {hw:.3f}'))
            if mk == 'best':
                s = '*' + s
            elif mk == 'second':
                s = '~' + s
            cells.append(f'{s:>{w}}')
        lines.append(f'{"":<{dw}}{label:<24}' + ''.join(cells))
    lines.append('')
    lines.append('(* = best in row, ~ = second best; +/- is the 90% bootstrap '
                 'half-width, AVERAGE uses t*SEM across datasets)')
    return '\n'.join(lines)


def _tcrit(n):
    try:
        from scipy import stats as _sps
        return float(_sps.t.ppf(0.95, df=n - 1))
    except Exception:
        return 1.833 if n == 10 else 1.645


def render_latex(df, methods, caption, label, kstars=None):
    disp = [m[0] for m in methods]
    ncol = len(disp)
    out = []
    out.append('\\begin{table*}[t]')
    out.append('\\centering')
    out.append('\\caption{%s}' % caption)
    out.append('\\label{%s}' % label)
    out.append('\\setlength{\\tabcolsep}{4pt}')
    out.append('\\renewcommand{\\arraystretch}{1.15}')
    out.append('\\begin{tabular}{ll' + 'c' * ncol + '}')
    out.append('\\hline')
    out.append('Dataset & Metric & ' + ' & '.join(_tex_escape(d) for d in disp)
               + ' \\\\')
    out.append('\\hline')
    for nice, short, src, task in ROWS:
        sub = df[df.Dataset_key == short].set_index('Method')
        ktag = (' ($k^{*}$=%d)' % kstars[short]) if kstars else ''
        first = True
        for label_m, col, higher, signed in METRIC_ROWS:
            vals, cis = [], []
            for d in disp:
                if d in sub.index:
                    vals.append(float(sub[col].get(d, np.nan)))
                    cis.append((float(sub[f'{col}_lo'].get(d, np.nan)),
                                float(sub[f'{col}_hi'].get(d, np.nan))))
                else:
                    vals.append(np.nan)
                    cis.append((np.nan, np.nan))
            marks = rank_marks(vals, higher)
            cells = []
            for v, ci, mk in zip(vals, cis, marks):
                s = latex_pm(v, ci, signed=signed)
                if mk == 'best':
                    s = '\\textbf{%s}' % s
                elif mk == 'second':
                    s = '\\underline{%s}' % s
                cells.append(s)
            rowname = ('\\multirow{3}{*}{%s%s}' % (_tex_escape(nice), ktag)) if first else ''
            out.append(f'{rowname} & {_tex_metric(label_m)} & '
                       + ' & '.join(cells) + ' \\\\')
            first = False
        out.append('\\hline')
    # Average
    first = True
    for label_m, col, higher, signed in METRIC_ROWS:
        vals, hws = [], []
        for d in disp:
            v = df[df.Method == d][col].to_numpy(dtype=np.float64)
            v = v[np.isfinite(v)]
            vals.append(float(v.mean()) if v.size else np.nan)
            hws.append(_tcrit(v.size) * float(v.std(ddof=1) / np.sqrt(v.size))
                       if v.size >= 2 else np.nan)
        marks = rank_marks(vals, higher)
        cells = []
        for v, hw, mk in zip(vals, hws, marks):
            if not np.isfinite(v):
                s = '--'
            else:
                s = (f'{v:+.3f}' if signed else f'{v:.3f}')
                if np.isfinite(hw):
                    s += f' $\\pm$ {hw:.3f}'
            if mk == 'best':
                s = '\\textbf{%s}' % s
            elif mk == 'second':
                s = '\\underline{%s}' % s
            cells.append(s)
        rowname = '\\multirow{3}{*}{\\textbf{Average}}' if first else ''
        out.append(f'{rowname} & {_tex_metric(label_m)} & '
                   + ' & '.join(cells) + ' \\\\')
        first = False
    out.append('\\hline')
    out.append('\\end{tabular}')
    out.append('\\end{table*}')
    return '\n'.join(out)


def _tex_escape(s):
    """Escape LaTeX-special characters in dataset / method labels.
    Dataset keys like ``Openml_620`` would otherwise be a math-mode error."""
    return str(s).replace('\\', r'\textbackslash{}').replace('_', r'\_') \
                 .replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')


def _tex_metric(label):
    return {'MI delta (mean top-k)': '$\\overline{\\Delta \\mathrm{MI}}$ ($\\uparrow$)',
            'K delta (mean top-k)': '$\\overline{\\Delta K}$ ($\\downarrow$)',
            'S (mean top-k)': '$\\overline{S}$ ($\\uparrow$)'}.get(label, label)


# ------------------------------------------------------------------- main
def main(tree='outputs_num'):
    global TREE, METHODS
    TREE = tree
    METHODS = METHODS_BY_TREE[tree]
    outdir = f'{TREE}/quality_compare/topk_tables'
    os.makedirs(outdir, exist_ok=True)

    kstars, limiters, counts = compute_kstars(METHODS)

    print('#' * 100)
    print('TOP-K TABLE II  |  tree =', TREE)
    print('#' * 100)
    print()
    print('--- (b0) FEATURE COUNTS PER (dataset, method) and k* ---')
    cnt_rows = []
    hdr = f'{"Dataset":<20}' + ''.join(f'{m[0]:>14}' for m in METHODS) + f'{"k*":>6}  k-limited by'
    print(hdr)
    print('-' * len(hdr))
    for nice, short, src, task in ROWS:
        line = f'{nice:<20}' + ''.join(f'{counts[short][m[0]]:>14}' for m in METHODS)
        line += f'{kstars[short]:>6}  ' + ','.join(limiters[short])
        print(line)
        r = {'Dataset': nice, 'Dataset_key': short, 'k_star': kstars[short],
             'k_limited_by': ';'.join(limiters[short])}
        r.update({m[0]: counts[short][m[0]] for m in METHODS})
        cnt_rows.append(r)
    pd.DataFrame(cnt_rows).to_csv(f'{outdir}/kstar_per_dataset.csv', index=False)
    print()

    # ---- (a) NATIVE k, B-MAFE at fixed lam=0.5 (PRIMARY) ----
    nat_fixed = build_table('native', 'fixed', METHODS)
    nat_fixed.to_csv(f'{outdir}/topk_native_lam0.5.csv', index=False)
    print(render_ascii(nat_fixed, METHODS,
                       '(a) NATIVE-k TABLE  [each method over its own features, cap 16]  '
                       '| B-MAFE at FIXED lambda=0.5  << PRIMARY >>'))
    print()

    # ---- (a2) NATIVE k, B-MAFE at argmax lambda (two argmax criteria) ----
    nat_arg = build_table('native', 'argmax_s', METHODS)
    nat_arg.to_csv(f'{outdir}/topk_native_argmaxlam_S.csv', index=False)
    print(render_ascii(nat_arg, METHODS,
                       '(a2) NATIVE-k TABLE | B-MAFE at ARGMAX-lambda by S (tunes B-MAFE only '
                       '-- baselines are lambda-independent) << SECONDARY, SEE WARNING >>'))
    print()

    nat_argmi = build_table('native', 'argmax_mi', METHODS)
    nat_argmi.to_csv(f'{outdir}/topk_native_argmaxlam_MI.csv', index=False)
    print(render_ascii(nat_argmi, METHODS,
                       '(a3) NATIVE-k TABLE | B-MAFE at ARGMAX-lambda by MI '
                       '(= build_paper_table.py LAM_PICK convention) << SECONDARY >>'))
    print()

    # ---- (b) MATCHED k ----
    mat_fixed = build_table('matched', 'fixed', METHODS, kstars)
    mat_fixed.to_csv(f'{outdir}/topk_matched_lam0.5.csv', index=False)
    print(render_ascii(mat_fixed, METHODS,
                       '(b) MATCHED-k TABLE [every method over exactly its top-k* features] '
                       '| B-MAFE at FIXED lambda=0.5  << PRIMARY >>',
                       kstars=kstars))
    print()

    mat_arg = build_table('matched', 'argmax_s', METHODS, kstars)
    mat_arg.to_csv(f'{outdir}/topk_matched_argmaxlam_S.csv', index=False)
    print(render_ascii(mat_arg, METHODS,
                       '(b2) MATCHED-k TABLE | B-MAFE at ARGMAX-lambda by S << SECONDARY, SEE WARNING >>',
                       kstars=kstars))
    print()

    mat_argmi = build_table('matched', 'argmax_mi', METHODS, kstars)
    mat_argmi.to_csv(f'{outdir}/topk_matched_argmaxlam_MI.csv', index=False)
    print(render_ascii(mat_argmi, METHODS,
                       '(b4) MATCHED-k TABLE | B-MAFE at ARGMAX-lambda by MI << SECONDARY >>',
                       kstars=kstars))
    print()

    # ---- (b5) PADDED k = 16 ----
    pad_fixed = build_table('padded16', 'fixed', METHODS)
    pad_fixed.to_csv(f'{outdir}/topk_padded16_lam0.5.csv', index=False)
    print(render_ascii(pad_fixed, METHODS,
                       '(b5) PADDED-k16 TABLE [every method at exactly k=16; short sets '
                       'padded with seeded random draws from the Random pool] '
                       '| B-MAFE at FIXED lambda=0.5  << PRIMARY under the padding policy >>'))
    print()
    print('--- (b5b) padding composition per cell: own + random = 16 ---')
    for nice, short, src, task in ROWS:
        sub = pad_fixed[pad_fixed.Dataset_key == short]
        bits = []
        for r in sub.itertuples():
            if pd.notna(r.n_own):
                bits.append(f'{r.Method}={int(r.n_own)}+{int(r.n_padded)}')
        print(f'  {nice:<20} ' + ', '.join(bits))
    print()

    # ---- (c) B-MAFE lambda sensitivity ----
    print('--- (c) B-MAFE: FIXED lambda=0.5 vs ARGMAX-lambda (baselines unaffected) ---')
    print('WARNING: composite_lam<L> is a DIFFERENT objective at each L '
          '(L=0 scores simplicity only),')
    print('         so "argmax-lambda by S" is not a comparison of like with '
          'like -- it always lands on')
    print('         lambda=0.0, where S is near 1 by construction and MI '
          'collapses. Reported for completeness only.')
    lam_rows = []
    bm = [m for m in METHODS if m[0].startswith('B-MAFE')]
    hdr = (f'{"Dataset":<20}{"Method":<12}{"kmode":<9}'
           f'{"lamS*":>7}{"lamMI*":>7}{"MId@0.5":>10}{"MId@aS":>10}{"MId@aMI":>10}'
           f'{"Kd@0.5":>9}{"Kd@aS":>9}{"Kd@aMI":>9}'
           f'{"S@0.5":>8}{"S@aS":>8}{"S@aMI":>8}')
    print(hdr)
    print('-' * len(hdr))
    for kmode, dff, dfa, dfm in (('native', nat_fixed, nat_arg, nat_argmi),
                                 ('matched', mat_fixed, mat_arg, mat_argmi)):
        for nice, short, src, task in ROWS:
            for disp, raw, is_lam, tm in bm:
                f = dff[(dff.Dataset_key == short) & (dff.Method == disp)]
                a = dfa[(dfa.Dataset_key == short) & (dfa.Method == disp)]
                mm = dfm[(dfm.Dataset_key == short) & (dfm.Method == disp)]
                if f.empty or a.empty or mm.empty:
                    continue
                f, a, mm = f.iloc[0], a.iloc[0], mm.iloc[0]
                print(f'{nice:<20}{disp:<12}{kmode:<9}'
                      f'{a.lam_used:>7.1f}{mm.lam_used:>7.1f}'
                      f'{f.MI_delta_mean:>10.3f}{a.MI_delta_mean:>10.3f}{mm.MI_delta_mean:>10.3f}'
                      f'{f.K_delta_mean:>9.3f}{a.K_delta_mean:>9.3f}{mm.K_delta_mean:>9.3f}'
                      f'{f.S_mean:>8.3f}{a.S_mean:>8.3f}{mm.S_mean:>8.3f}')
                lam_rows.append(dict(Dataset=nice, Method=disp, k_mode=kmode,
                                     lam_argmax_S=a.lam_used,
                                     lam_argmax_MI=mm.lam_used,
                                     MI_delta_lam05=f.MI_delta_mean,
                                     MI_delta_argmaxS=a.MI_delta_mean,
                                     MI_delta_argmaxMI=mm.MI_delta_mean,
                                     K_delta_lam05=f.K_delta_mean,
                                     K_delta_argmaxS=a.K_delta_mean,
                                     K_delta_argmaxMI=mm.K_delta_mean,
                                     S_lam05=f.S_mean, S_argmaxS=a.S_mean,
                                     S_argmaxMI=mm.S_mean,
                                     S_gain_argmaxS=a.S_mean - f.S_mean))
    pd.DataFrame(lam_rows).to_csv(f'{outdir}/bmafe_lambda_compare.csv', index=False)
    print()

    # ---- (d) OLD top-1 vs NEW mean-top-k ----
    print('--- (d) OLD (top-1 argmax-MI, argmax-lam) vs NEW (mean over top-k) ---')
    old_rows = []
    hdr = (f'{"Dataset":<20}{"Method":<12}'
           f'{"MI_old":>10}{"MI_natk":>10}{"MI_mk":>10}'
           f'{"K_old":>9}{"K_natk":>9}{"K_mk":>9}'
           f'{"S_old":>8}{"S_natk":>8}{"S_mk":>8}')
    print(hdr)
    print('-' * len(hdr))
    for nice, short, src, task in ROWS:
        for disp, raw, is_lam, tm in METHODS:
            o = old_top1(short, raw, is_lam, tm)
            f = nat_fixed[(nat_fixed.Dataset_key == short) & (nat_fixed.Method == disp)]
            m = mat_fixed[(mat_fixed.Dataset_key == short) & (mat_fixed.Method == disp)]
            pq = pad_fixed[(pad_fixed.Dataset_key == short) & (pad_fixed.Method == disp)]
            if o is None or f.empty or m.empty:
                continue
            pv = pq.iloc[0] if not pq.empty else None
            f, m = f.iloc[0], m.iloc[0]
            print(f'{nice:<20}{disp:<12}'
                  f'{o["mi_delta_top1"]:>10.3f}{f.MI_delta_mean:>10.3f}{m.MI_delta_mean:>10.3f}'
                  f'{o["k_delta_top1"]:>9.3f}{f.K_delta_mean:>9.3f}{m.K_delta_mean:>9.3f}'
                  f'{o["s_top1"]:>8.3f}{f.S_mean:>8.3f}{m.S_mean:>8.3f}')
            old_rows.append(dict(
                Dataset=nice, Dataset_key=short, Method=disp, Method_key=raw,
                old_lam=o['lam_used'], k_star=kstars[short],
                k_native=int(f.k_used),
                OLD_MI_delta_top1=o['mi_delta_top1'],
                NEW_MI_delta_mean_nativek=f.MI_delta_mean,
                NEW_MI_delta_mean_matchedk=m.MI_delta_mean,
                NEW_MI_delta_mean_padded16=(pv.MI_delta_mean if pv is not None
                                            else np.nan),
                NEW_K_delta_mean_padded16=(pv.K_delta_mean if pv is not None
                                           else np.nan),
                NEW_S_mean_padded16=(pv.S_mean if pv is not None else np.nan),
                OLD_K_delta_top1=o['k_delta_top1'],
                NEW_K_delta_mean_nativek=f.K_delta_mean,
                NEW_K_delta_mean_matchedk=m.K_delta_mean,
                OLD_S_top1=o['s_top1'],
                NEW_S_mean_nativek=f.S_mean,
                NEW_S_mean_matchedk=m.S_mean,
                NEW_MI_delta_kth_nativek=f.MI_delta_kth,
                NEW_MI_delta_kth_matchedk=m.MI_delta_kth,
                NEW_S_kth_nativek=f.S_kth,
                NEW_S_kth_matchedk=m.S_kth,
            ))
    old_df = pd.DataFrame(old_rows)
    old_df.to_csv(f'{outdir}/old_vs_new.csv', index=False)
    print()
    print('  per-method mean shift across the 10 datasets (NEW matched-k minus OLD top-1):')
    g = old_df.groupby('Method')
    for disp, _, _, _ in METHODS:
        if disp not in g.groups:
            continue
        s = g.get_group(disp)
        print(f'    {disp:<12} dMI={s.NEW_MI_delta_mean_matchedk.mean() - s.OLD_MI_delta_top1.mean():+8.3f}'
              f'  dK={s.NEW_K_delta_mean_matchedk.mean() - s.OLD_K_delta_top1.mean():+8.3f}'
              f'  dS={s.NEW_S_mean_matchedk.mean() - s.OLD_S_top1.mean():+8.3f}')
    print()

    # ---- worst-of-top-k (Eq. 10) view ----
    print('--- (b3) MATCHED-k WORST-OF-TOP-K (Eq.10 quantities): MI_delta_kth and S_kth ---')
    hdr = f'{"Dataset":<20}{"Metric":<16}' + ''.join(f'{m[0]:>14}' for m in METHODS)
    print(hdr)
    print('-' * len(hdr))
    for nice, short, src, task in ROWS:
        sub = mat_fixed[mat_fixed.Dataset_key == short].set_index('Method')
        for lbl, col in (('MI_delta_kth', 'MI_delta_kth'), ('S_kth', 'S_kth')):
            cells = ''.join(f'{sub[col].get(m[0], np.nan):>14.3f}' for m in METHODS)
            print(f'{(nice if lbl.startswith("MI") else ""):<20}{lbl:<16}{cells}')
    print()

    # ---- (f) head-to-head: B-MAFE(MI) vs each baseline, per metric ----
    print('--- (f) HEAD-TO-HEAD: B-MAFE(MI) at lambda=0.5 vs each baseline '
          '(wins/ties/losses over 10 datasets) ---')
    h2h_rows = []
    for kmode, dfx in (('native', nat_fixed), ('matched', mat_fixed),
                       ('padded16', pad_fixed)):
        print(f'  [{kmode}-k]')
        for disp, raw, is_lam, tm in METHODS:
            if disp == 'B-MAFE(MI)':
                continue
            line = f'    vs {disp:<12}'
            for label, col, higher, _ in METRIC_ROWS:
                w = l = t = 0
                for nice, short, _, _ in ROWS:
                    a = dfx[(dfx.Dataset_key == short) & (dfx.Method == 'B-MAFE(MI)')][col]
                    b = dfx[(dfx.Dataset_key == short) & (dfx.Method == disp)][col]
                    if a.empty or b.empty:
                        continue
                    av, bv = float(a.iloc[0]), float(b.iloc[0])
                    if not (np.isfinite(av) and np.isfinite(bv)):
                        continue
                    d = av - bv if higher else bv - av
                    if abs(d) < 1e-9:
                        t += 1
                    elif d > 0:
                        w += 1
                    else:
                        l += 1
                line += f'  {label.split(" (")[0]}: {w}W-{t}T-{l}L'
                h2h_rows.append(dict(k_mode=kmode, opponent=disp,
                                     metric=col, wins=w, ties=t, losses=l))
            print(line)
    pd.DataFrame(h2h_rows).to_csv(f'{outdir}/head_to_head.csv', index=False)
    print()

    # ---- (e) LaTeX ----
    paper_methods = [m for m in METHODS if m[0] in PAPER_METHODS]
    tex_paper_matched = render_latex(
        mat_fixed, paper_methods,
        'Feature-quality comparison with matched-$k$ top-$k$ aggregation. '
        'For each dataset every method is summarised over exactly its top-$k^{*}$ '
        'features ranked by MI, where $k^{*}$ is the smallest number of features any '
        'method contributed on that dataset. $\\overline{\\Delta \\mathrm{MI}}$ and '
        '$\\overline{\\Delta K}$ are means of the per-feature relative gaps against the '
        'best raw feature and a bare base leaf ($K_{ext}=\\log_2 n$); $\\overline{S}$ is '
        'the mean bi-objective composite at $\\lambda=0.5$. '
        '$\\pm$ is a 90\\% bootstrap interval half-width over feature resamples; the '
        'Average row uses $t_{0.95}\\cdot\\mathrm{SEM}$ across the ten datasets. '
        'Best in bold, second best underlined.',
        'tab:quality_matchedk', kstars=kstars)
    tex_paper_native = render_latex(
        nat_fixed, paper_methods,
        'Feature-quality comparison with native-$k$ top-$k$ aggregation '
        '(each method summarised over all of its own features, capped at 16). '
        'Same statistics as Table~\\ref{tab:quality_matchedk}; note that $k$ differs '
        'across methods here, which favours methods that returned more features.',
        'tab:quality_nativek')
    tex_full_matched = render_latex(
        mat_fixed, METHODS,
        'Matched-$k$ feature-quality comparison, all methods and all B-MAFE variants.',
        'tab:quality_matchedk_full', kstars=kstars)
    tex_paper_padded = render_latex(
        pad_fixed, paper_methods,
        'Feature-quality comparison with padded top-$k$ aggregation at a '
        'common $k{=}16$. Methods that contributed fewer than 16 features are '
        'padded to 16 with features drawn at random (fixed seed, no '
        'replacement) from an unbiased pool of valid random expressions over '
        'the same grammar, so every cell summarises exactly 16 features. '
        'Statistics as in Table~\\ref{tab:quality_matchedk}. '
        'Best in bold, second best underlined.',
        'tab:quality_padded16')

    with open(f'{outdir}/table2_topk.tex', 'w', encoding='utf-8') as fh:
        fh.write('% ===== PRIMARY: padded-k16, B-MAFE at fixed lambda = 0.5 =====\n')
        fh.write(tex_paper_padded + '\n\n')
        fh.write('% ===== SECONDARY: matched-k, B-MAFE at fixed lambda = 0.5 =====\n')
        fh.write(tex_paper_matched + '\n\n')
        fh.write('% ===== SECONDARY: native-k, B-MAFE at fixed lambda = 0.5 =====\n')
        fh.write(tex_paper_native + '\n\n')
        fh.write('% ===== APPENDIX: matched-k, all methods / all B-MAFE variants =====\n')
        fh.write(tex_full_matched + '\n')

    print('--- (e) LaTeX (PRIMARY = matched-k, B-MAFE at lambda=0.5) ---')
    print(tex_paper_matched)
    print()
    print('--- (e2) LaTeX (native-k, B-MAFE at lambda=0.5) ---')
    print(tex_paper_native)
    print()
    print(f'wrote CSVs + tex to {outdir}/')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tree', default='outputs_num',
                   choices=['outputs_num', 'outputs_numcat'])
    a = p.parse_args()
    main(a.tree)
