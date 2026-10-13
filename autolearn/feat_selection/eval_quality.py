"""Quality evaluation driver for the bi-objective B-MAFE pipeline.

Given one or more cached feature manifests (selected_topk_lam{λ}.json from
iter_train.py, or generic eval_manifest.json from the baselines), this script
produces, for each (dataset, method, λ) row:

  (1) Usefulness       — MI / Pearson / Fisher of selected features
  (2) Explainability   — Kext / DBF
  (3) Composite        — S(f; λ) = λ·MI_norm + (1-λ)·(1-Kext_norm), with MI and
                         Kext MinMax-normalized **across all features in the
                         eval call** (pooled across methods and λ) so the
                         composite is comparable cross-method. Reported at the
                         row's own λ and at the standard grid for sensitivity.
  (4) Downstream       — RF / LR / LGB / XGB CV on [base ∪ topk_derived],
                         averaged across models for the headline number.

Checkpoints written under ``--checkpoint_dir`` (defaults to dirname(--output)):
  - tables.csv              (one row per manifest — printable summary)
  - per_feature.csv         (long-form: one row per feature with raw + normed
                             metrics and composite scores at every grid λ)
  - pooled_stats.json       (MI/Kext min/max, lambdas, manifest list)
  - derived/<row_id>.parquet (materialized derived columns per row)

Use the per_feature.csv + pooled_stats.json to recompute composites at any
lambda without re-running the search or eval.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.extend(["."])

from autolearn.data import Dataset
from autolearn.feat_selection.env import SklearnEnv
from autolearn.feat_selection.nfo.feat_tree import generate_tree_from_str
from autolearn.feat_selection.nfo.metrics import (
    conditional_mutual_information,
    dbf as dbf_fn,
    fisher_score,
    kext as kext_fn,
    kext_dbf_from_formula,
    mutual_information,
    pearson,
    residual_from_base,
)
from autolearn.utils import log

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, RidgeCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


DEFAULT_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


# LightGBM ignores the constructor `verbose=-1` for native-side messages like
# "No further splits with positive gain". Register a no-op logger once so CV
# loops don't flood stderr (it can produce hundreds of lines per fold on small
# / saturated datasets and makes eval look hung).
def _silence_lightgbm():
    try:
        import lightgbm as _lgb
        class _NullLogger:
            def info(self, msg): pass
            def warning(self, msg): pass
        if hasattr(_lgb, 'register_logger'):
            _lgb.register_logger(_NullLogger())
    except Exception:
        pass

_silence_lightgbm()


def _lgbm_major_version():
    try:
        import lightgbm as _lgb
        v = getattr(_lgb, '__version__', '0.0.0').split('.')
        return int(v[0])
    except Exception:
        return 0


_LGBM_MAJOR = _lgbm_major_version()


def _safe_lgbm(cls, *, random_state=0):
    """Construct an LGBM sklearn wrapper with kwargs gated by LightGBM major version.
    LGBM sklearn wrappers accept arbitrary kwargs at __init__ and only raise at
    fit time, so a try/except around construction would not catch unknowns —
    must dispatch on ``lightgbm.__version__``.
        - ``force_col_wise`` and ``register_logger``: require LightGBM >= 3.0
        - ``verbosity`` (lowercased C++-side flag): safe everywhere since >= 2.2
    """
    kw = dict(verbose=-1, n_jobs=1, random_state=random_state)
    if _LGBM_MAJOR >= 2:
        kw['verbosity'] = -1
    if _LGBM_MAJOR >= 3:
        kw['force_col_wise'] = True
    return cls(**kw)


# Models that carry their own preprocessing (impute + scale) inside a Pipeline,
# so _evaluate_model must NOT pre-fill NaNs with the -1 sentinel for them — the
# pipeline's SimpleImputer handles missing values on the standardized scale.
_PIPELINE_MODELS = {'LIN', 'KNN'}


def _clip_std(X):
    """Clip standardized features to +/-5 to bound test-fold extrapolation.

    Symbolic features (reciprocal, division, exp) are heavy-tailed: even after
    robust scaling a single extreme test value can produce a wild linear
    prediction and a catastrophic negative R^2. Clipping the robust-scaled
    values caps each feature's leverage so the linear model stays well-behaved.
    """
    return np.clip(X, -5.0, 5.0)


def _make_linear_pipeline(task):
    """median-impute -> RobustScaler -> clip(+/-5) -> linear model.

    A capacity-limited downstream evaluator. Unlike RF/GBM it cannot synthesize
    nonlinear features internally, so its score reflects how much usable signal
    the *engineered* features actually carry — it discriminates feature quality
    across methods far better than RF (which saturates on the base features).

    Robustness matters: symbolic features have huge dynamic range + inf, so we
    use RobustScaler (median/IQR, outlier-resistant) instead of StandardScaler,
    then clip to +/-5 to stop a lone extreme test value from blowing up the
    linear prediction (the StandardScaler version produced R^2 ~ -2000 on the
    regression datasets). The impute step handles inf->NaN from _evaluate_model.
    """
    steps = [
        ('imp', SimpleImputer(strategy='median')),
        ('sc', RobustScaler()),
        ('clip', FunctionTransformer(_clip_std)),
    ]
    if task == 'regression':
        steps.append(('reg', RidgeCV(alphas=np.logspace(-3, 3, 13))))
    else:
        steps.append(('clf', LogisticRegression(max_iter=2000)))
    return Pipeline(steps)


def _make_knn_pipeline(task):
    """median-impute -> RobustScaler -> clip -> kNN.

    Distance-based learner; needs the same robust scaling as LIN (kNN distances
    are meaningless on raw symbolic features with huge dynamic range). Adds a
    non-linear, non-tree family to the downstream panel for a more robust average.
    """
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    steps = [
        ('imp', SimpleImputer(strategy='median')),
        ('sc', RobustScaler()),
        ('clip', FunctionTransformer(_clip_std)),
    ]
    if task == 'regression':
        steps.append(('knn', KNeighborsRegressor()))
    else:
        steps.append(('knn', KNeighborsClassifier()))
    return Pipeline(steps)


def _make_model(task, name, *, n_estimators=100, random_state=0):
    if name == 'LIN':
        return _make_linear_pipeline(task)
    if name == 'KNN':
        return _make_knn_pipeline(task)
    if task == 'regression':
        if name == 'RF':
            return RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        if name == 'LR':
            return Lasso()
        if name == 'LGB':
            from lightgbm import LGBMRegressor
            return _safe_lgbm(LGBMRegressor, random_state=random_state)
        if name == 'XGB':
            from xgboost import XGBRegressor
            return XGBRegressor(random_state=random_state, verbosity=0, n_jobs=1)
        raise ValueError(name)
    if name == 'RF':
        return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    if name == 'LR':
        return LogisticRegression(max_iter=1000)
    if name == 'LGB':
        from lightgbm import LGBMClassifier
        return _safe_lgbm(LGBMClassifier, random_state=random_state)
    if name == 'XGB':
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=random_state, verbosity=0, n_jobs=1,
                             use_label_encoder=False, eval_metric='logloss')
    raise ValueError(name)


def _evaluate_model(X, y, task, model_name, *, cv=5, scoring='f1_micro',
                    return_per_fold=False):
    X = X.copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    if model_name not in _PIPELINE_MODELS:
        # Tree/boosting models: keep the legacy -1 sentinel fill. Pipeline models
        # (LIN) impute medians on the standardized scale inside the pipeline, so
        # leave NaNs in place for them.
        X.fillna(-1, inplace=True)
    X.columns = X.columns.astype(str)
    model = _make_model(task, model_name)
    if task == 'regression':
        from autolearn.metrics import r2_score
        scorer = make_scorer(r2_score)
    else:
        scorer = scoring
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(model, X, y, scoring=scorer, cv=int(cv))
    if return_per_fold:
        return np.asarray(scores, dtype=np.float64)
    return float(scores.mean())


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _ci90_halfwidth(per_fold):
    """Half-width of a 90% confidence interval from per-fold CV scores.
    Uses t-distribution with df=n-1; returns NaN if fewer than 2 finite folds.
    """
    arr = np.asarray(per_fold, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float('nan')
    try:
        from scipy import stats as _sps
        tval = float(_sps.t.ppf(0.95, df=arr.size - 1))
    except Exception:
        # crude fallback (z=1.645 for two-sided 90% CI)
        tval = 1.645
    std = float(np.std(arr, ddof=1))
    return tval * std / np.sqrt(arr.size)


def _mannwhitney_one_sided(method_vals, base_vals, *, higher_is_better=True):
    """Mann-Whitney U one-sided p-value. NaN-safe.
    Returns NaN if either group has <2 finite values."""
    m = np.asarray(method_vals, dtype=np.float64)
    b = np.asarray(base_vals, dtype=np.float64)
    m = m[np.isfinite(m)]
    b = b[np.isfinite(b)]
    if m.size < 2 or b.size < 2:
        return float('nan')
    try:
        from scipy import stats as _sps
        alt = 'greater' if higher_is_better else 'less'
        _, p = _sps.mannwhitneyu(m, b, alternative=alt)
        return float(p)
    except Exception:
        return float('nan')


def _paired_t_one_sided(method_folds, base_folds, *, higher_is_better=True):
    """Paired t-test one-sided p-value, assuming fold indices align.
    Returns NaN if either group has <2 finite folds or all differences are 0."""
    m = np.asarray(method_folds, dtype=np.float64)
    b = np.asarray(base_folds, dtype=np.float64)
    if m.shape != b.shape:
        return float('nan')
    diff = m - b if higher_is_better else b - m
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return float('nan')
    if np.allclose(diff, 0):
        return float('nan')
    try:
        from scipy import stats as _sps
        # one-sided: divide two-sided p by 2 and check direction
        t, p_two = _sps.ttest_1samp(diff, 0.0, nan_policy='omit')
        p_one = p_two / 2.0 if t > 0 else 1.0 - p_two / 2.0
        return float(p_one)
    except Exception:
        return float('nan')


def _sig_marker(p, alpha=0.1):
    """Return '*' when p < alpha (default α = 0.1 → 90% confidence)."""
    if p is None or not isinstance(p, (int, float)) or not np.isfinite(p):
        return ''
    return '*' if p < alpha else ''


def _materialize_columns(env, feature_strs):
    """Return (derived_df, kept_feature_strs).

    Columns are renumbered consecutively (``new_0`` ... ``new_{k-1}``) and
    ``kept_feature_strs`` is aligned with them — so callers that need to pair
    per-feature tree metrics with column-wise usefulness metrics can iterate
    positionally without misalignment when some features fail to construct.
    """
    if not feature_strs:
        return pd.DataFrame(), []
    trees = generate_tree_from_str(feature_strs, env)
    cols = []
    kept = []
    for i, tree in enumerate(trees):
        if tree is None:
            log(f"[eval_quality] skipping unparseable feature {feature_strs[i]!r}")
            continue
        try:
            col = env.construct_feature(tree)
            cols.append(np.asarray(col, dtype=np.float64).reshape(-1))
            kept.append(feature_strs[i])
        except Exception as e:
            log(f"[eval_quality] skipping unbuildable feature {feature_strs[i]!r}: {e}")
    if not cols:
        return pd.DataFrame(), []
    df = pd.DataFrame({f"new_{j}": col for j, col in enumerate(cols)})
    return df, kept


def _build_augmented_frame(env, derived_df):
    X = env.dataset.instances.copy()
    X.columns = X.columns.astype(str)
    if derived_df is None or derived_df.empty:
        return X
    derived_df = derived_df.reset_index(drop=True)
    X = X.reset_index(drop=True)
    derived_df.columns = [f"new_{i}" for i in range(derived_df.shape[1])]
    return pd.concat([X, derived_df], axis=1)


def downstream_table(env, derived_df, models, *, cv=5, scoring='f1_micro',
                     top_k=None, row_label='', augment=True):
    """Run downstream CV. When ``augment=False``, X_aug = base features only —
    used for the synthetic ``base_features`` / ``base_only`` rows so they
    evaluate on just the original dataset (not [base ∪ base]).

    Returns ``(out, per_fold)`` where ``per_fold`` maps model name → ndarray of
    per-fold scores (used downstream for paired tests + 90% CIs).
    """
    if derived_df is not None and top_k is not None:
        derived_df = derived_df.iloc[:, :top_k]
    if augment:
        X_aug = _build_augmented_frame(env, derived_df)
    else:
        X_aug = env.dataset.instances.copy()
        X_aug.columns = X_aug.columns.astype(str)
    y = np.asarray(env.dataset.labels)
    task = env.dataset.task

    log(f"[eval_quality] {row_label} downstream baseline_RF on "
        f"{env.dataset.instances.shape[0]}x{env.dataset.instances.shape[1]} (base only)")
    t0 = time.time()
    base_folds = _evaluate_model(env.dataset.instances.copy(), y, task, 'RF',
                                 cv=cv, scoring=scoring, return_per_fold=True)
    base = float(np.nanmean(base_folds))
    log(f"[eval_quality] {row_label}   baseline_RF done in {time.time()-t0:.1f}s, score={base:.4f}")

    n_derived = 0 if derived_df is None else derived_df.shape[1]
    out = {'baseline_RF_base_only': base, 'top_k': n_derived if augment else 0}
    per_fold = {'baseline_RF': base_folds}
    per_model = []
    for name in models:
        log(f"[eval_quality] {row_label} downstream {name} on "
            f"{X_aug.shape[0]}x{X_aug.shape[1]} (cv={cv})  augment={augment}")
        t1 = time.time()
        try:
            folds = _evaluate_model(X_aug, y, task, name, cv=cv, scoring=scoring,
                                    return_per_fold=True)
            s = float(np.nanmean(folds))
            log(f"[eval_quality] {row_label}   {name} done in {time.time()-t1:.1f}s, score={s:.4f}")
        except Exception as e:
            log(f"[eval_quality] {row_label}   {name} failed: {type(e).__name__}: {e}")
            folds = np.array([float('nan')] * cv)
            s = float('nan')
        out[f'downstream_{name}'] = s
        out[f'downstream_{name}_ci90'] = _ci90_halfwidth(folds)
        per_fold[name] = folds
        per_model.append(s)
    out['downstream_avg'] = float(np.nanmean(per_model)) if per_model else float('nan')
    return out, per_fold


def load_manifest(env, path):
    """Load a manifest JSON and materialize derived columns.

    Shapes:
      A) Legacy VI cache: {'features': [...], 'lam': float, 'method': str}
      B) Generic baseline: {'kind': 'columns', 'columns_path': ..., 'feature_strs'|'formulas': [...]}
    Returns (derived_df, feature_strs|None, formulas|None, meta).
    """
    with open(path, 'r') as f:
        payload = json.load(f)

    if 'features' in payload and 'kind' not in payload:
        feats = payload['features']
        derived_df, kept = _materialize_columns(env, feats)
        meta = {
            'dataset': payload.get('dataset'),
            'method': payload.get('method', 'vi'),
            'lam': payload.get('lam'),
            'n_ops_override': None,
        }
        # Trim feature_strs so per-feature tree metrics stay aligned with the
        # materialized columns (some features may fail to construct).
        return derived_df, kept, None, meta

    base = os.path.dirname(path)
    derived_df = None
    feats = payload.get('feature_strs')
    formulas = payload.get('formulas')
    cpath = payload.get('columns_path')
    if cpath:
        cpath_full = cpath if os.path.isabs(cpath) else os.path.join(base, cpath)
        if cpath_full.endswith('.parquet'):
            derived_df = pd.read_parquet(cpath_full)
        else:
            derived_df = pd.read_csv(cpath_full)
    if derived_df is None and feats is not None:
        derived_df, feats = _materialize_columns(env, feats)
    method = payload.get('method', '')
    n_ops_override = None
    if method == 'openfe':
        n_ops_override = 20
    elif method == 'autofeat':
        n_ops_override = 11
    meta = {
        'dataset': payload.get('dataset'),
        'method': method,
        'lam': payload.get('lam'),
        'n_ops_override': n_ops_override,
    }
    return derived_df, feats, formulas, meta


# ---------------------------------------------------------------------------
# Per-feature metric collection
# ---------------------------------------------------------------------------

def _per_feature_metrics(env, derived_df, feature_strs, formulas, *,
                         n_ops_override, residual=None):
    """Compute raw per-feature metrics. Returns list[dict] (one per feature).

    Aligned with derived_df's columns. Kext/DBF use the tree path when feature
    strings parse, else fall back to the sympy/regex heuristic from formulas.

    ``residual`` is the precomputed base-feature residual (see
    ``residual_from_base`` in metrics.py). When provided, each record also
    carries a ``cmi`` field = MI(f; residual), i.e. conditional MI vs the
    base feature set.
    """
    if derived_df is None or derived_df.empty:
        return [], 'none'

    y = np.asarray(env.dataset.labels)
    task = env.dataset.task
    n_base = len(env.features)
    n_ops = n_ops_override if n_ops_override is not None else len(env.ops)

    trees, source = None, 'none'
    if feature_strs:
        try:
            trees = generate_tree_from_str(feature_strs, env)
            source = 'tree'
        except Exception:
            trees = None

    base_names = list(env.features) + [f"f{name}" for name in env.features]

    records = []
    for i in range(derived_df.shape[1]):
        col = derived_df.iloc[:, i].to_numpy()
        try:
            mi = float(mutual_information(col, y, task))
        except Exception:
            mi = float('nan')
        try:
            pr = float(pearson(col, y))
        except Exception:
            pr = float('nan')
        try:
            fs = float(fisher_score(col, y))
        except Exception:
            fs = float('nan')
        cmi = float('nan')
        if residual is not None:
            try:
                cmi = float(conditional_mutual_information(col, residual))
            except Exception:
                cmi = float('nan')

        k, d = float('nan'), float('nan')
        if trees is not None and i < len(trees) and trees[i] is not None:
            try:
                k = float(kext_fn(trees[i], n_base, n_ops))
                d = float(dbf_fn(trees[i]))
            except Exception:
                pass
        elif formulas and i < len(formulas):
            try:
                k_h, d_h = kext_dbf_from_formula(
                    formulas[i], n_base, n_ops, base_feature_names=base_names,
                )
                k = float(k_h) if k_h is not None else float('nan')
                d = float(d_h) if d_h is not None else float('nan')
                if source == 'none':
                    source = 'formula'
            except Exception:
                pass

        feat_label = None
        if feature_strs and i < len(feature_strs):
            feat_label = feature_strs[i]
        elif formulas and i < len(formulas):
            feat_label = formulas[i]

        records.append({
            'feat_idx': i,
            'feat_str': feat_label,
            'mi': mi,
            'cmi': cmi,
            'pearson': pr,
            'fisher': fs,
            'kext': k,
            'dbf': d,
        })
    return records, source


# ---------------------------------------------------------------------------
# Pooled normalization + composite
# ---------------------------------------------------------------------------

def _minmax(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr, float('nan'), float('nan')
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(arr), lo, hi
    return (arr - lo) / (hi - lo), lo, hi


def _composite_at(mi_norm, kext_norm, lam):
    u = np.asarray(mi_norm, dtype=np.float64)
    e = 1.0 - np.asarray(kext_norm, dtype=np.float64)
    return lam * u + (1.0 - lam) * e


# ---------------------------------------------------------------------------
# Per-row aggregation
# ---------------------------------------------------------------------------

def _aggregate_row(records_df, meta, downstream, row_lambdas):
    """Build one summary row from per-feature records DataFrame + downstream dict.

    Two composite families are aggregated in parallel:
        composite_*           — MI-based (legacy, kept for backward compat)
        composite_pearson_*   — |Pearson|-based (new; lets pearson-search cells
                                be ranked under the metric they were trained on)
    Both use the same Kext normalization so they are directly comparable.
    """
    if records_df is None or records_df.empty:
        agg = {
            'mi_mean': np.nan, 'mi_max': np.nan,
            'cmi_mean': np.nan, 'cmi_max': np.nan,
            'pearson_mean': np.nan, 'pearson_max': np.nan,
            'fisher_mean': np.nan, 'fisher_max': np.nan,
            'kext_mean': np.nan, 'kext_max': np.nan,
            'dbf_mean': np.nan, 'dbf_max': np.nan,
        }
        for lam in row_lambdas:
            for pre in ('composite', 'composite_pearson', 'composite_fisher'):
                agg[f'{pre}_mean_lam{lam}'] = np.nan
                agg[f'{pre}_max_lam{lam}'] = np.nan
        for pre in ('composite', 'composite_pearson', 'composite_fisher'):
            agg[f'{pre}_mean_at_row_lam'] = np.nan
            agg[f'{pre}_max_at_row_lam'] = np.nan
        return agg

    def _mm(col):
        a = records_df[col].to_numpy(dtype=np.float64)
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return np.nan, np.nan
        return float(np.nanmean(a)), float(np.nanmax(a))

    mi_mean, mi_max = _mm('mi')
    if 'cmi' in records_df.columns:
        cmi_mean, cmi_max = _mm('cmi')
    else:
        cmi_mean, cmi_max = np.nan, np.nan
    pr_mean, pr_max = _mm('pearson')
    fs_mean, fs_max = _mm('fisher')
    k_mean, k_max = _mm('kext')
    d_mean, d_max = _mm('dbf')

    agg = {
        'mi_mean': mi_mean, 'mi_max': mi_max,
        'cmi_mean': cmi_mean, 'cmi_max': cmi_max,
        'pearson_mean': pr_mean, 'pearson_max': pr_max,
        'fisher_mean': fs_mean, 'fisher_max': fs_max,
        'kext_mean': k_mean, 'kext_max': k_max,
        'dbf_mean': d_mean, 'dbf_max': d_max,
    }
    for lam in row_lambdas:
        for prefix in ('composite', 'composite_pearson', 'composite_fisher'):
            col = f'{prefix}_lam{lam}'
            if col in records_df.columns:
                a = records_df[col].to_numpy(dtype=np.float64)
                agg[f'{prefix}_mean_lam{lam}'] = float(np.nanmean(a)) if a.size else np.nan
                agg[f'{prefix}_max_lam{lam}']  = float(np.nanmax(a)) if a.size else np.nan
            else:
                agg[f'{prefix}_mean_lam{lam}'] = np.nan
                agg[f'{prefix}_max_lam{lam}'] = np.nan

    lam_row = meta.get('lam')
    if lam_row is not None and np.isfinite(lam_row):
        try:
            e = 1.0 - records_df['kext_norm'].to_numpy(dtype=np.float64)
        except Exception:
            e = None
        for (out_key, norm_col) in (
            ('composite',         'mi_norm'),
            ('composite_pearson', 'pearson_norm'),
            ('composite_fisher',  'fisher_norm'),
        ):
            if e is None or norm_col not in records_df.columns:
                agg[f'{out_key}_mean_at_row_lam'] = np.nan
                agg[f'{out_key}_max_at_row_lam'] = np.nan
                continue
            try:
                u = records_df[norm_col].to_numpy(dtype=np.float64)
                comp = lam_row * u + (1.0 - lam_row) * e
                agg[f'{out_key}_mean_at_row_lam'] = float(np.nanmean(comp))
                agg[f'{out_key}_max_at_row_lam'] = float(np.nanmax(comp))
            except Exception:
                agg[f'{out_key}_mean_at_row_lam'] = np.nan
                agg[f'{out_key}_max_at_row_lam'] = np.nan
    else:
        for out_key in ('composite', 'composite_pearson', 'composite_fisher'):
            agg[f'{out_key}_mean_at_row_lam'] = np.nan
            agg[f'{out_key}_max_at_row_lam'] = np.nan
    return agg


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _row_id(method, lam, source_idx):
    safe_method = re.sub(r'[^A-Za-z0-9._-]+', '_', str(method or 'unknown'))
    lam_str = 'none' if lam is None else f'{lam}'
    return f"{safe_method}__lam{lam_str}__src{source_idx}"


def collect_rows(env, paths, *, top_k, include_base_features,
                 base_features_label='base_features'):
    """Pass 1: load each manifest. Optionally append a synthetic ``base_features``
    row that scores every base feature directly and uses the augment-free
    downstream path (matches what was previously called ``base_only``, but with
    per-feature usefulness/explainability populated so % improvements can be
    computed against it).
    """
    rows = []
    for src_idx, path in enumerate(paths):
        log(f"[eval_quality] loading {path}")
        try:
            derived_df, feature_strs, formulas, meta = load_manifest(env, path)
        except Exception as e:
            log(f"[eval_quality] FAILED to load {path}: {type(e).__name__}: {e}")
            continue
        if top_k is not None and derived_df is not None:
            derived_df = derived_df.iloc[:, :top_k]
            if feature_strs is not None:
                feature_strs = feature_strs[:top_k]
            if formulas is not None:
                formulas = formulas[:top_k]
        rows.append({
            'meta': meta,
            'derived_df': derived_df,
            'feature_strs': feature_strs,
            'formulas': formulas,
            'source_path': path,
            'src_idx': src_idx,
            'row_id': _row_id(meta.get('method', 'unknown'), meta.get('lam'), src_idx),
            'augment_downstream': True,
        })

    if include_base_features:
        # Per-feature metrics: treat each base feature as its own feature.
        # ``feature_strs`` = base feature names (parse to single-leaf trees;
        # Kext = log2(n_base) per leaf, DBF = 1 per leaf).
        feat_strs = [str(c) for c in env.features]
        base_X = env.dataset.instances.copy()
        base_X.columns = [f"new_{i}" for i in range(base_X.shape[1])]
        rows.append({
            'meta': {'dataset': env.dataset_path, 'method': base_features_label,
                     'lam': None, 'n_ops_override': None},
            'derived_df': base_X,
            'feature_strs': feat_strs,
            'formulas': None,
            'source_path': '<synthetic: base features>',
            'src_idx': len(rows),
            'row_id': _row_id(base_features_label, None, len(rows)),
            # Use augment=False so downstream evaluates [base] not [base ∪ base].
            'augment_downstream': False,
        })

    return rows


def compute_per_feature(env, rows):
    """Pass 2: per-feature raw metrics for every row. Attaches:
        row['records'] = list[dict]
        row['kext_source'] = 'tree'|'formula'|'none'
    Returns the long-form per-feature DataFrame (raw metrics, not yet normed).
    """
    # Precompute the base-feature residual once per dataset so every row's
    # per-feature CMI = MI(f; r) reuses it for free.
    try:
        base_df = env.dataset.instances
        X_base = base_df.values if hasattr(base_df, 'values') else np.asarray(base_df)
        y = np.asarray(env.dataset.labels)
        residual = residual_from_base(X_base, y, env.dataset.task,
                                       n_estimators=10, cv=int(getattr(env, 'cv', 5) or 5))
        log(f"[eval_quality] residual built for CMI: n={len(residual)} "
            f"mean={float(np.mean(residual)):.4f} std={float(np.std(residual)):.4f}")
    except Exception as e:
        log(f"[eval_quality] residual_from_base failed ({type(e).__name__}: {e}); "
            f"cmi column will be NaN")
        residual = None

    feat_frames = []
    for row in rows:
        records, source = _per_feature_metrics(
            env, row['derived_df'], row['feature_strs'], row['formulas'],
            n_ops_override=row['meta'].get('n_ops_override'),
            residual=residual,
        )
        row['records'] = records
        row['kext_source'] = source
        if records:
            df = pd.DataFrame(records)
            df.insert(0, 'method', row['meta'].get('method', 'unknown'))
            df.insert(1, 'lam', row['meta'].get('lam'))
            df.insert(2, 'row_id', row['row_id'])
            df['kext_source'] = source
            feat_frames.append(df)
    if feat_frames:
        return pd.concat(feat_frames, ignore_index=True)
    return pd.DataFrame(columns=[
        'method', 'lam', 'row_id', 'feat_idx', 'feat_str',
        'mi', 'pearson', 'fisher', 'kext', 'dbf', 'kext_source',
    ])


def add_pooled_norms_and_composite(per_feat_df, lambdas):
    """Pool MI/Pearson/Fisher/Kext across all features, add *_norm + composite cols.

    Three composite families are emitted side-by-side so a sweep that mixes
    different usefulness-search cells can be ranked under any of the three
    metrics without re-running the search:
        composite_lam<λ>         = λ·MI_norm      + (1-λ)·(1-Kext_norm)
        composite_pearson_lam<λ> = λ·Pearson_norm + (1-λ)·(1-Kext_norm)
        composite_fisher_lam<λ>  = λ·Fisher_norm  + (1-λ)·(1-Kext_norm)
    Normalization is pooled across every feature in the eval call, so the
    composites are directly cross-method comparable within a single eval pool.
    Returns (per_feat_df_augmented, pooled_stats_dict).
    """
    if per_feat_df.empty:
        for col in ['mi_norm', 'pearson_norm', 'fisher_norm', 'kext_norm']:
            per_feat_df[col] = np.nan
        for lam in lambdas:
            per_feat_df[f'composite_lam{lam}'] = np.nan
            per_feat_df[f'composite_pearson_lam{lam}'] = np.nan
            per_feat_df[f'composite_fisher_lam{lam}'] = np.nan
        return per_feat_df, {
            'mi_min': None, 'mi_max': None,
            'pearson_min': None, 'pearson_max': None,
            'fisher_min': None, 'fisher_max': None,
            'kext_min': None, 'kext_max': None,
            'n_features_pool': 0, 'lambdas': list(lambdas),
        }

    mi_arr = per_feat_df['mi'].to_numpy(dtype=np.float64)
    pr_arr = per_feat_df['pearson'].to_numpy(dtype=np.float64)
    fi_arr = per_feat_df['fisher'].to_numpy(dtype=np.float64)
    kext_arr = per_feat_df['kext'].to_numpy(dtype=np.float64)
    mi_norm, mi_lo, mi_hi = _minmax(mi_arr)
    pr_norm, pr_lo, pr_hi = _minmax(pr_arr)
    fi_norm, fi_lo, fi_hi = _minmax(fi_arr)
    kext_norm, k_lo, k_hi = _minmax(kext_arr)
    per_feat_df['mi_norm'] = mi_norm
    per_feat_df['pearson_norm'] = pr_norm
    per_feat_df['fisher_norm'] = fi_norm
    per_feat_df['kext_norm'] = kext_norm
    for lam in lambdas:
        per_feat_df[f'composite_lam{lam}'] = _composite_at(mi_norm, kext_norm, lam)
        per_feat_df[f'composite_pearson_lam{lam}'] = _composite_at(pr_norm, kext_norm, lam)
        per_feat_df[f'composite_fisher_lam{lam}'] = _composite_at(fi_norm, kext_norm, lam)
    stats = {
        'mi_min': mi_lo, 'mi_max': mi_hi,
        'pearson_min': pr_lo, 'pearson_max': pr_hi,
        'fisher_min': fi_lo, 'fisher_max': fi_hi,
        'kext_min': k_lo, 'kext_max': k_hi,
        'n_features_pool': int(np.isfinite(mi_arr).sum()),
        'lambdas': list(lambdas),
    }
    return per_feat_df, stats


def build_summary(env, rows, per_feat_df, lambdas, *, models, cv, scoring,
                  base_method_label='base_features'):
    """Pass 3: per-row downstream + per-feature aggregate → summary DataFrame.

    Attaches each row's per-fold downstream scores at ``row['per_fold']`` so the
    later ratio/p-value pass can run paired tests against the base anchor.
    """
    summary_rows = []
    for row in rows:
        records_df = per_feat_df[per_feat_df['row_id'] == row['row_id']].reset_index(drop=True)

        t0 = time.time()
        row_label = f"[{row['meta'].get('method','?')} lam={row['meta'].get('lam')}]"
        try:
            dn, per_fold = downstream_table(
                env, row['derived_df'], models, cv=cv, scoring=scoring,
                row_label=row_label, augment=row.get('augment_downstream', True),
            )
        except Exception as e:
            log(f"[eval_quality] downstream FAILED for {row['source_path']}: {type(e).__name__}: {e}")
            dn = {'baseline_RF_base_only': float('nan'), 'top_k': 0,
                  'downstream_avg': float('nan')}
            for m in models:
                dn[f'downstream_{m}'] = float('nan')
                dn[f'downstream_{m}_ci90'] = float('nan')
            per_fold = {m: np.array([float('nan')] * cv) for m in models}
            per_fold['baseline_RF'] = np.array([float('nan')] * cv)
        row['per_fold'] = per_fold
        elapsed = time.time() - t0

        agg = _aggregate_row(records_df, row['meta'], dn, lambdas)
        n_eval = 0 if row['derived_df'] is None else row['derived_df'].shape[1]
        summary_rows.append({
            'dataset': row['meta'].get('dataset'),
            'method': row['meta'].get('method', 'unknown'),
            'lam': row['meta'].get('lam'),
            'k_evaluated': n_eval,
            **agg, **dn,
            'kext_source': row.get('kext_source', 'none'),
            'elapsed_sec': elapsed,
            'row_id': row['row_id'],
            'source': row['source_path'],
        })
    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Improvement ratios (% change) + statistical tests vs ``base_features`` anchor
# ---------------------------------------------------------------------------

_METRIC_DIRECTIONS = (
    # (column_name, mean_aggregator_col, higher_is_better)
    ('mi',       'mi_mean',       True),
    ('pearson',  'pearson_mean',  True),
    ('fisher',   'fisher_mean',   True),
    ('kext',     'kext_mean',     False),  # lower Kext = more explainable
    ('dbf',      'dbf_mean',      False),  # smaller distinct-base-features = simpler
)


def annotate_ratios_and_pvalues(summary_df, per_feat_df, rows, models,
                                base_method_label='base_features', alpha=0.1):
    """Add ``{metric}_ratio``, ``{metric}_pvalue``, ``downstream_*_ratio``,
    ``downstream_*_pvalue`` columns to ``summary_df`` in place.

    The ``_ratio`` columns are signed percent change vs the base anchor:
    ``(row_mean - base_mean) / base_mean * 100``. Sign is preserved (so
    "lower is better" metrics keep negative = improvement); plot code colors
    bars by the metric's favorable direction.

    Per-feature distributions are compared row-vs-base with Mann-Whitney U
    (one-sided in the metric's "better" direction).
    Downstream scores are compared with a paired t-test on per-fold CV scores
    (one-sided in the "higher is better" direction — for regression where the
    scorer is r2_score, higher is still better).
    """
    # Locate the base anchor row.
    base_row = None
    for r in rows:
        if r['meta'].get('method') == base_method_label:
            base_row = r
            break
    if base_row is None:
        log(f"[eval_quality] no '{base_method_label}' row present — skipping ratios/p-values")
        for col in ['mi_ratio', 'mi_pvalue', 'pearson_ratio', 'pearson_pvalue',
                    'fisher_ratio', 'fisher_pvalue', 'kext_ratio', 'kext_pvalue',
                    'dbf_ratio', 'dbf_pvalue']:
            if col not in summary_df.columns:
                summary_df[col] = np.nan
        for m in list(models) + ['avg']:
            for col in [f'downstream_{m}_ratio', f'downstream_{m}_pvalue']:
                if col not in summary_df.columns:
                    summary_df[col] = np.nan
        return summary_df

    base_records = per_feat_df[per_feat_df['row_id'] == base_row['row_id']]
    base_summary = summary_df[summary_df['row_id'] == base_row['row_id']]
    base_per_fold = base_row.get('per_fold', {})

    base_means = {}
    for raw_col, mean_col, _ in _METRIC_DIRECTIONS:
        v = base_summary[mean_col].iloc[0] if not base_summary.empty else np.nan
        base_means[raw_col] = float(v) if isinstance(v, (int, float)) and np.isfinite(v) else np.nan

    base_downstream_scores = {}
    for m in models:
        col = f'downstream_{m}'
        v = base_summary[col].iloc[0] if not base_summary.empty else np.nan
        base_downstream_scores[m] = float(v) if isinstance(v, (int, float)) and np.isfinite(v) else np.nan
    base_downstream_scores['avg'] = (
        float(base_summary['downstream_avg'].iloc[0])
        if not base_summary.empty and np.isfinite(base_summary['downstream_avg'].iloc[0])
        else np.nan
    )

    new_cols = {}

    def _pct(row_v, base_v):
        if not (np.isfinite(row_v) and np.isfinite(base_v)) or abs(base_v) < 1e-12:
            return np.nan
        return (row_v - base_v) / base_v * 100.0

    # Per-row vs base: per-feature % improvement + Mann-Whitney p-values
    for raw_col, mean_col, hib in _METRIC_DIRECTIONS:
        ratios, pvalues = [], []
        base_vals = base_records[raw_col].to_numpy(dtype=np.float64)
        for _, srow in summary_df.iterrows():
            row_mean = srow.get(mean_col)
            try:
                row_mean = float(row_mean) if np.isfinite(row_mean) else np.nan
            except Exception:
                row_mean = np.nan
            ratios.append(_pct(row_mean, base_means[raw_col]))
            if srow['row_id'] == base_row['row_id']:
                pvalues.append(float('nan'))
                continue
            row_vals = per_feat_df[per_feat_df['row_id'] == srow['row_id']][raw_col].to_numpy(dtype=np.float64)
            pvalues.append(_mannwhitney_one_sided(row_vals, base_vals, higher_is_better=hib))
        new_cols[f'{raw_col}_ratio'] = ratios
        new_cols[f'{raw_col}_pvalue'] = pvalues

    # Downstream % improvement + paired t-tests
    for m in list(models) + ['avg']:
        ratios, pvalues = [], []
        base_key = m if m in base_per_fold else ('baseline_RF' if m == 'avg' else None)
        base_folds = base_per_fold.get(base_key) if base_key else None
        for _, srow in summary_df.iterrows():
            row_score = srow.get(f'downstream_{m}', np.nan)
            try:
                row_score = float(row_score)
            except Exception:
                row_score = np.nan
            base_score = base_downstream_scores.get(m, np.nan)
            ratios.append(_pct(row_score, base_score))
            if srow['row_id'] == base_row['row_id']:
                pvalues.append(float('nan'))
                continue
            # locate the row's per_fold
            row_obj = next((r for r in rows if r['row_id'] == srow['row_id']), None)
            if row_obj is None or base_folds is None:
                pvalues.append(float('nan'))
                continue
            row_folds = row_obj.get('per_fold', {}).get(m if m != 'avg' else 'baseline_RF')
            if row_folds is None:
                pvalues.append(float('nan'))
                continue
            pvalues.append(_paired_t_one_sided(row_folds, base_folds, higher_is_better=True))
        new_cols[f'downstream_{m}_ratio'] = ratios
        new_cols[f'downstream_{m}_pvalue'] = pvalues

    for k, v in new_cols.items():
        summary_df[k] = v
    return summary_df


def save_checkpoints(checkpoint_dir, summary_df, per_feat_df, pooled_stats,
                     rows, lambdas, *, write_derived=True):
    """Write tables.csv, per_feature.csv, pooled_stats.json, derived/*.parquet.

    When ``write_derived`` is False, the ``derived/`` directory is skipped
    entirely (saves disk on long sweeps). Even when True, lambda-independent
    baseline rows (``meta.lam is None``) are written once and subsequent
    duplicates inside the same checkpoint_dir are skipped — the per-(dataset, λ)
    sweep layout in ``run_quality_compare.sh`` re-evaluates these baselines
    at every λ, so deduping cuts disk by ~5× for those methods.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    summary_path = os.path.join(checkpoint_dir, 'tables.csv')
    summary_df.to_csv(summary_path, index=False)
    log(f"[eval_quality] wrote {summary_path}")

    per_feat_path = os.path.join(checkpoint_dir, 'per_feature.csv')
    per_feat_df.to_csv(per_feat_path, index=False)
    log(f"[eval_quality] wrote {per_feat_path}")

    stats_path = os.path.join(checkpoint_dir, 'pooled_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({
            **pooled_stats,
            'lambdas': list(lambdas),
            'rows': [
                {
                    'row_id': r['row_id'],
                    'method': r['meta'].get('method'),
                    'lam': r['meta'].get('lam'),
                    'source_path': r['source_path'],
                    'k_evaluated': 0 if r['derived_df'] is None else int(r['derived_df'].shape[1]),
                }
                for r in rows
            ],
        }, f, indent=2)
    log(f"[eval_quality] wrote {stats_path}")

    if not write_derived:
        log("[eval_quality] --no_derived_checkpoint: skipping derived/*.parquet")
        return

    derived_dir = os.path.join(checkpoint_dir, 'derived')
    os.makedirs(derived_dir, exist_ok=True)
    seen_lam_indep = set()
    for r in rows:
        df = r['derived_df']
        if df is None or df.empty:
            continue
        method = r['meta'].get('method', 'unknown')
        lam = r['meta'].get('lam')
        # Dedup lambda-independent baselines within this checkpoint_dir.
        if lam is None:
            if method in seen_lam_indep:
                log(f"[eval_quality] dedup: skipping derived parquet for "
                    f"lambda-independent {method!r} (already written)")
                continue
            seen_lam_indep.add(method)
        out_df = df.copy()
        out_df.columns = [str(c) for c in out_df.columns]
        out_path = os.path.join(derived_dir, f"{r['row_id']}.parquet")
        try:
            out_df.to_parquet(out_path, index=False)
        except Exception:
            csv_path = os.path.join(derived_dir, f"{r['row_id']}.csv")
            out_df.to_csv(csv_path, index=False)
            log(f"[eval_quality] parquet unavailable, wrote {csv_path} instead")


def expand_inputs(patterns):
    out = []
    for p in patterns:
        matched = sorted(glob.glob(p))
        if matched:
            out.extend(matched)
        elif os.path.exists(p):
            out.append(p)
        else:
            log(f"[eval_quality] no match for {p!r}")
    return out


def parse_args():
    p = argparse.ArgumentParser(description='Quality evaluation for B-MAFE features')
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--inputs', type=str, nargs='*', default=[],
                   help='glob(s) for selected_topk_lam*.json / eval_manifest.json (optional)')
    p.add_argument('--output', type=str, default=None,
                   help='CSV path for the per-row summary (also used as checkpoint dir if --checkpoint_dir omitted)')
    p.add_argument('--checkpoint_dir', type=str, default=None,
                   help='dir to write tables.csv, per_feature.csv, pooled_stats.json, derived/')
    p.add_argument('--models', type=str, nargs='+',
                   default=['RF', 'LR', 'LGB', 'XGB'])
    p.add_argument('--skip_models', type=str, nargs='+', default=[],
                   help='subset of --models to exclude (e.g. --skip_models LGB)')
    p.add_argument('--cv', type=int, default=5)
    p.add_argument('--c_scoring', type=str, default='f1_micro')
    p.add_argument('--max_order', type=int, default=4)
    p.add_argument('--top_k', type=int, default=None)
    p.add_argument('--lambdas', type=float, nargs='+', default=list(DEFAULT_LAMBDAS),
                   help='lambda grid for composite reporting (default: 0 0.25 0.5 0.75 1.0)')
    p.add_argument('--include_base_only', action='store_true',
                   help='alias for --include_base_features (emits the base_features anchor row)')
    p.add_argument('--include_base_features', action='store_true',
                   help='emit synthetic base_features row: per-feature MI/Pearson/Fisher/Kext/DBF '
                        'on each base feature + augment-free downstream. Used as the anchor for '
                        'improvement ratios and one-sided significance tests.')
    p.add_argument('--alpha', type=float, default=0.1,
                   help='significance threshold for printed marker (default 0.1 = 90%% confidence)')
    p.add_argument('--no_derived_checkpoint', action='store_true',
                   help='skip writing derived/*.parquet (saves disk on long sweeps)')
    return p.parse_args()


def main():
    args = parse_args()

    # Force UTF-8 stdout: the print block at the end uses Δ/—/↑/↓ which crash
    # Windows cp1252 consoles, and that crash was discarding all CV results
    # because save_checkpoints used to run *after* printing.
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    env = SklearnEnv(args.data, args.max_order, parallel=False, score_mode='cv')

    paths = expand_inputs(args.inputs)

    models = [m for m in args.models if m not in set(args.skip_models)]
    if args.skip_models:
        log(f"[eval_quality] skipping models: {args.skip_models} (final: {models})")
    args.models = models

    include_base_features = bool(args.include_base_features or args.include_base_only)
    log(f"[eval_quality] dataset={args.data} models={args.models} inputs={len(paths)} "
        f"include_base_features={include_base_features}")

    if not paths and not include_base_features:
        log("[eval_quality] no input manifests and no base anchor requested; nothing to do")
        return

    rows = collect_rows(env, paths, top_k=args.top_k,
                        include_base_features=include_base_features)
    if not rows:
        log("[eval_quality] no rows produced; exiting")
        return

    per_feat_df = compute_per_feature(env, rows)
    per_feat_df, pooled_stats = add_pooled_norms_and_composite(per_feat_df, args.lambdas)
    summary_df = build_summary(env, rows, per_feat_df, args.lambdas,
                                models=args.models, cv=args.cv, scoring=args.c_scoring)
    summary_df = annotate_ratios_and_pvalues(
        summary_df, per_feat_df, rows, args.models, alpha=args.alpha,
    )

    summary_df = summary_df.sort_values(
        ['method', 'lam'], na_position='last', kind='stable',
    ).reset_index(drop=True)

    # Decide checkpoint dir
    if args.checkpoint_dir:
        ckpt_dir = args.checkpoint_dir
    elif args.output:
        ckpt_dir = os.path.dirname(args.output) or '.'
    else:
        ckpt_dir = None

    # Persist results BEFORE printing — if the print block crashes (e.g. on a
    # Unicode-incompatible console), the CV work shouldn't be lost.
    if ckpt_dir is not None:
        save_checkpoints(ckpt_dir, summary_df, per_feat_df, pooled_stats, rows, args.lambdas,
                         write_derived=not args.no_derived_checkpoint)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        summary_df.to_csv(args.output, index=False)
        log(f"[eval_quality] wrote {args.output}")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', lambda x: f"{x:.4f}")

    def _annotated(df, raw_col, mean_col):
        """Return a Series formatted as 'mean (Δ%=±d%, p=0.012*)' for printing."""
        out = []
        for _, r in df.iterrows():
            mean_v = r.get(mean_col)
            ratio = r.get(f'{raw_col}_ratio')
            p = r.get(f'{raw_col}_pvalue')
            s = f"{mean_v:.4f}" if isinstance(mean_v, (int, float)) and np.isfinite(mean_v) else 'nan'
            if isinstance(ratio, (int, float)) and np.isfinite(ratio):
                s += f"  Δ%={ratio:+.2f}%"
            if isinstance(p, (int, float)) and np.isfinite(p):
                s += f"  p={p:.3g}{_sig_marker(p, args.alpha)}"
            out.append(s)
        return out

    print()
    print("=" * 80)
    print(f"USEFULNESS (model-agnostic) — Δ% vs base_features, p = Mann-Whitney U, "
          f"* = p < {args.alpha}")
    print("=" * 80)
    use_view = summary_df[['method', 'lam', 'k_evaluated']].copy()
    use_view['mi'] = _annotated(summary_df, 'mi', 'mi_mean')
    use_view['pearson'] = _annotated(summary_df, 'pearson', 'pearson_mean')
    use_view['fisher'] = _annotated(summary_df, 'fisher', 'fisher_mean')
    print(use_view.to_string(index=False))

    print()
    print("=" * 80)
    print(f"EXPLAINABILITY — lower Kext/DBF = better; Δ% vs base_features, "
          f"p = Mann-Whitney U one-sided, * = p < {args.alpha}")
    print("=" * 80)
    expl_view = summary_df[['method', 'lam', 'k_evaluated', 'kext_source']].copy()
    expl_view['kext'] = _annotated(summary_df, 'kext', 'kext_mean')
    expl_view['dbf'] = _annotated(summary_df, 'dbf', 'dbf_mean')
    print(expl_view.to_string(index=False))

    def _fmt(v):
        return 'nan' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.4f}"
    print()
    print("=" * 80)
    print("COMPOSITE (MI-based)  S(f; lam) = lam*MI_norm + (1-lam)*(1-Kext_norm)")
    print(f"           normalized across {pooled_stats.get('n_features_pool', 0)} pooled features")
    print(f"           MI range   [{_fmt(pooled_stats.get('mi_min'))}, {_fmt(pooled_stats.get('mi_max'))}]")
    print(f"           Kext range [{_fmt(pooled_stats.get('kext_min'))}, {_fmt(pooled_stats.get('kext_max'))}]")
    print("=" * 80)
    comp_mean_cols = ['method', 'lam', 'k_evaluated']
    comp_mean_cols += [f'composite_mean_lam{lam}' for lam in args.lambdas]
    comp_mean_cols += ['composite_mean_at_row_lam']
    print("Mean composite per row:")
    print(summary_df[comp_mean_cols].to_string(index=False))
    print()
    print("Max composite per row:")
    comp_max_cols = ['method', 'lam', 'k_evaluated']
    comp_max_cols += [f'composite_max_lam{lam}' for lam in args.lambdas]
    comp_max_cols += ['composite_max_at_row_lam']
    print(summary_df[comp_max_cols].to_string(index=False))

    print()
    print("=" * 80)
    print("COMPOSITE (Pearson-based)  S(f; lam) = lam*|Pearson|_norm + (1-lam)*(1-Kext_norm)")
    print(f"           normalized across {pooled_stats.get('n_features_pool', 0)} pooled features")
    print(f"           |Pearson| range [{_fmt(pooled_stats.get('pearson_min'))}, {_fmt(pooled_stats.get('pearson_max'))}]")
    print(f"           Kext range      [{_fmt(pooled_stats.get('kext_min'))}, {_fmt(pooled_stats.get('kext_max'))}]")
    print("=" * 80)
    cp_mean_cols = ['method', 'lam', 'k_evaluated']
    cp_mean_cols += [f'composite_pearson_mean_lam{lam}' for lam in args.lambdas]
    cp_mean_cols += ['composite_pearson_mean_at_row_lam']
    print("Mean Pearson-composite per row:")
    print(summary_df[cp_mean_cols].to_string(index=False))
    print()
    print("Max Pearson-composite per row:")
    cp_max_cols = ['method', 'lam', 'k_evaluated']
    cp_max_cols += [f'composite_pearson_max_lam{lam}' for lam in args.lambdas]
    cp_max_cols += ['composite_pearson_max_at_row_lam']
    print(summary_df[cp_max_cols].to_string(index=False))

    print()
    print("=" * 80)
    print(f"DOWNSTREAM EFFECTIVENESS — CV mean ± 90% CI; Δ% vs base_features, "
          f"p = paired t-test one-sided, * = p < {args.alpha}")
    print("=" * 80)

    def _downstream_cell(df, model):
        out = []
        for _, r in df.iterrows():
            mean_v = r.get(f'downstream_{model}')
            ci = r.get(f'downstream_{model}_ci90')
            ratio = r.get(f'downstream_{model}_ratio')
            p = r.get(f'downstream_{model}_pvalue')
            if not isinstance(mean_v, (int, float)) or not np.isfinite(mean_v):
                out.append('nan'); continue
            s = f"{mean_v:.4f}"
            if isinstance(ci, (int, float)) and np.isfinite(ci):
                s += f"±{ci:.3f}"
            if isinstance(ratio, (int, float)) and np.isfinite(ratio):
                s += f"  Δ%={ratio:+.2f}%"
            if isinstance(p, (int, float)) and np.isfinite(p):
                s += f"  p={p:.3g}{_sig_marker(p, args.alpha)}"
            out.append(s)
        return out

    dn_view = summary_df[['method', 'lam', 'k_evaluated', 'baseline_RF_base_only']].copy()
    for m in args.models:
        dn_view[m] = _downstream_cell(summary_df, m)
    dn_view['avg'] = _downstream_cell(summary_df, 'avg')
    print(dn_view.to_string(index=False))

    print()
    print("=" * 80)
    print("MI RANKING (sorted by mi_mean desc; total = composite_mean_at_row_lam, MI-based)")
    print("=" * 80)
    mi_cols = ['method', 'lam', 'k_evaluated', 'mi_mean', 'mi_max',
               'composite_mean_at_row_lam']
    mi_df = summary_df[mi_cols].copy()
    mi_df = mi_df.rename(columns={'composite_mean_at_row_lam': 'total_mi'})
    mi_df = mi_df.sort_values('mi_mean', ascending=False, na_position='last')
    mi_df.insert(0, 'rank', range(1, len(mi_df) + 1))
    print(mi_df.to_string(index=False))

    print()
    print("=" * 80)
    print("PEARSON RANKING (sorted by pearson_mean desc; total = composite_pearson_mean_at_row_lam)")
    print("=" * 80)
    pr_cols = ['method', 'lam', 'k_evaluated', 'pearson_mean', 'pearson_max',
               'composite_pearson_mean_at_row_lam']
    pr_df = summary_df[pr_cols].copy()
    pr_df = pr_df.rename(columns={'composite_pearson_mean_at_row_lam': 'total_pearson'})
    pr_df = pr_df.sort_values('pearson_mean', ascending=False, na_position='last')
    pr_df.insert(0, 'rank', range(1, len(pr_df) + 1))
    print(pr_df.to_string(index=False))

    print()
    print("=" * 80)
    print("TOTAL SCORE per method  (composite = lam*U_norm + (1-lam)*(1-Kext_norm))")
    print("                        U_mi   = MI_norm                  → total_mi_lam<lam>")
    print("                        U_pr   = |Pearson|_norm           → total_pearson_lam<lam>")
    print("                        Means over the row's selected features at each lam in the grid.")
    print("=" * 80)
    total_cols = ['method', 'lam', 'k_evaluated']
    rename_map = {}
    for lam in args.lambdas:
        c_mi = f'composite_mean_lam{lam}'
        c_pr = f'composite_pearson_mean_lam{lam}'
        total_cols += [c_mi, c_pr]
        rename_map[c_mi] = f'total_mi_lam{lam}'
        rename_map[c_pr] = f'total_pearson_lam{lam}'
    total_df = summary_df[total_cols].copy().rename(columns=rename_map)
    print(total_df.to_string(index=False))


if __name__ == '__main__':
    main()
