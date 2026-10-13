"""
AutoFeat baseline: runs the AutoFeat auto-feature-engineering library
(https://github.com/cod3licious/autofeat) on a dataset and evaluates the
augmented feature set with the same sklearn cross-val protocol as SklearnEnv
in autolearn.feat_selection.env.

AutoFeat performs feature engineering and selection internally, so we report a
single iter,score,top_k,method row where top_k is the number of new features
AutoFeat selected. Output shape matches openfe_baseline.py / iter_train.py so
run_compare.sh-style scripts can ingest it.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC, LinearSVR

sys.path.extend(["."])

from autolearn.data import Dataset
from autolearn.metrics import r2_score
from autolearn.utils import log


def _make_model(task, model_name, n_estimators):
    if task == 'regression':
        if model_name == 'LR':
            return Lasso()
        if model_name == 'SVM':
            return LinearSVR()
        if model_name == 'LGB':
            from lightgbm import LGBMRegressor
            return LGBMRegressor()
        if model_name == 'XGB':
            from xgboost import XGBRegressor
            return XGBRegressor()
        return RandomForestRegressor(n_estimators=n_estimators, random_state=0)
    if model_name == 'LR':
        return LogisticRegression()
    if model_name == 'SVM':
        return LinearSVC()
    if model_name == 'LGB':
        from lightgbm import LGBMClassifier
        return LGBMClassifier()
    if model_name == 'XGB':
        from xgboost import XGBClassifier
        return XGBClassifier()
    return RandomForestClassifier(n_estimators=n_estimators, random_state=0)


def _evaluate(x, y, task, model_name, n_estimators, cv, scoring):
    x = x.copy()
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    x.fillna(-1, inplace=True)
    x.columns = x.columns.astype(str)
    model = _make_model(task, model_name, n_estimators)
    scorer = make_scorer(r2_score) if task == 'regression' else scoring
    return cross_val_score(model, x, y, scoring=scorer, cv=int(cv)).mean()


def _split_categorical(X, meta=None):
    """Split column names into (categorical_cols, numeric_cols).

    Prefers ``dataset.meta`` when available: project CSVs are loaded with
    header=None and cat columns are typically already integer-encoded, so a
    pure-dtype split would mislabel them as numeric. Columns are renamed
    upstream to "f<i>", so we map "f3" back to meta key "3". Falls back to
    dtype for any column missing from meta.
    """
    cat_cols, num_cols = [], []
    for col in X.columns:
        m = None
        if meta is not None and isinstance(col, str) and col.startswith('f'):
            m = meta.get(col[1:])
        if m == 'cat':
            cat_cols.append(col)
        elif m == 'num':
            num_cols.append(col)
        elif pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)
    return cat_cols, num_cols


def run_autofeat(args):
    if args.task_type == 'classification':
        try:
            from autofeat import AutoFeatClassifier as AutoFeatModel
        except ImportError as e:
            raise ImportError("AutoFeat is not installed. Run `pip install autofeat`.") from e
    elif args.task_type == 'regression':
        try:
            from autofeat import AutoFeatRegressor as AutoFeatModel
        except ImportError as e:
            raise ImportError("AutoFeat is not installed. Run `pip install autofeat`.") from e
    else:
        AutoFeatModel = None  # resolved below based on dataset.task

    dataset = Dataset(args.data)
    X = dataset.instances.copy()
    # AutoFeat (and feather serialization downstream) require string column names;
    # Dataset loads CSVs with header=None so columns are int indices. Prefix with
    # 'f' so column names ("f0", "f1", ...) are unambiguous in formula strings
    # (raw digit names collide with integer literals like 2 in x**2 when parsed
    # by sympy for explainability metrics).
    X.columns = [f"f{c}" for c in X.columns]
    y = pd.Series(dataset.labels)
    task = dataset.task

    # Optional held-out test set (fair eval): same base columns, f-prefixed.
    X_test = None
    if getattr(args, 'test_data', ''):
        _test_ds = Dataset(args.test_data)
        X_test = _test_ds.instances.copy()
        X_test.columns = [f"f{c}" for c in X_test.columns]
        log(f"[AutoFeat] held-out test set {args.test_data}: n={len(X_test)}")

    if AutoFeatModel is None:
        if task == 'classification':
            from autofeat import AutoFeatClassifier as AutoFeatModel  # noqa: F811
        else:
            from autofeat import AutoFeatRegressor as AutoFeatModel  # noqa: F811

    log(f"[AutoFeat] dataset={args.data} task={task} n={len(X)} d={X.shape[1]}")

    baseline = _evaluate(X, y, task, args.eval_model, args.n_estimators, args.cv, args.c_scoring)
    log(f"[AutoFeat] baseline (no new features): {baseline:.6f}")

    cat_cols, num_cols = _split_categorical(X, meta=dataset.meta)
    log(f"[AutoFeat] {len(num_cols)} numeric / {len(cat_cols)} categorical columns (from dataset.meta)")

    # AutoFeat one-hot encodes everything in `categorical_cols`. With high-
    # cardinality features (e.g. amazon: col 0 = 7518 unique, col 1 = 4243),
    # the combinatorial one-hot expansion blows past memory — we saw the
    # request grow to 58 TiB on amazon. Cap categorical_cols at a reasonable
    # cardinality; demoted cats stay in the frame as integer-coded numerics
    # (AutoFeat composes arithmetic on them, which is the best we can do
    # without making explicit categorical features).
    if cat_cols:
        kept, demoted = [], []
        for c in cat_cols:
            if X[c].nunique() <= args.max_cat_cardinality:
                kept.append(c)
            else:
                demoted.append(c)
        if demoted:
            log(f"[AutoFeat] demoting {len(demoted)} high-cardinality cats "
                f"(>{args.max_cat_cardinality} unique) to numeric: {demoted}")
            num_cols = num_cols + demoted
            cat_cols = kept

    af_kwargs = dict(
        feateng_steps=args.feateng_steps,
        featsel_runs=args.featsel_runs,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )
    if cat_cols:
        af_kwargs['categorical_cols'] = cat_cols
    if num_cols:
        af_kwargs['feateng_cols'] = num_cols
    if args.max_gb is not None:
        af_kwargs['max_gb'] = args.max_gb
    if args.transformations:
        # AutoFeat takes ``transformations`` as a tuple of unary-op short names;
        # binary +/-/*// are always implicit in feateng composition.
        toks = tuple(t.strip() for t in args.transformations.split() if t.strip())
        if toks:
            af_kwargs['transformations'] = toks
            log(f"[AutoFeat] restricted transformations = {toks}")

    t0 = time.time()
    try:
        af = AutoFeatModel(**af_kwargs)
        X_aug = af.fit_transform(X, y)
    except Exception as e:
        log(f"[AutoFeat] fit_transform failed: {type(e).__name__}: {e}")
        log(f"[AutoFeat] reporting baseline only")
        best = (0, baseline)
        if args.feat_pool:
            _write_result(args.feat_pool, best)
        return best
    fit_time = time.time() - t0

    if isinstance(X_aug, np.ndarray):
        X_aug = pd.DataFrame(X_aug)
    new_feats = list(getattr(af, 'new_feat_cols_', []) or [])
    log(f"[AutoFeat] fit done in {fit_time:.1f}s, {len(new_feats)} new features selected")

    try:
        score = _evaluate(X_aug, y, task, args.eval_model, args.n_estimators, args.cv, args.c_scoring)
    except Exception as e:
        log(f"[AutoFeat] evaluation on augmented frame failed: {type(e).__name__}: {e}")
        log(f"[AutoFeat] falling back to baseline")
        best = (0, baseline)
        if args.feat_pool:
            _write_result(args.feat_pool, best)
        return best

    log(f"[AutoFeat] augmented score={score:.6f} (top_k={len(new_feats)})")
    best = (len(new_feats), score) if score > baseline else (0, baseline)
    log(f"[AutoFeat] best {best[1]:.6f} at top_k={best[0]} "
        f"(baseline {baseline:.6f}, delta {best[1] - baseline:+.6f})")

    if args.feat_pool:
        _write_result(args.feat_pool, best)

        formulas_list = []
        if new_feats:
            feats_path = os.path.join(args.feat_pool, 'autofeat_features.txt')
            formulas_map = getattr(af, 'feature_formulas_', {}) or {}
            with open(feats_path, 'w') as f:
                for i, name in enumerate(new_feats):
                    formula = formulas_map.get(name, name)
                    f.write(f"{i + 1}\t{formula}\n")
                    formulas_list.append(str(formula))
            log(f"[AutoFeat] wrote feature expressions to {feats_path}")

        # Always write eval_manifest.json — even with zero new features.
        # Otherwise run_quality_compare.sh's skip-if-cached check (which
        # looks for eval_manifest.json) keeps re-running AutoFeat every
        # sweep on datasets where it finds nothing useful, e.g. when the
        # base feature set already beats anything AutoFeat composes.
        try:
            if new_feats:
                keep = [c for c in new_feats if c in X_aug.columns]
                derived = X_aug.loc[:, keep].copy()
                derived.columns = [f"new_{i}" for i in range(derived.shape[1])]
                # Transform the held-out test with AutoFeat's fitted engineer.
                if X_test is not None:
                    try:
                        X_te_aug = af.transform(X_test)
                        derived_test = X_te_aug.loc[:, [c for c in keep if c in X_te_aug.columns]].copy()
                        derived_test.columns = [f"new_{i}" for i in range(derived_test.shape[1])]
                        dt_path = os.path.join(args.feat_pool, 'derived_test_columns.csv')
                        derived_test.to_csv(dt_path, index=False)
                        log(f"[AutoFeat] wrote held-out test columns to {dt_path} "
                            f"({derived_test.shape[1]} cols x {len(derived_test)} rows)")
                    except Exception as e:
                        log(f"[AutoFeat] test transform failed: {type(e).__name__}: {e}")
            else:
                derived = pd.DataFrame()
                log("[AutoFeat] no new features beat baseline; writing empty manifest "
                    "so the sweep records 'ran, found nothing' instead of retrying")
            _write_baseline_manifest(
                args.feat_pool, dataset=args.data, method='autofeat',
                derived=derived, formulas=formulas_list[:derived.shape[1]],
            )
        except Exception as e:
            log(f"[AutoFeat] failed to write eval_manifest: {type(e).__name__}: {e}")

    return best


def _write_baseline_manifest(out_dir, *, dataset, method, derived, formulas):
    """Same writer used by openfe_baseline; duplicated to avoid cross-imports.

    When ``derived`` has zero columns, the derived_columns.csv is skipped
    and ``columns_path`` is set to None — pd.read_csv chokes on a truly
    empty CSV (no header, no rows), so eval_quality.load_manifest reads
    ``columns_path is None`` and treats the row as "ran, produced no
    features" (downstream falls back to base-only).
    """
    import json as _json
    n_features = int(derived.shape[1])
    columns_path = None
    if n_features > 0:
        cols_path = os.path.join(out_dir, 'derived_columns.csv')
        derived.to_csv(cols_path, index=False)
        columns_path = 'derived_columns.csv'
    manifest = {
        'dataset': dataset,
        'method': method,
        'lam': None,
        'kind': 'columns',
        'columns_path': columns_path,
        'n_features': n_features,
        'formulas': formulas,
    }
    manifest_path = os.path.join(out_dir, 'eval_manifest.json')
    with open(manifest_path, 'w') as f:
        _json.dump(manifest, f, indent=2)
    log(f"[{method}] wrote eval_manifest to {manifest_path} "
        f"(n_features={n_features})")


def _write_result(feat_pool, best):
    os.makedirs(feat_pool, exist_ok=True)
    results_path = os.path.join(feat_pool, 'results.csv')
    if not os.path.exists(results_path):
        with open(results_path, 'w') as f:
            f.write("iter,score,top_k,method\n")
    with open(results_path, 'a') as f:
        f.write(f"0,{best[1]},{best[0]},autofeat\n")
    log(f"[AutoFeat] appended result to {results_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='AutoFeat baseline')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval_model', type=str, default='RF')
    parser.add_argument('--c_scoring', type=str, default='f1_micro')
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--n_estimators', type=int, default=10)
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--feateng_steps', type=int, default=3,
                        help='AutoFeat feateng_steps (depth of feature composition; '
                             'upstream default is 2)')
    parser.add_argument('--featsel_runs', type=int, default=3,
                        help='AutoFeat featsel_runs (number of selection runs)')
    parser.add_argument('--max_gb', type=float, default=None,
                        help='AutoFeat memory cap in GB (None = unlimited)')
    parser.add_argument(
        '--max_cat_cardinality', type=int, default=64,
        help='Categorical features with more than this many unique values are '
             'demoted to numeric inputs (label-coded ints) instead of being '
             'one-hot encoded. Prevents amazon-style OOM where 9 cats with '
             'cardinalities up to 7518 would explode AutoFeat to 488M columns. '
             'Default 64 keeps low-cardinality cats one-hot encoded (where '
             'AutoFeat handles them well) and demotes the rest.'
    )
    parser.add_argument(
        '--transformations', type=str, default=None,
        help='space-separated AutoFeat unary transformation names (e.g. "log sqrt 1/"). '
             'When set, restricts AutoFeat to these unary ops only; binary +/-/*// '
             'are always implicit in feateng composition. Matches the NFO search '
             'space when set to "log sqrt 1/".'
    )
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument(
        '--test_data', type=str, default='',
        help='held-out test dataset name (e.g. <name>__te). When set, AutoFeat '
             'fits on --data (train) and transforms this test set with its fitted '
             'engineer, saving derived_test_columns.csv for leak-free held-out eval.')
    parser.add_argument('--task_type', type=str, default=None,
                        choices=[None, 'classification', 'regression'],
                        help='override task type; defaults to dataset metadata')
    parser.add_argument('--feat_pool', type=str, default=None,
                        help='directory to write results.csv; if None, only logs')
    return parser.parse_args()


if __name__ == '__main__':
    run_autofeat(parse_args())
