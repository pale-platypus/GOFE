"""
OpenFE baseline: runs the OpenFE auto-feature-engineering library
(https://github.com/IIIS-Li-Group/OpenFE) on a dataset and evaluates the
augmented feature set with the same sklearn cross-val protocol as SklearnEnv
in autolearn.feat_selection.env.

Output is written to <feat_pool>/results.csv in the same iter,score,top_k,method
shape that iter_train.py uses, so run_compare.sh-style scripts can ingest it.
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


# Silence LightGBM in this (parent) process. Note: OpenFE spawns worker
# subprocesses via ProcessPoolExecutor; on Windows ``spawn`` doesn't inherit
# this registered logger because workers import lightgbm fresh. The only
# reliable way to keep workers quiet is to reduce concurrent worker count
# so the parent's stdout pipe doesn't saturate (see ``run_openfe.sh``
# n_jobs default = 2).
def _silence_lightgbm_here():
    try:
        import lightgbm as _lgb
        class _NullLogger:
            def info(self, msg): pass
            def warning(self, msg): pass
        if hasattr(_lgb, 'register_logger'):
            _lgb.register_logger(_NullLogger())
    except Exception:
        pass

_silence_lightgbm_here()


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


def run_openfe(args):
    try:
        from openfe import OpenFE, transform
    except ImportError as e:
        raise ImportError(
            "OpenFE is not installed. Run `pip install openfe`."
        ) from e
    try:
        from openfe import tree_to_formula
    except ImportError:
        tree_to_formula = None

    dataset = Dataset(args.data)
    X = dataset.instances.copy()
    # OpenFE serializes via feather, which requires string column names; the
    # Dataset class loads CSVs with header=None so columns are int indices.
    # Prefix with 'f' so column names ("f0", "f1", ...) are unambiguous in
    # formula strings (raw digit names collide with integer literals like 2 in
    # x**2 when parsed by sympy/regex for explainability metrics).
    X.columns = [f"f{c}" for c in X.columns]
    y = pd.Series(dataset.labels)
    task = dataset.task

    # Optional held-out test set (fair eval): same base columns, f-prefixed.
    X_test = None
    if getattr(args, 'test_data', ''):
        test_ds = Dataset(args.test_data)
        X_test = test_ds.instances.copy()
        X_test.columns = [f"f{c}" for c in X_test.columns]
        log(f"[OpenFE] held-out test set {args.test_data}: n={len(X_test)}")

    # Categorical / numerical split from dataset.meta. CSV cat columns are
    # typically integer-encoded so a pure-dtype split would mislabel them as
    # numeric — feed OpenFE the meta-derived lists instead so it generates
    # cat-specific candidates (groupby aggregations etc.) on the right cols.
    cat_features = [f"f{k}" for k, v in dataset.meta.items()
                    if k != 'task' and v == 'cat' and f"f{k}" in X.columns]
    num_features = [f"f{k}" for k, v in dataset.meta.items()
                    if k != 'task' and v == 'num' and f"f{k}" in X.columns]
    # Defensive fallback: any X column not in meta (shouldn't happen with the
    # current Dataset loader, but cheap insurance) gets dtype-bucketed.
    seen = set(cat_features) | set(num_features)
    for col in X.columns:
        if col in seen:
            continue
        if pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
            num_features.append(col)
        else:
            cat_features.append(col)

    log(f"[OpenFE] dataset={args.data} task={task} n={len(X)} d={X.shape[1]} "
        f"({len(num_features)} num / {len(cat_features)} cat from meta)")

    baseline = _evaluate(X, y, task, args.eval_model, args.n_estimators, args.cv, args.c_scoring)
    log(f"[OpenFE] baseline (no new features): {baseline:.6f}")

    # Redirect OS file descriptors 1 (stdout) and 2 (stderr) to NUL for the
    # duration of OpenFE's parallel fit, then restore.
    #
    # WHY: OpenFE spawns ProcessPoolExecutor workers that train LightGBM.
    # LightGBM's C++ code logs warnings through a ctypes callback to Python's
    # ``print()`` in the worker, which writes to fd 1. Spawned workers
    # *inherit* the parent's fd 1, so when the parent's fd 1 is a pipe to
    # ``tee`` (and many workers contend on it), Windows surfaces the
    # pipe-saturation as ``errno 28 / ENOSPC``. Replacing the parent's fd 1/2
    # with NUL **before** OpenFE spawns workers means the workers inherit NUL
    # and can dump arbitrary amounts of LightGBM output without contention.
    # Using ``contextlib.redirect_stdout`` is NOT sufficient — that only
    # rebinds the Python-level ``sys.stdout`` in this process; child processes
    # don't see it.
    # If --restrict_ops is set, pre-filter the OpenFE candidate pool down to
    # the listed op names (apples-to-apples with the NFO 4+5 search space).
    candidate_features_list = None
    if args.restrict_ops:
        try:
            from openfe import get_candidate_features
            allowed = {tok.strip() for tok in args.restrict_ops.split(',') if tok.strip()}
            all_cands = get_candidate_features(
                numerical_features=num_features,
                categorical_features=cat_features or None,
                ordinal_features=None,
                order=1,
            )
            candidate_features_list = [c for c in all_cands if c.name in allowed]
            log(f"[OpenFE] op restriction: kept {len(candidate_features_list)}/{len(all_cands)} "
                f"candidates whose op is in {sorted(allowed)}")
            if not candidate_features_list:
                log("[OpenFE] restricted candidate list empty; falling back to default pool")
                candidate_features_list = None
        except Exception as e:
            log(f"[OpenFE] candidate-restriction failed ({type(e).__name__}: {e}); "
                f"falling back to default pool")
            candidate_features_list = None

    t0 = time.time()
    ofe = OpenFE()
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    # OpenFE 0.0.12 (current pinned release) infers numericals as "all columns
    # not in categorical_features" and rejects an explicit numerical_features
    # kwarg. Later versions added it back. Probe the signature so we stay
    # compatible with both — the cat/num split is preserved either way.
    import inspect as _inspect
    _fit_params = _inspect.signature(ofe.fit).parameters
    _fit_kwargs = dict(
        data=X, label=y, task=task,
        n_jobs=args.n_jobs, seed=args.seed, verbose=False,
        candidate_features_list=candidate_features_list,
        categorical_features=cat_features or None,
    )
    if 'numerical_features' in _fit_params:
        _fit_kwargs['numerical_features'] = num_features
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        new_features = ofe.fit(**_fit_kwargs)
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)
    fit_time = time.time() - t0
    log(f"[OpenFE] fit done in {fit_time:.1f}s, {len(new_features)} candidate features generated")

    n_max = min(args.max_top_k, len(new_features))
    best = None
    if n_max < 1:
        log("[OpenFE] no features generated; reporting baseline only")
    else:
        top_k = 1
        while top_k <= n_max:
            try:
                X_aug, _ = transform(X.copy(), X.copy(), new_features[:top_k], n_jobs=args.n_jobs)
                score = _evaluate(X_aug, y, task, args.eval_model, args.n_estimators, args.cv, args.c_scoring)
            except Exception as e:
                # The feature just added (index top_k - 1) broke transform/evaluate
                # (e.g. categorical column that fillna(-1) can't handle). Drop it
                # and retry the same top_k with the remaining features.
                new_features.pop(top_k - 1)
                n_max = min(n_max, len(new_features))
                log(f"[OpenFE] skipping feature at index {top_k - 1}: "
                    f"{type(e).__name__}: {e}")
                continue
            if best is None or score > best[1]:
                best = (top_k, score)
            log(f"[OpenFE] top_k={top_k} score={score:.6f}")
            top_k += 1

    if best is None:
        best = (0, baseline)

    log(f"[OpenFE] best {best[1]:.6f} at top_k={best[0]} "
        f"(baseline {baseline:.6f}, delta {best[1] - baseline:+.6f})")

    if args.feat_pool:
        os.makedirs(args.feat_pool, exist_ok=True)
        results_path = os.path.join(args.feat_pool, 'results.csv')
        if not os.path.exists(results_path):
            with open(results_path, 'w') as f:
                f.write("iter,score,top_k,method\n")
        with open(results_path, 'a') as f:
            f.write(f"0,{best[1]},{best[0]},openfe\n")
        log(f"[OpenFE] appended result to {results_path}")

        formulas = []
        if tree_to_formula is not None and n_max > 0:
            feats_path = os.path.join(args.feat_pool, 'openfe_features.txt')
            with open(feats_path, 'w') as f:
                for i, feat in enumerate(new_features[:n_max]):
                    try:
                        formula = tree_to_formula(feat)
                    except Exception as ex:
                        formula = f"<unprintable: {ex}>"
                    f.write(f"{i + 1}\t{formula}\n")
                    formulas.append(formula)
            log(f"[OpenFE] wrote feature expressions to {feats_path}")

        # Always write eval_manifest.json — even when OpenFE finds nothing
        # that beats the baseline. Otherwise run_quality_compare.sh's
        # skip-if-cached check (which looks for eval_manifest.json) keeps
        # re-running OpenFE every sweep on datasets where it produces no
        # useful features.
        try:
            if n_max > 0 and best[0] > 0:
                k_keep = min(args.max_top_k, len(new_features))
                # Transform BOTH train and the held-out test (if given) with the
                # SAME discovered features, using OpenFE's own computation.
                te_in = X_test.copy() if X_test is not None else X.copy()
                X_aug_full, X_te_aug = transform(X.copy(), te_in,
                                                 new_features[:k_keep], n_jobs=args.n_jobs)
                derived = X_aug_full.iloc[:, X.shape[1]:].copy()
                derived.columns = [f"new_{i}" for i in range(derived.shape[1])]
                if X_test is not None:
                    derived_test = X_te_aug.iloc[:, X_test.shape[1]:].copy()
                    derived_test.columns = [f"new_{i}" for i in range(derived_test.shape[1])]
                    dt_path = os.path.join(args.feat_pool, 'derived_test_columns.csv')
                    derived_test.to_csv(dt_path, index=False)
                    log(f"[OpenFE] wrote held-out test columns to {dt_path} "
                        f"({derived_test.shape[1]} cols x {len(derived_test)} rows)")
            else:
                derived = pd.DataFrame()
                log("[OpenFE] no new features beat baseline; writing empty manifest "
                    "so the sweep records 'ran, found nothing' instead of retrying")
            _write_baseline_manifest(
                args.feat_pool, dataset=args.data, method='openfe',
                derived=derived, formulas=formulas[:derived.shape[1]],
            )
        except Exception as e:
            log(f"[OpenFE] failed to write eval_manifest: {type(e).__name__}: {e}")

    return best


def _write_baseline_manifest(out_dir, *, dataset, method, derived, formulas):
    """Common writer used by all baselines: derived_columns.csv + eval_manifest.json.

    When ``derived`` has zero columns, the derived_columns.csv is skipped
    and ``columns_path`` is None — pd.read_csv chokes on a truly empty CSV
    (no header, no rows), so eval_quality.load_manifest sees columns_path
    is None and treats the row as "ran, produced no features".
    """
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
    import json as _json
    with open(manifest_path, 'w') as f:
        _json.dump(manifest, f, indent=2)
    log(f"[{method}] wrote eval_manifest to {manifest_path} "
        f"(n_features={n_features})")


def parse_args():
    parser = argparse.ArgumentParser(description='OpenFE baseline')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval_model', type=str, default='RF')
    parser.add_argument('--c_scoring', type=str, default='f1_micro')
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--n_estimators', type=int, default=10)
    parser.add_argument('--n_jobs', type=int, default=8)
    parser.add_argument('--max_top_k', type=int, default=32,
                        help='max number of OpenFE-generated features to sweep over')
    parser.add_argument('--feat_pool', type=str, default=None,
                        help='directory to write results.csv; if None, only logs')
    parser.add_argument(
        '--restrict_ops', type=str, default=None,
        help='comma-separated allowed OpenFE op names (e.g. "log,sqrt,+,-,*,/"). '
             'When set, the OpenFE candidate list is pre-filtered to these ops '
             'before the search, matching the NFO operator set for apples-to-apples '
             'comparison. When unset, OpenFE uses its default candidate pool.'
    )
    parser.add_argument(
        '--test_data', type=str, default='',
        help='held-out test dataset name (e.g. <name>__te). When set, OpenFE '
             'fits on --data (train) and ALSO transforms this test set with the '
             'same discovered features, saving derived_test_columns.csv next to '
             'the train manifest — enabling a leak-free held-out-test evaluation.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    run_openfe(parse_args())
