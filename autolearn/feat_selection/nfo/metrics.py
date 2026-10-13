"""Model-agnostic feature scoring used by the bi-objective B-MAFE framework.

Two families of scores:

  Usefulness U(f):
    - mutual_information               (MI; KSG-style via sklearn)
    - conditional_mutual_information   (CMI; MI vs base-feature residual)
    - pearson                          (absolute Pearson correlation, |rho|)
    - fisher                           (univariate F-statistic of y ~ a + b*f)

  Complexity / explainability E(f):
    - kext                (Eq. 5+6: K(f) plus depth-imbalance penalty)
    - dbf                 (Eq. 8: distinct base features)

All functions accept either:
  * a numpy column for f (when scoring usefulness against y), or
  * a feature_str / FeatNode tree (when scoring complexity).

Returned values are raw (not normalized). The composite score
S(f; lambda) = lambda * U_norm + (1 - lambda) * E_norm is computed by the
caller after MinMax-normalizing U and E across the pool — see
``composite_from_pool``.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


# ---------------------------------------------------------------------------
# Usefulness
# ---------------------------------------------------------------------------

def _clean(col: np.ndarray) -> np.ndarray:
    """Replace inf/-inf/NaN so sklearn estimators don't choke on operator output."""
    col = np.asarray(col, dtype=np.float64).reshape(-1)
    col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
    return col


def mutual_information(col: np.ndarray, y: np.ndarray, task: str, *, random_state: int = 0) -> float:
    """KSG-style MI between a single feature column and the target.

    For classification we use mutual_info_classif (KNN-based, robust to
    discrete/continuous mix). For regression we use mutual_info_regression.
    Returns a non-negative float; 0 means independence.
    """
    f = _clean(col).reshape(-1, 1)
    y = np.asarray(y).reshape(-1)
    if task == 'classification':
        mi = mutual_info_classif(f, y, random_state=random_state)
    else:
        mi = mutual_info_regression(f, y, random_state=random_state)
    return float(mi[0])


def residual_from_base(
    X_base: np.ndarray,
    y: np.ndarray,
    task: str,
    *,
    n_estimators: int = 10,
    cv: int = 5,
    random_state: int = 0,
):
    """Out-of-fold residual of ``y`` predicted from the base feature matrix.

    Used as the target for conditional mutual information: a candidate
    feature's usefulness is measured by how well it predicts the *residual*
    — i.e., what the base features alone could not explain about ``y``.

    Regression
    ----------
    Fits ``RandomForestRegressor`` with out-of-fold predictions and returns
    ``r = y - y_hat`` (signed). High |r| means the base RF was uncertain or
    wrong for that row.

    Classification
    --------------
    Fits ``RandomForestClassifier`` with out-of-fold ``predict_proba`` and
    returns ``r_i = 1 - p_hat(y_true_i | X_base_i)`` (in [0, 1]). Zero = base
    RF confidently correct; near 1 = base RF wrong / uncertain.

    ``cv`` is automatically reduced if the minimum class count is smaller
    than the requested fold count (classification only).
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold

    X = np.asarray(X_base, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y).reshape(-1)

    if task == 'classification':
        from collections import Counter
        class_counts = Counter(y)
        min_class = min(class_counts.values()) if class_counts else 0
        # cross_val_predict with StratifiedKFold requires cv <= min class size.
        eff_cv = max(2, min(cv, int(min_class))) if min_class >= 2 else 2
        if eff_cv != cv:
            from autolearn.utils import log as _log
            _log(f"[CMI] reducing cv from {cv} to {eff_cv} "
                 f"(min class count = {min_class})")
        splitter = StratifiedKFold(n_splits=eff_cv, shuffle=True, random_state=random_state)
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        proba = cross_val_predict(clf, X, y, cv=splitter, method='predict_proba')
        classes = np.unique(y)
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        true_idx = np.array([cls_to_idx[c] for c in y], dtype=np.int64)
        p_true = proba[np.arange(len(y)), true_idx]
        residual = 1.0 - p_true
        return residual.astype(np.float64)
    else:
        eff_cv = max(2, int(cv))
        splitter = KFold(n_splits=eff_cv, shuffle=True, random_state=random_state)
        reg = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        y_hat = cross_val_predict(reg, X, y.astype(np.float64), cv=splitter)
        return (y.astype(np.float64) - y_hat).astype(np.float64)


def conditional_mutual_information(
    col: np.ndarray,
    residual: np.ndarray,
    *,
    random_state: int = 0,
) -> float:
    """MI between a candidate feature and the base-feature residual.

    ``residual`` is the output of ``residual_from_base``; it is always
    treated as continuous, so we use ``mutual_info_regression`` regardless
    of the original task. Returns a non-negative float; 0 means the feature
    adds nothing beyond what the base features already capture.
    """
    f = _clean(col).reshape(-1, 1)
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    if r.std() < 1e-12:
        return 0.0
    mi = mutual_info_regression(f, r, random_state=random_state)
    return float(mi[0])


def pearson(col: np.ndarray, y: np.ndarray) -> float:
    """Absolute Pearson correlation in [0, 1]; NaN-safe (returns 0 if degenerate)."""
    f = _clean(col)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if f.std() == 0 or y.std() == 0:
        return 0.0
    rho = np.corrcoef(f, y)[0, 1]
    if not np.isfinite(rho):
        return 0.0
    return float(abs(rho))


def fisher_score(col: np.ndarray, y: np.ndarray) -> float:
    """Univariate F-statistic of y ~ a + b*f (Eq. 4).

    F = SS_reg / (SS_res / (N - 2)). Returns 0 for degenerate input.
    """
    f = _clean(col)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = f.shape[0]
    if n < 3 or f.std() == 0:
        return 0.0
    f_mean, y_mean = f.mean(), y.mean()
    fc, yc = f - f_mean, y - y_mean
    denom = (fc * fc).sum()
    if denom == 0:
        return 0.0
    beta = (fc * yc).sum() / denom
    alpha = y_mean - beta * f_mean
    y_hat = alpha + beta * f
    ss_reg = ((y_hat - y_mean) ** 2).sum()
    ss_res = ((y - y_hat) ** 2).sum()
    if ss_res <= 0:
        return float('inf')
    return float(ss_reg / (ss_res / (n - 2)))


def _otsu(col: np.ndarray, bins: int = 256):
    """(score, threshold) of the best single-threshold split (Otsu).

    score = sigma^2_between(t*) / sigma^2_total in [0, 1] — a label-free
    measure of how cleanly the feature's distribution splits in two.
    Returns (0.0, nan) for degenerate columns.
    """
    f = _clean(col)
    lo, hi = float(np.min(f)), float(np.max(f))
    if not np.isfinite(hi - lo) or hi - lo <= 0:
        return 0.0, float('nan')
    hist, edges = np.histogram(f, bins=bins)
    p = hist.astype(np.float64) / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu_cum = np.cumsum(p * centers)
    mu_total = float(mu_cum[-1])
    var_total = float(np.sum(p * (centers - mu_total) ** 2))
    valid = (w0 > 1e-12) & (w1 > 1e-12)
    if var_total <= 0 or not valid.any():
        return 0.0, float('nan')
    mu0 = mu_cum[valid] / w0[valid]
    mu1 = (mu_total - mu_cum[valid]) / w1[valid]
    sigma_b = w0[valid] * w1[valid] * (mu0 - mu1) ** 2
    j = int(np.argmax(sigma_b))
    t = float(edges[1:][valid][j])
    return float(sigma_b[j] / var_total), t


def otsu_score(col: np.ndarray, *, bins: int = 256) -> float:
    """Label-free split quality U(f) for score_mode='split' (see _otsu)."""
    return _otsu(col, bins)[0]


def otsu_threshold(col: np.ndarray, *, bins: int = 256) -> float:
    """The threshold attaining otsu_score — used by the held-out evaluation
    (fit the threshold on train, apply it to test rows)."""
    return _otsu(col, bins)[1]


def anomaly_auc(col: np.ndarray, y: np.ndarray) -> float:
    """|2*AUC - 1| of the raw feature as an anomaly score
    (score_mode='anom_auc'; direction-invariant, rank-based Mann-Whitney).

    y is a binary anomaly indicator with 1 = anomaly. Returns 0 for
    degenerate input (a single class, or a constant column).
    """
    from scipy.stats import rankdata
    f = _clean(col)
    y = np.asarray(y).reshape(-1).astype(int)
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0 or f.std() == 0:
        return 0.0
    r = rankdata(f)
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1)
    return float(abs(2.0 * auc - 1.0))


def _tight_conds(tree, leaf):
    """Per-variable tightened conditions of a leaf's root-to-leaf path.

    A path may test the same feature repeatedly (x1 > 132 AND x1 > 156);
    tightening keeps, per (feature, direction), only the binding threshold.
    Returns a list of (feature_idx, op, threshold) with op in {'>', '<='}.
    """
    t = tree.tree_
    lo, hi = {}, {}   # feature -> binding '>' threshold / '<=' threshold
    node = 0
    while t.children_left[node] != -1:
        f, thr = int(t.feature[node]), float(t.threshold[node])
        left = t.children_left[node]
        stack, in_left = [left], False
        while stack:
            nd = stack.pop()
            if nd == leaf:
                in_left = True
                break
            if t.children_left[nd] != -1:
                stack.extend([t.children_left[nd], t.children_right[nd]])
        if in_left:
            hi[f] = min(hi.get(f, np.inf), thr)
            node = left
        else:
            lo[f] = max(lo.get(f, -np.inf), thr)
            node = t.children_right[node]
    conds = [(f, '>', thr) for f, thr in lo.items()]
    conds += [(f, '<=', thr) for f, thr in hi.items()]
    return conds


def conds_mask(X: np.ndarray, conds, *, negate: bool = False) -> np.ndarray:
    """Boolean mask of rows satisfying the conjunction of ``conds``.

    ``negate=True`` flips every condition individually ('>' <-> '<='), i.e.
    the OPPOSITE-corner region (a conjunction, not the set complement)."""
    X = np.asarray(X, dtype=np.float64)
    m = np.ones(len(X), dtype=bool)
    for f, op, thr in conds:
        if negate:
            op = '<=' if op == '>' else '>'
        m &= (X[:, f] > thr) if op == '>' else (X[:, f] <= thr)
    return m


def conds_str(conds, *, negate: bool = False, feature_names=None) -> str:
    parts = []
    for f, op, thr in conds:
        if negate:
            op = '<=' if op == '>' else '>'
        name = feature_names[f] if feature_names is not None else f'x{f}'
        parts.append(f'{name} {op} {thr:.3g}')
    return ' AND '.join(parts)


def g3_balanced(col: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Class-balanced (1 - g3) separability of two groups by one threshold.

    Best achievable balanced accuracy over all thresholds on ``col`` and
    both directions == (1 + KS)/2, with KS the Kolmogorov-Smirnov distance
    between the two groups' value distributions. Scale-invariant, one pass
    after sorting. Used as U for score_mode='g3out' (rule-conformity
    outliers). Plain g3 (minimum fraction of rows removed so a threshold
    separates the groups perfectly) is dominated by the majority group
    under imbalance -- a constant column would score 1 - minority
    fraction -- hence the per-group balancing."""
    v = np.asarray(col, dtype=np.float64)
    a = v[np.asarray(mask_a, dtype=bool)]
    b = v[np.asarray(mask_b, dtype=bool)]
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.5
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind='mergesort')
    grp = np.concatenate([np.zeros(len(a)), np.ones(len(b))])[order]
    ca = np.cumsum(grp == 0) / float(len(a))
    cb = np.cumsum(grp == 1) / float(len(b))
    v_sorted = allv[order]
    ok = np.r_[v_sorted[1:] != v_sorted[:-1], True]  # cut only between
    ks = float(np.max(np.abs(ca[ok] - cb[ok])))      # distinct values
    return 0.5 * (1.0 + ks)


def ruleconf_ground_truth(X: np.ndarray, y: np.ndarray, *, max_depth: int = 3,
                          min_leaf_frac: float = 0.05, random_state: int = 0,
                          rule_rank: int = 0, min_violations: int = 15,
                          require_disagree: bool = False,
                          feature_names=None):
    """Two-sided rule-conformity ground truth for score_mode='g3out'.

    Fits the same depth-limited tree and ranks leaf rules by |WRAcc|.
    For the ``rule_rank``-th VALID rule A (conjunction) and its opposite
    rule B (every inequality flipped): CONFORMERS follow the rule pair
    (A-rows carrying A's majority label plus B-rows carrying B's majority
    label); VIOLATORS carry the minority label on either side. A rule is
    valid iff opposite-B covers >= the tree's own 5% leaf floor and there
    are >= ``min_violations`` violators; ``require_disagree=True``
    additionally demands the two sides have different majority labels
    (the stricter original variant). Multiclass targets are binarized at
    the median class. Returns mask_a = conformers, mask_b = violators
    (drop-in shape for the env evaluators)."""
    from sklearn.tree import DecisionTreeClassifier

    X = np.asarray(X, dtype=np.float64)
    X = np.where(np.isfinite(X), X, np.nan)
    med = np.nanmedian(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(med, idx[1])
    y = np.asarray(y).reshape(-1).astype(int)
    if len(np.unique(y)) > 2:
        y = (y > np.median(y)).astype(int)
    n = len(y)
    base_rate = float(y.mean())

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=max(1, int(np.ceil(min_leaf_frac * n))),
        random_state=random_state,
    ).fit(X, y)

    leaf_of = tree.apply(X)
    wracc = {}
    for lf in np.unique(leaf_of):
        m = leaf_of == lf
        wracc[int(lf)] = float(m.mean()) * (float(y[m].mean()) - base_rate)

    min_b = max(2, int(np.ceil(min_leaf_frac * n)))
    valid = []
    for lf in sorted(wracc, key=lambda k: abs(wracc[k]), reverse=True):
        c = _tight_conds(tree, lf)
        ma = conds_mask(X, c)
        mb = conds_mask(X, c, negate=True)
        if ma.sum() == 0 or mb.sum() < min_b:
            continue
        maj_a = int(np.bincount(y[ma]).argmax())
        maj_b = int(np.bincount(y[mb]).argmax())
        if require_disagree and maj_a == maj_b:
            continue
        viol = (ma & (y != maj_a)) | (mb & (y != maj_b))
        if viol.sum() >= min_violations:
            valid.append((lf, c, maj_a, maj_b))
    if rule_rank >= len(valid):
        raise ValueError(f'rule_rank={rule_rank}, but only {len(valid)} '
                         'valid two-sided rule pairs exist')
    lf, conds, maj_a, maj_b = valid[rule_rank]
    ma = conds_mask(X, conds)
    mb = conds_mask(X, conds, negate=True)
    conf = (ma & (y == maj_a)) | (mb & (y == maj_b))
    viol = (ma & (y != maj_a)) | (mb & (y != maj_b))
    rule = conds_str(conds, feature_names=feature_names)
    return dict(tree=tree, conds=conds, maj_a=maj_a, maj_b=maj_b, leaf=lf,
                rule=rule, mask_a=conf, mask_b=viol, n_valid=len(valid),
                rule_a=f'followers of [{rule}] and of its opposite',
                rule_b='violators (minority label on either side)',
                wracc_a=wracc[lf], wracc_b=wracc[lf])


def ifout_ground_truth(X: np.ndarray, *, contamination: float = 0.10,
                       n_estimators: int = 200, random_state: int = 0):
    """Detector-referenced outlier ground truth for score_mode='ifout'.

    LABEL-FREE: fits an IsolationForest on X and flags the
    ``contamination`` fraction with the lowest anomaly scores as OUTLIERS
    (mask_b); the rest are INLIERS (mask_a). The train-side score
    threshold is frozen so the evaluator can apply the same verdict to
    held-out rows. The search objective (Cohen's d via subgroup_evaluate)
    asks for ONE readable expression that reproduces the detector's
    multivariate verdict -- explainable anomaly detection."""
    from sklearn.ensemble import IsolationForest

    X = np.asarray(X, dtype=np.float64)
    X = np.where(np.isfinite(X), X, np.nan)
    med = np.nanmedian(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(med, idx[1])
    det = IsolationForest(n_estimators=n_estimators,
                          random_state=random_state).fit(X)
    s = det.score_samples(X)
    thr = float(np.quantile(s, contamination))
    mask_o = s <= thr
    return dict(detector=det, thr=thr, contamination=contamination,
                mask_a=~mask_o, mask_b=mask_o,
                rule_a=f'IForest inliers (score > {thr:.4f})',
                rule_b=f'IForest outliers (bottom {contamination:.0%})',
                wracc_a=0.0, wracc_b=0.0)


def normout_ground_truth(X: np.ndarray, y: np.ndarray, *, max_depth: int = 3,
                         min_leaf_frac: float = 0.05, random_state: int = 0,
                         rule_rank: int = 0, min_outliers: int = 15,
                         feature_names=None):
    """Two-rule normal/outlier ground truth for score_mode='normout'.

    Picks TWO discovered tree rules per pair, one predicting each class.
    Candidate pairs are ALL (class-1 rule, class-0 rule) combinations,
    ranked by combined rule strength (WRAcc contrast, descending);
    ``rule_rank`` selects the r-th valid pair. NORMAL
    rows follow either rule (covered by one of the two rules and carrying
    that rule's majority label); OUTLIER rows are covered but
    misclassified by their rule. Rows covered by neither rule are
    excluded. The search objective maximizes the distance (Cohen's d)
    between the two groups, reusing subgroup_evaluate. A pair is valid
    iff it has >= ``min_outliers`` outliers. Multiclass targets are
    binarized at the median class."""
    from sklearn.tree import DecisionTreeClassifier

    X = np.asarray(X, dtype=np.float64)
    X = np.where(np.isfinite(X), X, np.nan)
    med = np.nanmedian(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(med, idx[1])
    y = np.asarray(y).reshape(-1).astype(int)
    if len(np.unique(y)) > 2:
        y = (y > np.median(y)).astype(int)
    n = len(y)
    base_rate = float(y.mean())

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=max(1, int(np.ceil(min_leaf_frac * n))),
        random_state=random_state,
    ).fit(X, y)

    leaf_of = tree.apply(X)
    wracc, maj = {}, {}
    for lf in np.unique(leaf_of):
        m = leaf_of == lf
        wracc[int(lf)] = float(m.mean()) * (float(y[m].mean()) - base_rate)
        maj[int(lf)] = int(np.bincount(y[m]).argmax())
    pos = [lf for lf in wracc if maj[lf] == 1]  # rules predicting class 1
    neg = [lf for lf in wracc if maj[lf] == 0]  # rules predicting class 0
    combos = sorted(((la, lb) for la in pos for lb in neg),
                    key=lambda p: wracc[p[0]] - wracc[p[1]], reverse=True)

    valid = []
    for la, lb in combos:
        ma, mb = leaf_of == la, leaf_of == lb
        out = (ma & (y != maj[la])) | (mb & (y != maj[lb]))
        if out.sum() >= min_outliers:
            valid.append((la, lb))
    if rule_rank >= len(valid):
        raise ValueError(f'rule_rank={rule_rank}, but only {len(valid)} '
                         'valid two-rule normal/outlier pairs exist')
    la, lb = valid[rule_rank]
    ma, mb = leaf_of == la, leaf_of == lb
    normal = (ma & (y == maj[la])) | (mb & (y == maj[lb]))
    outlier = (ma & (y != maj[la])) | (mb & (y != maj[lb]))
    ra = conds_str(_tight_conds(tree, la), feature_names=feature_names)
    rb = conds_str(_tight_conds(tree, lb), feature_names=feature_names)
    return dict(tree=tree, leaf_a=la, leaf_b=lb,
                maj_a=maj[la], maj_b=maj[lb], n_valid=len(valid),
                mask_a=normal, mask_b=outlier,
                rule_a=f'[{ra}] => {maj[la]}',
                rule_b=f'[{rb}] => {maj[lb]}',
                wracc_a=wracc[la], wracc_b=wracc[lb])


def outlier_ground_truth(X: np.ndarray, y: np.ndarray, *, max_depth: int = 3,
                         min_leaf_frac: float = 0.05, random_state: int = 0,
                         rule_rank: int = 0, min_exceptions: int = 15,
                         feature_names=None):
    """Rule-non-conformance ground truth for score_mode='rule_out'.

    Fits the same depth-limited tree as ``subgroup_ground_truth`` and ranks
    leaf rules by |WRAcc| (pattern strength, either direction). The
    ``rule_rank``-th VALID rule is interpreted as a ground-truth pattern:
    in-rule rows carrying the rule's majority label are CONFORMERS,
    in-rule rows contradicting it are OUTLIERS (exceptions to the rule).
    A rule is valid only if it has >= ``min_exceptions`` exceptions.
    Multiclass targets are binarized at the median class so 'majority
    label' is meaningful (e.g. wine quality: good vs rest).

    Returns a dict that is a drop-in for SklearnEnv.subgroup_evaluate:
    mask_a = conformers, mask_b = outliers, plus tree/conds/maj/rule for
    the held-out evaluator.
    """
    from sklearn.tree import DecisionTreeClassifier

    X = np.asarray(X, dtype=np.float64)
    X = np.where(np.isfinite(X), X, np.nan)
    med = np.nanmedian(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(med, idx[1])
    y = np.asarray(y).reshape(-1).astype(int)
    if len(np.unique(y)) > 2:
        y = (y > np.median(y)).astype(int)
    n = len(y)
    base_rate = float(y.mean())

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=max(1, int(np.ceil(min_leaf_frac * n))),
        random_state=random_state,
    ).fit(X, y)

    leaf_of = tree.apply(X)
    wracc = {}
    for lf in np.unique(leaf_of):
        m = leaf_of == lf
        wracc[int(lf)] = float(m.mean()) * (float(y[m].mean()) - base_rate)

    valid = []
    for lf in sorted(wracc, key=lambda k: abs(wracc[k]), reverse=True):
        c = _tight_conds(tree, lf)
        mr = conds_mask(X, c)
        if mr.sum() == 0:
            continue
        maj = int(np.bincount(y[mr]).argmax())
        if (mr & (y != maj)).sum() >= min_exceptions:
            valid.append((lf, c, maj))
    if rule_rank >= len(valid):
        raise ValueError(f'rule_rank={rule_rank}, but only {len(valid)} '
                         f'rules have >= {min_exceptions} exceptions')
    lf, conds, maj = valid[rule_rank]
    mask_r = conds_mask(X, conds)
    mask_c = mask_r & (y == maj)
    mask_o = mask_r & (y != maj)
    rule = conds_str(conds, feature_names=feature_names)
    return dict(tree=tree, conds=conds, maj=maj, leaf=lf, rule=rule,
                mask_a=mask_c, mask_b=mask_o, n_valid=len(valid),
                rule_a=rule + f' & y == {maj} (conformers)',
                rule_b=rule + f' & y != {maj} (exceptions)',
                wracc_a=wracc[lf], wracc_b=wracc[lf])


def subgroup_ground_truth(X: np.ndarray, y: np.ndarray, *, max_depth: int = 3,
                          min_leaf_frac: float = 0.05, random_state: int = 0,
                          feature_names=None, opposite: bool = False,
                          rule_rank: int = 0):
    """Rule-based subgroup ground truth for score_mode='subgroup'.

    Fits a single depth-limited decision tree on (X, y) (classification),
    scores every leaf rule by WRAcc = coverage * (leaf positive rate - global
    positive rate), and returns the two OPPOSING rules: the best rule toward
    class 1 (max WRAcc) and the best toward class 0 (min WRAcc). Deterministic
    (fixed random_state); the returned tree lets the evaluator apply the same
    frozen rules to held-out rows via ``tree.apply``.

    Returns dict with: tree, leaf_a, leaf_b, mask_a, mask_b (boolean, on X),
    rule_a, rule_b (readable strings), wracc_a, wracc_b.
    """
    from sklearn.tree import DecisionTreeClassifier

    X = np.asarray(X, dtype=np.float64)
    X = np.where(np.isfinite(X), X, np.nan)
    med = np.nanmedian(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(med, idx[1])
    y = np.asarray(y).reshape(-1).astype(int)
    n = len(y)
    base_rate = float(y.mean())

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=max(1, int(np.ceil(min_leaf_frac * n))),
        random_state=random_state,
    ).fit(X, y)

    leaf_of = tree.apply(X)
    leaves = np.unique(leaf_of)
    wracc = {}
    for lf in leaves:
        m = leaf_of == lf
        cov = float(m.mean())
        pos = float(y[m].mean())
        wracc[int(lf)] = cov * (pos - base_rate)
    ranked = sorted(wracc, key=wracc.get, reverse=True)  # best rule first
    leaf_b = min(wracc, key=wracc.get)   # leans toward class 0
    if len(ranked) < 2:
        raise ValueError('degenerate tree: a single leaf')

    if opposite:
        # OPPOSITE-RULE variant: B is not a second discovered leaf but the
        # opposite corner of A's own (tightened) conditions — every
        # inequality flipped, conjunction kept (e.g. A: x>2 -> B: x<=2).
        # ``rule_rank`` selects the r-th best VALID rule (by WRAcc, filtered
        # to rules whose opposite-B is non-degenerate) as subgroup A — used
        # by the multi-rule experiments (rule_rank in {0, 1, 2}).
        valid = []
        min_b = max(2, int(np.ceil(min_leaf_frac * n)))  # same floor as the
        for lf in ranked:                                # tree's own leaves
            c = _tight_conds(tree, lf)
            mb = conds_mask(X, c, negate=True)
            if mb.sum() >= min_b:
                valid.append((lf, c))
        if rule_rank >= len(valid):
            raise ValueError(f'rule_rank={rule_rank}, but only {len(valid)} '
                             'valid opposite-rule subgroup pairs exist')
        leaf_a, conds = valid[rule_rank]
        mask_a = conds_mask(X, conds)
        mask_b = conds_mask(X, conds, negate=True)
        pos_b = float(y[mask_b].mean())
        return dict(tree=tree, leaf_a=leaf_a, leaf_b=None, conds=conds,
                    opposite=True, mask_a=mask_a, mask_b=mask_b,
                    rule_a=conds_str(conds, feature_names=feature_names),
                    rule_b=conds_str(conds, negate=True,
                                     feature_names=feature_names),
                    wracc_a=wracc[leaf_a],
                    wracc_b=float(mask_b.mean()) * (pos_b - base_rate))

    def rule_str(leaf):
        t = tree.tree_
        # walk from root to the leaf, recording conditions
        path, node = [], 0
        while t.children_left[node] != -1:
            f, thr = t.feature[node], t.threshold[node]
            name = (feature_names[f] if feature_names is not None else f'x{f}')
            left = t.children_left[node]
            # does `leaf` live in the left subtree of `node`?
            stack, in_left = [left], False
            while stack:
                nd = stack.pop()
                if nd == leaf:
                    in_left = True
                    break
                if t.children_left[nd] != -1:
                    stack.extend([t.children_left[nd], t.children_right[nd]])
            if in_left:
                path.append(f'{name} <= {thr:.3g}')
                node = left
            else:
                path.append(f'{name} > {thr:.3g}')
                node = t.children_right[node]
        return ' AND '.join(path)

    return dict(tree=tree, leaf_a=leaf_a, leaf_b=leaf_b,
                mask_a=leaf_of == leaf_a, mask_b=leaf_of == leaf_b,
                rule_a=rule_str(leaf_a), rule_b=rule_str(leaf_b),
                wracc_a=wracc[leaf_a], wracc_b=wracc[leaf_b])


def cohens_d(col: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Standardized mean distance between two subgroups in the feature's
    values (score_mode='subgroup'):

        U_sub(f) = |mean_A f - mean_B f| / s_pooled.

    Scale-invariant by construction (raw distances are gameable by scaling a
    feature). NaN/inf cleaned; zero pooled variance -> 0.
    """
    f = _clean(col)
    a, b = f[mask_a], f[mask_b]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled <= 1e-12:
        return 0.0
    return float(abs(a.mean() - b.mean()) / pooled)


def tail_contrast(col: np.ndarray, *, alpha: float = 0.05) -> float:
    """Label-free tail-separation score (score_mode='anom_tail').

    Robust-standardize (median/MAD, std fallback), then compare the mean
    robust |z| of the top-alpha fraction against the |z| at the (1-alpha)
    boundary: log1p(mean_top / z_boundary - 1). Large when a small tail sits
    far beyond the bulk; 0 for degenerate columns. Note _clean maps inf to 0,
    so expressions that blow up numerically earn no tail credit.
    """
    f = _clean(col)
    med = float(np.median(f))
    mad = float(np.median(np.abs(f - med))) * 1.4826
    scale = mad if mad > 0 else float(f.std())
    if scale <= 0:
        return 0.0
    z = np.abs((f - med) / scale)
    n = len(z)
    k = max(1, int(math.ceil(alpha * n)))
    zs = np.sort(z)
    boundary = float(zs[max(0, n - k - 1)])
    top = float(zs[n - k:].mean())
    if boundary <= 1e-12:
        return 0.0
    ratio = top / boundary - 1.0
    if not np.isfinite(ratio) or ratio <= 0:
        return 0.0
    return float(np.log1p(ratio))


# ---------------------------------------------------------------------------
# Complexity
# ---------------------------------------------------------------------------

def _walk_leaves_with_depth(node, depth: int = 0):
    """Yield (leaf_node, depth) for every leaf reachable from ``node``."""
    if node is None:
        return
    if node.arity == 0:
        yield node, depth
        return
    for child in node.children:
        yield from _walk_leaves_with_depth(child, depth + 1)


def _count_nodes(node):
    """Return (n_leaves, n_op_nodes) for the subtree rooted at ``node``."""
    if node is None:
        return 0, 0
    if node.arity == 0:
        return 1, 0
    leaves, ops = 0, 1  # this node is an operator
    for child in node.children:
        l, o = _count_nodes(child)
        leaves += l
        ops += o
    return leaves, ops


def kext(tree, n_base_features: int, n_operators: int) -> float:
    """K_ext(f) per Eq. 6: K(f) + log2(s) * d_tilde(f).

    Args:
        tree: a FeatNode root (from generate_tree_from_str)
        n_base_features: |X| — size of the base feature vocabulary
        n_operators: |O| — size of the operator vocabulary

    Returns 0 for a single-leaf (no-op) feature.
    """
    s, n_ops = _count_nodes(tree)
    if s <= 1 and n_ops == 0:
        # bare base feature: K = log2(n) bits to name it, no imbalance term
        if n_base_features <= 1:
            return 0.0
        return float(math.log2(n_base_features))

    # depth-imbalance penalty (Eq. 6 + 7)
    d_tilde = 0.0
    if s > 1:
        leaf_depths = [d for _, d in _walk_leaves_with_depth(tree)]
        d_bar = sum(leaf_depths) / s
        d_min = math.log2(s)
        d_max = s - 1
        denom = d_max - d_min
        if denom > 0:
            d_tilde = max(0.0, min(1.0, (d_bar - d_min) / denom))

    return _kext_value(s, n_ops, n_base_features, n_operators, d_tilde=d_tilde)


def dbf(tree) -> int:
    """Distinct base features (Eq. 8): count of unique leaves used in the tree."""
    return len({leaf.attr for leaf, _ in _walk_leaves_with_depth(tree)})


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def minmax_normalize(values: Sequence[float]) -> np.ndarray:
    """Map values to [0, 1]; if all equal returns zeros."""
    arr = np.asarray(values, dtype=np.float64)
    lo, hi = arr.min(), arr.max()
    if hi - lo <= 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _kext_value(s: int, n_ops: int, n_base: int, n_operators: int,
                d_tilde: float = 0.0) -> float:
    """Eq. 5+6 with a given (s, n_ops, d_tilde). Shared by tree and string paths."""
    log_n = math.log2(max(n_base, 2))
    log_o = math.log2(max(n_operators, 2))
    K = s * log_n + n_ops * log_o
    if s > 1 and d_tilde > 0:
        K += math.log2(s) * d_tilde
    return float(K)


def kext_dbf_from_sympy(expr, n_base: int, n_operators: int):
    """Compute (Kext, DBF) from a sympy expression.

    n_leaves = count of leaf symbols (with repetition).
    n_ops    = count of internal nodes (operators + functions).
    The depth-imbalance penalty is omitted (would require an explicit tree
    walk over sympy's args; the dominant terms are the base K(f)).
    """
    import sympy
    symbols = []
    n_op = 0
    for node in sympy.preorder_traversal(expr):
        if node.is_Symbol:
            symbols.append(node)
        elif node.is_Number:
            # numeric leaves are not base features; treat as no-op leaves
            symbols.append(node)
        elif isinstance(node, sympy.Atom):
            symbols.append(node)
        else:
            # internal node: any Add, Mul, Pow, Function call, etc.
            n_op += 1
    s = max(len(symbols), 1)
    K = _kext_value(s, n_op, n_base, n_operators)
    base_syms = {sym for sym in symbols if isinstance(sym, sympy.Symbol)}
    return K, len(base_syms)


_BASE_FEAT_RE = re.compile(r'(?<![A-Za-z_])(?:col_|f|x)?(\d+)(?![A-Za-z_0-9])')
_OP_TOKEN_RE = re.compile(
    r'[+\-*/^]|\b(?:log|sqrt|exp|abs|sigmoid|min_max|minmax|reciprocal|sqrt_abs|'
    r'division|mod_column|count_hash|nunique|cat2num_mean|inv|mod|'
    r'GroupByThen[A-Za-z]+|Combine[A-Za-z]*|NthMostFrequent|freq|round|square)\b'
)


def kext_dbf_from_formula(formula: str, n_base: int, n_operators: int,
                           *, base_feature_names=None):
    """Heuristic (Kext, DBF) from a free-form formula string.

    Tries sympy first; falls back to regex tokenization that counts operator
    occurrences and distinct base-feature references. Useful for baseline
    outputs (OpenFE / AutoFeat formulas) whose strings are not parseable
    into the DIFER tree format.

    base_feature_names: optional iterable of known base feature names (strings).
        When provided, only matches that are in this set are counted as base
        features; everything else is operators/literals. When omitted, any
        integer-looking token is treated as a base feature.
    """
    if not formula or not isinstance(formula, str):
        return float('nan'), float('nan')

    # Try sympy first — works for AutoFeat and simple algebraic expressions.
    # When base feature names are pure digits (this codebase: features are
    # named "0","1",..."n-1"), sympy treats them as integer literals. To keep
    # them as symbols we substitute digit-only names with a `b<digit>_` prefix
    # before parsing, then map back.
    try:
        import sympy
        cleaned = formula
        local_dict = {}
        if base_feature_names is not None:
            names = sorted({str(n) for n in base_feature_names}, key=len, reverse=True)
            for name in names:
                if name.isdigit():
                    alias = f"b{name}_"
                    cleaned = re.sub(rf'(?<![A-Za-z_0-9]){re.escape(name)}(?![A-Za-z_0-9])',
                                     alias, cleaned)
                    local_dict[alias] = sympy.Symbol(alias)
                else:
                    local_dict[name] = sympy.Symbol(name)
        expr = sympy.sympify(cleaned, locals=local_dict)
        return kext_dbf_from_sympy(expr, n_base, n_operators)
    except Exception:
        pass

    # Fallback regex heuristic.
    base_matches = _BASE_FEAT_RE.findall(formula)
    if base_feature_names is not None:
        name_set = {str(n) for n in base_feature_names}
        base_matches = [m for m in base_matches if m in name_set]
    if not base_matches:
        return float('nan'), float('nan')
    s = len(base_matches)
    op_matches = _OP_TOKEN_RE.findall(formula)
    n_op = len(op_matches)
    K = _kext_value(s, n_op, n_base, n_operators)
    return K, len(set(base_matches))


def composite_from_pool(
    mi_scores: Sequence[float],
    kext_scores: Sequence[float],
    lam: float,
) -> np.ndarray:
    """S(f; lambda) = lambda * U_norm + (1 - lambda) * (1 - K_norm).

    Higher is better for both terms after normalization.
    """
    u_norm = minmax_normalize(mi_scores)
    k_norm = minmax_normalize(kext_scores)
    e_norm = 1.0 - k_norm
    return lam * u_norm + (1.0 - lam) * e_norm


# ---------------------------------------------------------------------------
# Convenience: score a list of feature strings against a dataset
# ---------------------------------------------------------------------------

def score_features(
    feature_strs: Iterable[str],
    env,
    *,
    metrics: Sequence[str] = ('mi', 'pearson', 'fisher', 'kext', 'dbf'),
) -> dict:
    """Compute requested metrics for each feature string.

    Returns a dict {metric_name: list_of_floats} aligned with feature_strs.
    Invalid features get NaN.
    """
    from autolearn.feat_selection.nfo.feat_tree import generate_tree_from_str

    feature_strs = list(feature_strs)
    out = {m: [float('nan')] * len(feature_strs) for m in metrics}

    y = np.asarray(env.dataset.labels)
    task = env.dataset.task
    n_base = len(env.features)
    n_ops = len(env.ops)

    trees = []
    valid = []
    for i, fs in enumerate(feature_strs):
        try:
            tree = generate_tree_from_str([fs], env)[0]
            trees.append(tree)
            valid.append(i)
        except Exception:
            trees.append(None)

    for i, tree in zip(valid, [trees[i] for i in valid]):
        if tree is None:
            continue
        # complexity metrics work on the tree
        if 'kext' in metrics:
            try:
                out['kext'][i] = kext(tree, n_base, n_ops)
            except Exception:
                pass
        if 'dbf' in metrics:
            try:
                out['dbf'][i] = float(dbf(tree))
            except Exception:
                pass

        # usefulness metrics need the materialized column
        need_col = any(m in metrics for m in ('mi', 'pearson', 'fisher'))
        if need_col:
            try:
                col = env.construct_feature(tree)
            except Exception:
                continue
            if 'mi' in metrics:
                try:
                    out['mi'][i] = mutual_information(col, y, task)
                except Exception:
                    pass
            if 'pearson' in metrics:
                try:
                    out['pearson'][i] = pearson(col, y)
                except Exception:
                    pass
            if 'fisher' in metrics:
                try:
                    out['fisher'][i] = fisher_score(col, y)
                except Exception:
                    pass
    return out
