"""Held-out PURPOSE evaluation (plan: paper_drafts/purpose_plan.md).

Two purposes, both under the leakage-free protocol (search + selection on
<name>__tr only; every reported number computed on the untouched __te rows;
padded k=16 policy as in plugin_metric_eval):

(a) SPLIT / data understanding, on pima + german:
    - otsu_te : Otsu split quality of the feature on the TEST rows
                (does the discovered bimodality generalize?)
    - ari/nmi : threshold fitted on TRAIN values (Otsu), applied to TEST
                rows; agreement of the induced 2-way split with the held-out
                class labels, which the 'split' search NEVER saw.

(b) ANOMALY, on pima_anom + spambase_anom (rare-class conversions, 5%):
    - auc_te   : direction-corrected AUROC max(a, 1-a) of the feature as a
                 1-D anomaly score on TEST.
    - iforest  : AUROC of IsolationForest fit on TRAIN [base] vs
                 [base + top-k] (detector is unsupervised; labels only used
                 to score the TEST rows).

Also reports surrogate learnability per new score_mode from the SURR_DUMP
csvs (Spearman rho on the valid split, last iteration) — the pre-registered
go/no-go signal.

Usage:
    python scripts/purpose_eval.py --part all --out_prefix outputs_fair/purpose
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from scripts.fair_test_eval import get_env, read_features, FAIR
from scripts.plugin_metric_eval import (method_columns, random_pool,
                                        pad_with_random, TOP_K)
from autolearn.feat_selection.nfo.metrics import (
    otsu_score, otsu_threshold, anomaly_auc)
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             roc_auc_score)
from sklearn.ensemble import IsolationForest

SPLIT_DATASETS = ['pima', 'german', 'winequality-red', 'housing', 'spambase']
ANOM_DATASETS = ['pima_anom', 'spambase_anom', 'german_anom',
                 'credit_default_anom']

# (display, kind, path template on {d}=<name>__tr)
SPLIT_METHODS = [
    ('B-MAFE(split-CA)', 'expr', 'nfo/RF/lsbo/mi_gat_split_ca/lam1.0/{d}/selected_topk_lam1.0.json'),
    ('B-MAFE(split)',   'expr', 'nfo/RF/lsbo/mi_gat_split/lam1.0/{d}/selected_topk_lam1.0.json'),
    ('B-MAFE(MI)',      'expr', 'nfo/RF/lsbo/mi_gat/lam1.0/{d}/selected_topk_lam1.0.json'),
    ('B-MAFE(Pearson)', 'expr', 'nfo/RF/lsbo/mi_gat_pearson/lam1.0/{d}/selected_topk_lam1.0.json'),
    ('OpenFE',          'cols', 'nfo/RF/openfe/iter/{d}'),
    ('AutoFeat',        'cols', 'nfo/RF/autofeat/iter/{d}'),
    ('DIFER',           'expr', 'nfo/RF/infer/iter_difer_orig/{d}/eval_manifest.json'),
    ('E-AFE',           'expr', 'nfo/RF/eafe/iter/{d}/eval_manifest.json'),
    ('Random',          'expr', 'nfo/RF/random/iter/{d}/eval_manifest.json'),
]
ANOM_METHODS = [
    ('B-MAFE(anom-AUC)',  'expr', 'nfo/RF/lsbo/mi_gat_anom_auc/lam1.0/{d}/selected_topk_lam1.0.json'),
    ('B-MAFE(anom-tail)', 'expr', 'nfo/RF/lsbo/mi_gat_anom_tail/lam1.0/{d}/selected_topk_lam1.0.json'),
    ('B-MAFE(MI)',        'expr', 'nfo/RF/lsbo/mi_gat/lam1.0/{d}/selected_topk_lam1.0.json'),
    ('B-MAFE(DS-RF)',     'expr', 'nfo/RF/lsbo/mi_gat_downstream/lam1.0/{d}/selected_topk_lam1.0.json'),
    ('OpenFE',            'cols', 'nfo/RF/openfe/iter/{d}'),
    ('AutoFeat',          'cols', 'nfo/RF/autofeat/iter/{d}'),
    ('DIFER',             'expr', 'nfo/RF/infer/iter_difer_orig/{d}/eval_manifest.json'),
    ('E-AFE',             'expr', 'nfo/RF/eafe/iter/{d}/eval_manifest.json'),
    ('Random',            'expr', 'nfo/RF/random/iter/{d}/eval_manifest.json'),
]


def _clean_mat(X):
    X = np.asarray(X, dtype=np.float64)
    X[~np.isfinite(X)] = -1.0
    return X


def dir_auc(col, y):
    """Direction-corrected AUROC of a 1-D score; nan-safe."""
    f = np.asarray(col, dtype=np.float64).reshape(-1)
    f[~np.isfinite(f)] = np.nanmedian(f[np.isfinite(f)]) if np.isfinite(f).any() else 0.0
    if len(set(y)) < 2 or np.std(f) == 0:
        return 0.5
    a = roc_auc_score(y, f)
    return float(max(a, 1.0 - a))


def split_metrics(ctr, cte, y_te, cl_te=None):
    """(otsu_te, ari, nmi, ca) for one feature: threshold from TRAIN, split
    TEST. ca = agreement with the train-fitted 2-means structure predicted on
    the test rows (the held-out validation of the split_ca objective)."""
    o = otsu_score(cte)
    t = otsu_threshold(ctr)
    if not np.isfinite(t):
        return o, 0.0, 0.0, 0.0
    cte = np.asarray(cte, dtype=np.float64)
    cte = np.where(np.isfinite(cte), cte, np.nanmedian(cte[np.isfinite(cte)])
                   if np.isfinite(cte).any() else 0.0)
    grp = (cte > t).astype(int)
    if grp.min() == grp.max():
        return o, 0.0, 0.0, 0.0
    ca = (float(adjusted_rand_score(cl_te, grp)) if cl_te is not None else 0.0)
    return (o, float(adjusted_rand_score(y_te, grp)),
            float(normalized_mutual_info_score(y_te, grp)), ca)


def train_clusters(env_tr, env_te):
    """2-means fitted on standardized TRAIN base X, predicted on TEST rows."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    def prep(env, scaler=None):
        X = env.dataset.instances.to_numpy(dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        med = np.nanmedian(X, axis=0)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(med, idx[1])
        if scaler is None:
            scaler = StandardScaler().fit(X)
        return scaler.transform(X), scaler

    Xtr, scaler = prep(env_tr)
    Xte, _ = prep(env_te, scaler)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(Xtr)
    return km.predict(Xte)


def padded_cols(d, nice, kind, tmpl, pool):
    got = method_columns(d, kind, tmpl, top_k=TOP_K)
    if got is None:
        return None
    ctr, cte, own = got
    n_own = len(ctr)
    n_pad = 0
    if nice != 'Random':
        n_pad = pad_with_random(d, nice, ctr, cte, own, pool, TOP_K)
    return ctr, cte, n_own, n_pad


def eval_split(out_prefix):
    rows = []
    for d in SPLIT_DATASETS:
        env_tr, env_te = get_env(d + '__tr'), get_env(d + '__te')
        y_te = np.asarray(env_te.dataset.labels)
        if env_tr.dataset.task == 'regression':
            # label-alignment columns need discrete classes: binarize the
            # continuous target at the TRAIN median (leakage-free)
            med = float(np.median(np.asarray(env_tr.dataset.labels,
                                             dtype=np.float64)))
            y_te = (np.asarray(y_te, dtype=np.float64) > med).astype(int)
        Xtr = env_tr.dataset.instances.to_numpy(dtype=np.float64)
        Xte = env_te.dataset.instances.to_numpy(dtype=np.float64)
        pool = random_pool(d, top_k=TOP_K)
        cl_te = train_clusters(env_tr, env_te)

        def agg(per, **kw):
            return dict(otsu_best=max(p[0] for p in per),
                        otsu_mean=float(np.mean([p[0] for p in per])),
                        ari_best=max(p[1] for p in per),
                        ari_mean=float(np.mean([p[1] for p in per])),
                        nmi_best=max(p[2] for p in per),
                        ca_best=max(p[3] for p in per),
                        ca_mean=float(np.mean([p[3] for p in per])), **kw)

        per = [split_metrics(Xtr[:, j], Xte[:, j], y_te, cl_te)
               for j in range(Xte.shape[1])]
        rows.append(dict(dataset=d, method='Base (best feature)',
                         n_own=Xte.shape[1], n_padded=0, **agg(per)))
        print(f'\n== SPLIT {d} ==', flush=True)
        for nice, kind, tmpl in SPLIT_METHODS:
            got = padded_cols(d, nice, kind, tmpl, pool)
            if got is None:
                print(f'  [miss] {nice}', flush=True)
                continue
            ctr, cte, n_own, n_pad = got
            per = [split_metrics(a, b, y_te, cl_te) for a, b in zip(ctr, cte)]
            r = dict(dataset=d, method=nice, n_own=n_own, n_padded=n_pad,
                     **agg(per))
            rows.append(r)
            print(f"  {nice:17} n={n_own:2d}+{n_pad:2d}  otsu_mean={r['otsu_mean']:.3f}  "
                  f"CA_best={r['ca_best']:+.3f} CA_mean={r['ca_mean']:+.3f}  "
                  f"ARI_best={r['ari_best']:+.3f} NMI_best={r['nmi_best']:.3f}",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{out_prefix}_split.csv', index=False)
    return df


def eval_anom(out_prefix):
    rows = []
    for d in ANOM_DATASETS:
        env_tr, env_te = get_env(d + '__tr'), get_env(d + '__te')
        y_tr = np.asarray(env_tr.dataset.labels).astype(int)
        y_te = np.asarray(env_te.dataset.labels).astype(int)
        Xtr = _clean_mat(env_tr.dataset.instances.to_numpy(dtype=np.float64))
        Xte = _clean_mat(env_te.dataset.instances.to_numpy(dtype=np.float64))
        pool = random_pool(d, top_k=TOP_K)

        def iforest_auc(Atr, Ate):
            clf = IsolationForest(n_estimators=100, random_state=0)
            clf.fit(Atr)
            return float(roc_auc_score(y_te, -clf.score_samples(Ate)))

        base_if = iforest_auc(Xtr, Xte)
        per = [dir_auc(Xte[:, j], y_te) for j in range(Xte.shape[1])]
        rows.append(dict(dataset=d, method='Base (best feature)',
                         n_own=Xte.shape[1], n_padded=0,
                         auc_best=max(per), auc_mean=float(np.mean(per)),
                         iforest=base_if, iforest_delta=0.0))
        print(f'\n== ANOM {d} ==  base IForest AUROC={base_if:.3f}', flush=True)
        for nice, kind, tmpl in ANOM_METHODS:
            got = padded_cols(d, nice, kind, tmpl, pool)
            if got is None:
                print(f'  [miss] {nice}', flush=True)
                continue
            ctr, cte, n_own, n_pad = got
            per = [dir_auc(b, y_te) for b in cte]
            Atr = np.concatenate([Xtr] + [_clean_mat(a).reshape(-1, 1) for a in ctr], axis=1)
            Ate = np.concatenate([Xte] + [_clean_mat(b).reshape(-1, 1) for b in cte], axis=1)
            ifa = iforest_auc(Atr, Ate)
            r = dict(dataset=d, method=nice, n_own=n_own, n_padded=n_pad,
                     auc_best=max(per), auc_mean=float(np.mean(per)),
                     iforest=ifa, iforest_delta=ifa - base_if)
            rows.append(r)
            print(f"  {nice:18} n={n_own:2d}+{n_pad:2d}  AUC_best={r['auc_best']:.3f} "
                  f"AUC_mean={r['auc_mean']:.3f}  IForest={ifa:.3f} "
                  f"(d={r['iforest_delta']:+.3f})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{out_prefix}_anom.csv', index=False)
    return df


def surrogate_report():
    print('\n== Surrogate learnability (valid-split Spearman, last iteration) ==')
    from scipy.stats import spearmanr
    for p in sorted(glob.glob(os.path.join(FAIR, 'surrogate', '*.csv'))):
        df = pd.read_csv(p)
        v = df[df.split == 'valid']
        last = v[v.cur_iter == v.cur_iter.max()]
        pooled = spearmanr(v.true, v.pred).correlation
        rho = spearmanr(last.true, last.pred).correlation
        print(f"  {os.path.basename(p):40} last-iter rho={rho:.3f}  "
              f"pooled rho={pooled:.3f}  (n={len(last)}/{len(v)})")


SUBGROUP_DATASETS = ['pima', 'german', 'winequality-red', 'spambase']
SUBGROUP_METHODS = [
    ('B-MAFE(subgroup)', 'expr', 'nfo/RF/lsbo/mi_gat_subgroup/lam1.0/{d}/selected_topk_lam1.0.json'),
    ('B-MAFE(MI)',       'expr', 'nfo/RF/lsbo/mi_gat/lam1.0/{d}/selected_topk_lam1.0.json'),
    ('B-MAFE(Pearson)',  'expr', 'nfo/RF/lsbo/mi_gat_pearson/lam1.0/{d}/selected_topk_lam1.0.json'),
    ('OpenFE',           'cols', 'nfo/RF/openfe/iter/{d}'),
    ('AutoFeat',         'cols', 'nfo/RF/autofeat/iter/{d}'),
    ('DIFER',            'expr', 'nfo/RF/infer/iter_difer_orig/{d}/eval_manifest.json'),
    ('E-AFE',            'expr', 'nfo/RF/eafe/iter/{d}/eval_manifest.json'),
    ('Random',           'expr', 'nfo/RF/random/iter/{d}/eval_manifest.json'),
]


def eval_subgroup(out_prefix, variant='tree', rules=1):
    """Subgroup purpose (paper_drafts/subgroup_plan.md): held-out separation
    transfer (Cohen's d between the frozen rule-defined subgroups on TEST
    rows) + membership recovery (train-fitted midpoint threshold on the
    best-train-d feature, balanced accuracy on test A/B rows).

    variant='tree'     — B is the opposing WRAcc leaf (original design).
    variant='opposite' — B is the opposite corner of A's own tightened
    conditions (every inequality flipped); the plugged row reads the
    manifests of the search run against THIS objective (subgroup_op), and
    the tree-optimized plugged variant is added as an extra cross-check row.

    rules>1 (opposite only): repeat the whole evaluation for the top
    ``rules`` WRAcc rules (rule_rank=0..rules-1), each with its opposite
    counterpart; the plugged row reads per-rule manifests
    (mi_gat_subgroup_op, then mi_gat_subgroup_op_r{r}); results are written
    per-rule AND averaged over rules per (dataset, method).
    """
    from autolearn.feat_selection.nfo.metrics import (subgroup_ground_truth,
                                                      cohens_d, conds_mask)
    from sklearn.metrics import balanced_accuracy_score

    def prep(env):
        X = env.dataset.instances.to_numpy(dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        med = np.nanmedian(X, axis=0)
        i = np.where(np.isnan(X))
        X[i] = np.take(med, i[1])
        return X

    def recovery(ctr, cte, gtr_a, gtr_b, gte_a, gte_b):
        """Balanced accuracy of the train-fitted midpoint threshold on test
        A/B rows; direction from train means. 0.5 = chance."""
        f_tr = np.asarray(ctr, dtype=np.float64)
        f_tr = np.where(np.isfinite(f_tr), f_tr, np.nanmedian(f_tr[np.isfinite(f_tr)])
                        if np.isfinite(f_tr).any() else 0.0)
        ma, mb = f_tr[gtr_a].mean(), f_tr[gtr_b].mean()
        if not np.isfinite(ma - mb) or ma == mb:
            return 0.5
        thr = (ma + mb) / 2.0
        f_te = np.asarray(cte, dtype=np.float64)
        f_te = np.where(np.isfinite(f_te), f_te, np.nanmedian(f_te[np.isfinite(f_te)])
                        if np.isfinite(f_te).any() else 0.0)
        sel = gte_a | gte_b
        if gte_a.sum() < 2 or gte_b.sum() < 2:
            return float('nan')
        y_true = gte_a[sel].astype(int)
        pred = ((f_te[sel] > thr) == (ma > mb)).astype(int)
        return float(balanced_accuracy_score(y_true, pred))

    D_CAP = 20.0  # Cohen's d is unbounded when a subgroup has ~zero
    # within-group variance (e.g. a clamped 1/x feature constant on one
    # subgroup = a perfect separator); cap for aggregation, disclosed.
    def cdc(col, ma, mb):
        return float(min(cohens_d(np.asarray(col, dtype=np.float64),
                                  ma, mb), D_CAP))

    def methods_for(rule):
        if variant != 'opposite':
            return list(SUBGROUP_METHODS)
        tag = ('mi_gat_subgroup_op' if rule == 0
               else f'mi_gat_subgroup_op_r{rule}')
        plugged = [('B-MAFE(subgroup)', 'expr',
                    'nfo/RF/lsbo/' + tag +
                    '/lam1.0/{d}/selected_topk_lam1.0.json')]
        if rules == 1:  # single-rule run keeps the tree-B cross-check row
            plugged.append(('B-MAFE(subgr-tree)', 'expr',
                            'nfo/RF/lsbo/mi_gat_subgroup/lam1.0/{d}/'
                            'selected_topk_lam1.0.json'))
        # User decision 2026-09-28: the outlier comparisons exclude the
        # generic B-MAFE(MI)/B-MAFE(Pearson) rows.
        drop = {'B-MAFE(subgroup)', 'B-MAFE(MI)', 'B-MAFE(Pearson)'}
        return plugged + [m for m in SUBGROUP_METHODS if m[0] not in drop]

    rows = []
    for d in SUBGROUP_DATASETS:
      env_tr, env_te = get_env(d + '__tr'), get_env(d + '__te')
      Xtr, Xte = prep(env_tr), prep(env_te)
      for rule in range(rules):
        methods = methods_for(rule)
        try:
            gt = subgroup_ground_truth(
                Xtr, np.asarray(env_tr.dataset.labels),
                opposite=(variant == 'opposite'), rule_rank=rule)
        except ValueError as e:
            print(f'[{d}] rule r{rule} unavailable ({e}) -- '
                  f'averaging over the first {rule} rules', flush=True)
            break
        if variant == 'opposite':
            gte_a = conds_mask(Xte, gt['conds'])
            gte_b = conds_mask(Xte, gt['conds'], negate=True)
        else:
            leaf_te = gt['tree'].apply(Xte)
            gte_a = leaf_te == gt['leaf_a']
            gte_b = leaf_te == gt['leaf_b']
        gtr_a, gtr_b = gt['mask_a'], gt['mask_b']
        pool = random_pool(d, top_k=TOP_K)
        print(f"\n== SUBGROUP {d} r{rule} ==  rule A: {gt['rule_a']}  |  "
              f"rule B: {gt['rule_b']}", flush=True)
        print(f"   train A/B = {gtr_a.sum()}/{gtr_b.sum()}   "
              f"test A/B = {gte_a.sum()}/{gte_b.sum()}", flush=True)

        per_b = [(cdc(Xte[:, j], gte_a, gte_b),
                  recovery(Xtr[:, j], Xte[:, j], gtr_a, gtr_b, gte_a, gte_b),
                  cdc(Xtr[:, j], gtr_a, gtr_b))
                 for j in range(Xte.shape[1])]
        jb = int(np.argmax([p[2] for p in per_b]))
        rows.append(dict(dataset=d, rule=rule, method='Base (best feature)',
                         n_own=Xte.shape[1], n_padded=0,
                         d_best=max(p[0] for p in per_b),
                         d_mean=float(np.mean([p[0] for p in per_b])),
                         recov=per_b[jb][1]))
        for nice, kind, tmpl in methods:
            got = padded_cols(d, nice, kind, tmpl, pool)
            if got is None:
                print(f'  [miss] {nice}', flush=True)
                continue
            ctr, cte, n_own, n_pad = got
            per = [(cdc(b, gte_a, gte_b),
                    recovery(a, b, gtr_a, gtr_b, gte_a, gte_b),
                    cdc(a, gtr_a, gtr_b))
                   for a, b in zip(ctr, cte)]
            jt = int(np.argmax([p[2] for p in per]))  # best by TRAIN d
            r = dict(dataset=d, rule=rule, method=nice,
                     n_own=n_own, n_padded=n_pad,
                     d_best=max(p[0] for p in per),
                     d_mean=float(np.mean([p[0] for p in per])),
                     recov=per[jt][1])
            rows.append(r)
            print(f"  {nice:17} n={n_own:2d}+{n_pad:2d}  d_best={r['d_best']:.2f} "
                  f"d_mean={r['d_mean']:.2f}  recovery={r['recov']:.3f}",
                  flush=True)
    df = pd.DataFrame(rows)
    suffix = '_subgroup_op' if variant == 'opposite' else '_subgroup'
    if rules > 1:
        suffix += f'{rules}'
        df.to_csv(f'{out_prefix}{suffix}.csv', index=False)
        avg = (df.groupby(['dataset', 'method'], sort=False)
                 [['d_best', 'd_mean', 'recov']].mean().reset_index())
        avg['n_rules'] = df.groupby(['dataset', 'method'],
                                    sort=False)['rule'].count().values
        avg.to_csv(f'{out_prefix}{suffix}_avg.csv', index=False)
        print(f'\n== AVERAGE over {rules} rules ==')
        for d in SUBGROUP_DATASETS:
            print(f'-- {d} --')
            for _, r in avg[avg.dataset == d].iterrows():
                print(f"  {r['method']:17} d_best={r['d_best']:.2f} "
                      f"d_mean={r['d_mean']:.2f} recov={r['recov']:.3f} "
                      f"({int(r['n_rules'])} rules)")
        return avg
    df.to_csv(f'{out_prefix}{suffix}.csv', index=False)
    return df


def eval_rule_out(out_prefix, rules=5):
    """Rule-non-conformance purpose (score_mode='rule_out'): held-out
    Cohen's d between rule conformers and in-rule exceptions, plus
    balanced accuracy of the train-fitted midpoint threshold flagging
    exceptions on test rows of the rule region. Averaged over all valid
    rules (>= 15 train exceptions), mirroring the multi-rule subgroup
    protocol; d capped at 20 (perfect separators)."""
    from autolearn.feat_selection.nfo.metrics import (outlier_ground_truth,
                                                      cohens_d, conds_mask)
    from sklearn.metrics import balanced_accuracy_score

    D_CAP = 20.0

    def prep(env):
        X = env.dataset.instances.to_numpy(dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        med = np.nanmedian(X, axis=0)
        i = np.where(np.isnan(X))
        X[i] = np.take(med, i[1])
        return X

    def cdc(col, ma, mb):
        return float(min(cohens_d(np.asarray(col, dtype=np.float64),
                                  ma, mb), D_CAP))

    def bacc(ctr, cte, gtr_a, gtr_b, gte_a, gte_b):
        f_tr = np.asarray(ctr, dtype=np.float64)
        f_tr = np.where(np.isfinite(f_tr), f_tr,
                        np.nanmedian(f_tr[np.isfinite(f_tr)])
                        if np.isfinite(f_tr).any() else 0.0)
        ma, mb = f_tr[gtr_a].mean(), f_tr[gtr_b].mean()
        if not np.isfinite(ma - mb) or ma == mb:
            return 0.5
        thr = (ma + mb) / 2.0
        f_te = np.asarray(cte, dtype=np.float64)
        f_te = np.where(np.isfinite(f_te), f_te,
                        np.nanmedian(f_te[np.isfinite(f_te)])
                        if np.isfinite(f_te).any() else 0.0)
        sel = gte_a | gte_b
        if gte_a.sum() < 2 or gte_b.sum() < 2:
            return float('nan')
        y_true = gte_a[sel].astype(int)
        pred = ((f_te[sel] > thr) == (ma > mb)).astype(int)
        return float(balanced_accuracy_score(y_true, pred))

    def methods_for(rule):
        plugged = [('B-MAFE(rule-out)', 'expr',
                    f'nfo/RF/lsbo/mi_gat_rule_out_r{rule}' +
                    '/lam1.0/{d}/selected_topk_lam1.0.json')]
        # User decision 2026-09-28: the outlier comparisons exclude the
        # generic B-MAFE(MI)/B-MAFE(Pearson) rows.
        drop = {'B-MAFE(subgroup)', 'B-MAFE(MI)', 'B-MAFE(Pearson)'}
        return plugged + [m for m in SUBGROUP_METHODS if m[0] not in drop]

    rows = []
    for d in SUBGROUP_DATASETS:
      env_tr, env_te = get_env(d + '__tr'), get_env(d + '__te')
      Xtr, Xte = prep(env_tr), prep(env_te)
      ytr = np.asarray(env_tr.dataset.labels).astype(int)
      yte = np.asarray(env_te.dataset.labels).astype(int)
      if len(np.unique(ytr)) > 2:  # mirror outlier_ground_truth binarization
          yte_b = (yte > np.median(ytr)).astype(int)
      else:
          yte_b = yte
      pool = random_pool(d, top_k=TOP_K)
      for rule in range(rules):
        try:
            gt = outlier_ground_truth(Xtr, ytr, rule_rank=rule)
        except ValueError as e:
            print(f'[{d}] rule r{rule} unavailable ({e})', flush=True)
            break
        gtr_a, gtr_b = gt['mask_a'], gt['mask_b']
        reg_te = conds_mask(Xte, gt['conds'])
        gte_a = reg_te & (yte_b == gt['maj'])
        gte_b = reg_te & (yte_b != gt['maj'])
        print(f"\n== RULE_OUT {d} r{rule} ==  rule: {gt['rule']} "
              f"(maj={gt['maj']})", flush=True)
        print(f"   conform tr/te = {gtr_a.sum()}/{gte_a.sum()}   "
              f"exceptions tr/te = {gtr_b.sum()}/{gte_b.sum()}", flush=True)

        per_b = [(cdc(Xte[:, j], gte_a, gte_b),
                  bacc(Xtr[:, j], Xte[:, j], gtr_a, gtr_b, gte_a, gte_b),
                  cdc(Xtr[:, j], gtr_a, gtr_b))
                 for j in range(Xte.shape[1])]
        jb = int(np.argmax([p[2] for p in per_b]))
        rows.append(dict(dataset=d, rule=rule, method='Base (best feature)',
                         n_own=Xte.shape[1], n_padded=0,
                         d_best=max(p[0] for p in per_b),
                         d_mean=float(np.mean([p[0] for p in per_b])),
                         recov=per_b[jb][1]))
        for nice, kind, tmpl in methods_for(rule):
            got = padded_cols(d, nice, kind, tmpl, pool)
            if got is None:
                print(f'  [miss] {nice}', flush=True)
                continue
            ctr, cte, n_own, n_pad = got
            per = [(cdc(b, gte_a, gte_b),
                    bacc(a, b, gtr_a, gtr_b, gte_a, gte_b),
                    cdc(a, gtr_a, gtr_b))
                   for a, b in zip(ctr, cte)]
            jt = int(np.argmax([p[2] for p in per]))
            r = dict(dataset=d, rule=rule, method=nice,
                     n_own=n_own, n_padded=n_pad,
                     d_best=max(p[0] for p in per),
                     d_mean=float(np.mean([p[0] for p in per])),
                     recov=per[jt][1])
            rows.append(r)
            print(f"  {nice:17} n={n_own:2d}+{n_pad:2d}  "
                  f"d_best={r['d_best']:.2f} d_mean={r['d_mean']:.2f}  "
                  f"acc={r['recov']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f'{out_prefix}_rule_out.csv', index=False)
    avg = (df.groupby(['dataset', 'method'], sort=False)
             [['d_best', 'd_mean', 'recov']].mean().reset_index())
    avg['n_rules'] = df.groupby(['dataset', 'method'],
                                sort=False)['rule'].count().values
    avg.to_csv(f'{out_prefix}_rule_out_avg.csv', index=False)
    print('\n== RULE_OUT AVERAGE over valid rules ==')
    for d in SUBGROUP_DATASETS:
        print(f'-- {d} --')
        for _, r in avg[avg.dataset == d].iterrows():
            print(f"  {r['method']:17} d_best={r['d_best']:.2f} "
                  f"d_mean={r['d_mean']:.2f} acc={r['recov']:.3f} "
                  f"({int(r['n_rules'])} rules)")
    return avg


def eval_g3out(out_prefix, rules=8, tag='mi_gat_g3out2',
               out_suffix='_g3out2'):
    """Two-sided rule-conformity purpose (score_mode='g3out'): U(f) is the
    class-balanced (1 - g3) threshold separability between rule followers
    (of A and of its opposite B) and violators. Held-out columns:
    u_best/u_mean = test-side balanced separability (oracle threshold on
    test, mirroring the d_b/d_m convention), acc = balanced accuracy on
    test of the TRAIN-fitted optimal threshold of the best-by-train-U
    feature. Averaged over all valid rule pairs."""
    from autolearn.feat_selection.nfo.metrics import (ruleconf_ground_truth,
                                                      g3_balanced, conds_mask)

    def prep(env):
        X = env.dataset.instances.to_numpy(dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        med = np.nanmedian(X, axis=0)
        i = np.where(np.isnan(X))
        X[i] = np.take(med, i[1])
        return X

    def fit_thr(col, ma, mb):
        """Train-side optimal threshold + direction for the balanced cut."""
        v = np.asarray(col, dtype=np.float64)
        a, b = v[ma], v[mb]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            return None
        allv = np.concatenate([a, b])
        order = np.argsort(allv, kind='mergesort')
        grp = np.concatenate([np.zeros(len(a)), np.ones(len(b))])[order]
        ca = np.cumsum(grp == 0) / float(len(a))
        cb = np.cumsum(grp == 1) / float(len(b))
        vs = allv[order]
        ok = np.r_[vs[1:] != vs[:-1], True]
        diffs = (ca - cb)[ok]
        pos = np.where(ok)[0][int(np.argmax(np.abs(diffs)))]
        return float(vs[pos]), bool((ca - cb)[pos] > 0)  # True: conf <= thr

    def heldout_acc(ctr, cte, gtr_a, gtr_b, gte_a, gte_b):
        ft = fit_thr(ctr, gtr_a, gtr_b)
        if ft is None or gte_a.sum() < 2 or gte_b.sum() < 2:
            return float('nan')
        thr, conf_low = ft
        v = np.asarray(cte, dtype=np.float64)
        v = np.where(np.isfinite(v), v, thr)
        pred_conf = (v <= thr) if conf_low else (v > thr)
        return 0.5 * (float(pred_conf[gte_a].mean())
                      + float((~pred_conf)[gte_b].mean()))

    def methods_for(rule):
        plugged = [('B-MAFE(g3out)', 'expr',
                    f'nfo/RF/lsbo/{tag}_r{rule}' +
                    '/lam1.0/{d}/selected_topk_lam1.0.json')]
        # User decision 2026-09-28: the outlier comparisons exclude the
        # generic B-MAFE(MI)/B-MAFE(Pearson) rows.
        drop = {'B-MAFE(subgroup)', 'B-MAFE(MI)', 'B-MAFE(Pearson)'}
        return plugged + [m for m in SUBGROUP_METHODS if m[0] not in drop]

    rows = []
    for d in SUBGROUP_DATASETS:
      env_tr, env_te = get_env(d + '__tr'), get_env(d + '__te')
      Xtr, Xte = prep(env_tr), prep(env_te)
      ytr = np.asarray(env_tr.dataset.labels).astype(int)
      yte = np.asarray(env_te.dataset.labels).astype(int)
      if len(np.unique(ytr)) > 2:  # mirror ruleconf_ground_truth binarization
          yte_b = (yte > np.median(ytr)).astype(int)
      else:
          yte_b = yte
      pool = random_pool(d, top_k=TOP_K)
      for rule in range(rules):
        try:
            gt = ruleconf_ground_truth(Xtr, ytr, rule_rank=rule)
        except ValueError as e:
            print(f'[{d}] rule r{rule} unavailable ({e})', flush=True)
            break
        gtr_a, gtr_b = gt['mask_a'], gt['mask_b']
        ma_te = conds_mask(Xte, gt['conds'])
        mb_te = conds_mask(Xte, gt['conds'], negate=True)
        gte_a = ((ma_te & (yte_b == gt['maj_a']))
                 | (mb_te & (yte_b == gt['maj_b'])))
        gte_b = ((ma_te & (yte_b != gt['maj_a']))
                 | (mb_te & (yte_b != gt['maj_b'])))
        print(f"\n== G3OUT {d} r{rule} ==  rule: {gt['rule']} "
              f"(maj A={gt['maj_a']}, B={gt['maj_b']})", flush=True)
        print(f"   conform tr/te = {gtr_a.sum()}/{gte_a.sum()}   "
              f"violators tr/te = {gtr_b.sum()}/{gte_b.sum()}", flush=True)

        per_b = [(g3_balanced(Xte[:, j], gte_a, gte_b),
                  heldout_acc(Xtr[:, j], Xte[:, j],
                              gtr_a, gtr_b, gte_a, gte_b),
                  g3_balanced(Xtr[:, j], gtr_a, gtr_b))
                 for j in range(Xte.shape[1])]
        jb = int(np.argmax([p[2] for p in per_b]))
        rows.append(dict(dataset=d, rule=rule, method='Base (best feature)',
                         n_own=Xte.shape[1], n_padded=0,
                         d_best=max(p[0] for p in per_b),
                         d_mean=float(np.mean([p[0] for p in per_b])),
                         recov=per_b[jb][1]))
        for nice, kind, tmpl in methods_for(rule):
            got = padded_cols(d, nice, kind, tmpl, pool)
            if got is None:
                print(f'  [miss] {nice}', flush=True)
                continue
            ctr, cte, n_own, n_pad = got
            per = [(g3_balanced(np.asarray(b, dtype=np.float64),
                                gte_a, gte_b),
                    heldout_acc(a, b, gtr_a, gtr_b, gte_a, gte_b),
                    g3_balanced(np.asarray(a, dtype=np.float64),
                                gtr_a, gtr_b))
                   for a, b in zip(ctr, cte)]
            jt = int(np.argmax([p[2] for p in per]))
            r = dict(dataset=d, rule=rule, method=nice,
                     n_own=n_own, n_padded=n_pad,
                     d_best=max(p[0] for p in per),
                     d_mean=float(np.mean([p[0] for p in per])),
                     recov=per[jt][1])
            rows.append(r)
            print(f"  {nice:17} n={n_own:2d}+{n_pad:2d}  "
                  f"u_best={r['d_best']:.3f} u_mean={r['d_mean']:.3f}  "
                  f"acc={r['recov']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f'{out_prefix}{out_suffix}.csv', index=False)
    avg = (df.groupby(['dataset', 'method'], sort=False)
             [['d_best', 'd_mean', 'recov']].mean().reset_index())
    avg['n_rules'] = df.groupby(['dataset', 'method'],
                                sort=False)['rule'].count().values
    avg.to_csv(f'{out_prefix}{out_suffix}_avg.csv', index=False)
    print('\n== G3OUT AVERAGE over valid rule pairs ==')
    for d in SUBGROUP_DATASETS:
        print(f'-- {d} --')
        for _, r in avg[avg.dataset == d].iterrows():
            print(f"  {r['method']:17} u_best={r['d_best']:.3f} "
                  f"u_mean={r['d_mean']:.3f} acc={r['recov']:.3f} "
                  f"({int(r['n_rules'])} rules)")
    return avg


def eval_normout(out_prefix, rules=5, tag='mi_gat_normout2',
                 out_suffix='_normout2', objective='d'):
    """Two-rule normal/outlier purpose (score_mode='normout'): rules A/B
    are discovered tree leaves, one predicting each class; normal =
    covered and correctly classified by either rule, outlier = covered
    but misclassified. Objective = Cohen's d (capped at 20). Held-out
    columns: d_best/d_mean on the test rows + Acc (balanced accuracy of
    the train-fitted midpoint threshold of the best-by-train-d feature).
    Averaged over valid pairs; roster excludes the generic MI/Pearson
    rows per the 2026-09-28 decision."""
    from autolearn.feat_selection.nfo.metrics import (normout_ground_truth,
                                                      cohens_d)
    from sklearn.metrics import balanced_accuracy_score

    D_CAP = 20.0

    def prep(env):
        X = env.dataset.instances.to_numpy(dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        med = np.nanmedian(X, axis=0)
        i = np.where(np.isnan(X))
        X[i] = np.take(med, i[1])
        return X

    def cdc(col, ma, mb):
        return float(min(cohens_d(np.asarray(col, dtype=np.float64),
                                  ma, mb), D_CAP))

    def bacc(ctr, cte, gtr_a, gtr_b, gte_a, gte_b):
        f_tr = np.asarray(ctr, dtype=np.float64)
        f_tr = np.where(np.isfinite(f_tr), f_tr,
                        np.nanmedian(f_tr[np.isfinite(f_tr)])
                        if np.isfinite(f_tr).any() else 0.0)
        ma, mb = f_tr[gtr_a].mean(), f_tr[gtr_b].mean()
        if not np.isfinite(ma - mb) or ma == mb:
            return 0.5
        thr = (ma + mb) / 2.0
        f_te = np.asarray(cte, dtype=np.float64)
        f_te = np.where(np.isfinite(f_te), f_te,
                        np.nanmedian(f_te[np.isfinite(f_te)])
                        if np.isfinite(f_te).any() else 0.0)
        sel = gte_a | gte_b
        if gte_a.sum() < 2 or gte_b.sum() < 2:
            return float('nan')
        y_true = gte_a[sel].astype(int)
        pred = ((f_te[sel] > thr) == (ma > mb)).astype(int)
        return float(balanced_accuracy_score(y_true, pred))

    if objective == 'g3':
        # normg3 variant: metric = class-balanced (1 - g3) threshold
        # separability on test; detection = train-fitted optimal (KS)
        # threshold applied to test. Rebinds the local metric functions,
        # so the evaluation body below is shared between objectives.
        from autolearn.feat_selection.nfo.metrics import g3_balanced

        def fit_thr(col, ma, mb):
            v = np.asarray(col, dtype=np.float64)
            a, b = v[ma], v[mb]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) < 2 or len(b) < 2:
                return None
            allv = np.concatenate([a, b])
            order = np.argsort(allv, kind='mergesort')
            grp = np.concatenate([np.zeros(len(a)), np.ones(len(b))])[order]
            ca = np.cumsum(grp == 0) / float(len(a))
            cb = np.cumsum(grp == 1) / float(len(b))
            vs = allv[order]
            ok = np.r_[vs[1:] != vs[:-1], True]
            diffs = (ca - cb)[ok]
            pos = np.where(ok)[0][int(np.argmax(np.abs(diffs)))]
            return float(vs[pos]), bool((ca - cb)[pos] > 0)

        def ks_acc(ctr, cte, gtr_a, gtr_b, gte_a, gte_b):
            ft = fit_thr(ctr, gtr_a, gtr_b)
            if ft is None or gte_a.sum() < 2 or gte_b.sum() < 2:
                return float('nan')
            thr, a_low = ft
            v = np.asarray(cte, dtype=np.float64)
            v = np.where(np.isfinite(v), v, thr)
            pred_a = (v <= thr) if a_low else (v > thr)
            return 0.5 * (float(pred_a[gte_a].mean())
                          + float((~pred_a)[gte_b].mean()))

        cdc = (lambda col, ma, mb:
               g3_balanced(np.asarray(col, dtype=np.float64), ma, mb))
        bacc = ks_acc

    def methods_for(rule):
        plugged = [('B-MAFE(normout)', 'expr',
                    f'nfo/RF/lsbo/{tag}_r{rule}' +
                    '/lam1.0/{d}/selected_topk_lam1.0.json')]
        drop = {'B-MAFE(subgroup)', 'B-MAFE(MI)', 'B-MAFE(Pearson)'}
        return plugged + [m for m in SUBGROUP_METHODS if m[0] not in drop]

    rows = []
    for d in SUBGROUP_DATASETS:
      env_tr, env_te = get_env(d + '__tr'), get_env(d + '__te')
      Xtr, Xte = prep(env_tr), prep(env_te)
      ytr = np.asarray(env_tr.dataset.labels).astype(int)
      yte = np.asarray(env_te.dataset.labels).astype(int)
      if len(np.unique(ytr)) > 2:  # mirror normout_ground_truth binarization
          yte_b = (yte > np.median(ytr)).astype(int)
      else:
          yte_b = yte
      pool = random_pool(d, top_k=TOP_K)
      for rule in range(rules):
        try:
            gt = normout_ground_truth(Xtr, ytr, rule_rank=rule)
        except ValueError as e:
            print(f'[{d}] pair r{rule} unavailable ({e})', flush=True)
            break
        gtr_a, gtr_b = gt['mask_a'], gt['mask_b']
        leaf_te = gt['tree'].apply(Xte)
        in_a = leaf_te == gt['leaf_a']
        in_b = leaf_te == gt['leaf_b']
        gte_a = ((in_a & (yte_b == gt['maj_a']))
                 | (in_b & (yte_b == gt['maj_b'])))
        gte_b = ((in_a & (yte_b != gt['maj_a']))
                 | (in_b & (yte_b != gt['maj_b'])))
        print(f"\n== NORMOUT {d} r{rule} ==  A: {gt['rule_a']}  |  "
              f"B: {gt['rule_b']}", flush=True)
        print(f"   normal tr/te = {gtr_a.sum()}/{gte_a.sum()}   "
              f"outliers tr/te = {gtr_b.sum()}/{gte_b.sum()}", flush=True)

        per_b = [(cdc(Xte[:, j], gte_a, gte_b),
                  bacc(Xtr[:, j], Xte[:, j], gtr_a, gtr_b, gte_a, gte_b),
                  cdc(Xtr[:, j], gtr_a, gtr_b))
                 for j in range(Xte.shape[1])]
        jb = int(np.argmax([p[2] for p in per_b]))
        rows.append(dict(dataset=d, rule=rule, method='Base (best feature)',
                         n_own=Xte.shape[1], n_padded=0,
                         d_best=max(p[0] for p in per_b),
                         d_mean=float(np.mean([p[0] for p in per_b])),
                         recov=per_b[jb][1]))
        for nice, kind, tmpl in methods_for(rule):
            got = padded_cols(d, nice, kind, tmpl, pool)
            if got is None:
                print(f'  [miss] {nice}', flush=True)
                continue
            ctr, cte, n_own, n_pad = got
            per = [(cdc(b, gte_a, gte_b),
                    bacc(a, b, gtr_a, gtr_b, gte_a, gte_b),
                    cdc(a, gtr_a, gtr_b))
                   for a, b in zip(ctr, cte)]
            jt = int(np.argmax([p[2] for p in per]))
            r = dict(dataset=d, rule=rule, method=nice,
                     n_own=n_own, n_padded=n_pad,
                     d_best=max(p[0] for p in per),
                     d_mean=float(np.mean([p[0] for p in per])),
                     recov=per[jt][1])
            rows.append(r)
            print(f"  {nice:17} n={n_own:2d}+{n_pad:2d}  "
                  f"d_best={r['d_best']:.2f} d_mean={r['d_mean']:.2f}  "
                  f"acc={r['recov']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f'{out_prefix}{out_suffix}.csv', index=False)
    avg = (df.groupby(['dataset', 'method'], sort=False)
             [['d_best', 'd_mean', 'recov']].mean().reset_index())
    avg['n_rules'] = df.groupby(['dataset', 'method'],
                                sort=False)['rule'].count().values
    avg.to_csv(f'{out_prefix}{out_suffix}_avg.csv', index=False)
    print('\n== NORMOUT AVERAGE over valid pairs ==')
    for d in SUBGROUP_DATASETS:
        print(f'-- {d} --')
        for _, r in avg[avg.dataset == d].iterrows():
            print(f"  {r['method']:17} d_best={r['d_best']:.2f} "
                  f"d_mean={r['d_mean']:.2f} acc={r['recov']:.3f} "
                  f"({int(r['n_rules'])} pairs)")
    return avg


def eval_ifout(out_prefix, alphas=(0.05, 0.10, 0.15)):
    """Detector-referenced outlier purpose (score_mode='ifout',
    LABEL-FREE): frozen train-side IsolationForest verdict (bottom-10%
    scores = outliers); the purpose is one readable expression separating
    inliers from flagged outliers = an explainable surrogate anomaly
    score. Held-out: the frozen detector+threshold are applied to the
    test rows; d_best/d_mean (capped at 20) + Acc (train-fitted midpoint
    threshold, balanced). Roster excludes MI/Pearson (2026-09-28
    decision)."""
    from autolearn.feat_selection.nfo.metrics import (ifout_ground_truth,
                                                      cohens_d)
    from sklearn.metrics import balanced_accuracy_score

    D_CAP = 20.0

    def prep(env):
        X = env.dataset.instances.to_numpy(dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        med = np.nanmedian(X, axis=0)
        i = np.where(np.isnan(X))
        X[i] = np.take(med, i[1])
        return X

    def cdc(col, ma, mb):
        return float(min(cohens_d(np.asarray(col, dtype=np.float64),
                                  ma, mb), D_CAP))

    def bacc(ctr, cte, gtr_a, gtr_b, gte_a, gte_b):
        f_tr = np.asarray(ctr, dtype=np.float64)
        f_tr = np.where(np.isfinite(f_tr), f_tr,
                        np.nanmedian(f_tr[np.isfinite(f_tr)])
                        if np.isfinite(f_tr).any() else 0.0)
        ma, mb = f_tr[gtr_a].mean(), f_tr[gtr_b].mean()
        if not np.isfinite(ma - mb) or ma == mb:
            return 0.5
        thr = (ma + mb) / 2.0
        f_te = np.asarray(cte, dtype=np.float64)
        f_te = np.where(np.isfinite(f_te), f_te,
                        np.nanmedian(f_te[np.isfinite(f_te)])
                        if np.isfinite(f_te).any() else 0.0)
        sel = gte_a | gte_b
        if gte_a.sum() < 2 or gte_b.sum() < 2:
            return float('nan')
        y_true = gte_a[sel].astype(int)
        pred = ((f_te[sel] > thr) == (ma > mb)).astype(int)
        return float(balanced_accuracy_score(y_true, pred))

    def methods_for(alpha):
        tag = {0.05: 'mi_gat_ifout_a05', 0.10: 'mi_gat_ifout',
               0.15: 'mi_gat_ifout_a15'}[round(alpha, 2)]
        drop = {'B-MAFE(subgroup)', 'B-MAFE(MI)', 'B-MAFE(Pearson)'}
        return ([('B-MAFE(ifout)', 'expr',
                  'nfo/RF/lsbo/' + tag +
                  '/lam1.0/{d}/selected_topk_lam1.0.json')]
                + [m for m in SUBGROUP_METHODS if m[0] not in drop])

    rows = []
    for d in SUBGROUP_DATASETS:
      env_tr, env_te = get_env(d + '__tr'), get_env(d + '__te')
      Xtr, Xte = prep(env_tr), prep(env_te)
      pool = random_pool(d, top_k=TOP_K)
      for alpha in alphas:
        methods = methods_for(alpha)
        gt = ifout_ground_truth(Xtr, contamination=alpha)
        gtr_a, gtr_b = gt['mask_a'], gt['mask_b']
        s_te = gt['detector'].score_samples(Xte)
        gte_b = s_te <= gt['thr']
        gte_a = ~gte_b
        print(f"\n== IFOUT {d} alpha={alpha:.0%} ==  frozen IForest "
              f"verdict", flush=True)
        print(f"   inliers tr/te = {gtr_a.sum()}/{gte_a.sum()}   "
              f"outliers tr/te = {gtr_b.sum()}/{gte_b.sum()}", flush=True)

        per_b = [(cdc(Xte[:, j], gte_a, gte_b),
                  bacc(Xtr[:, j], Xte[:, j], gtr_a, gtr_b, gte_a, gte_b),
                  cdc(Xtr[:, j], gtr_a, gtr_b))
                 for j in range(Xte.shape[1])]
        jb = int(np.argmax([p[2] for p in per_b]))
        rows.append(dict(dataset=d, rule=alpha, method='Base (best feature)',
                         n_own=Xte.shape[1], n_padded=0,
                         d_best=max(p[0] for p in per_b),
                         d_mean=float(np.mean([p[0] for p in per_b])),
                         recov=per_b[jb][1]))
        for nice, kind, tmpl in methods:
            got = padded_cols(d, nice, kind, tmpl, pool)
            if got is None:
                print(f'  [miss] {nice}', flush=True)
                continue
            ctr, cte, n_own, n_pad = got
            per = [(cdc(b, gte_a, gte_b),
                    bacc(a, b, gtr_a, gtr_b, gte_a, gte_b),
                    cdc(a, gtr_a, gtr_b))
                   for a, b in zip(ctr, cte)]
            jt = int(np.argmax([p[2] for p in per]))
            r = dict(dataset=d, rule=alpha, method=nice,
                     n_own=n_own, n_padded=n_pad,
                     d_best=max(p[0] for p in per),
                     d_mean=float(np.mean([p[0] for p in per])),
                     recov=per[jt][1])
            rows.append(r)
            print(f"  {nice:17} n={n_own:2d}+{n_pad:2d}  "
                  f"d_best={r['d_best']:.2f} d_mean={r['d_mean']:.2f}  "
                  f"acc={r['recov']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f'{out_prefix}_ifout.csv', index=False)
    avg = (df.groupby(['dataset', 'method'], sort=False)
             [['d_best', 'd_mean', 'recov']].mean().reset_index())
    avg['n_alphas'] = df.groupby(['dataset', 'method'],
                                 sort=False)['rule'].count().values
    avg.to_csv(f'{out_prefix}_ifout_avg.csv', index=False)
    print(f'\n== IFOUT AVERAGE over alphas {tuple(alphas)} ==')
    for d in SUBGROUP_DATASETS:
        print(f'-- {d} --')
        for _, r in avg[avg.dataset == d].iterrows():
            print(f"  {r['method']:17} d_best={r['d_best']:.2f} "
                  f"d_mean={r['d_mean']:.2f} acc={r['recov']:.3f}")
    return avg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--part', default='all',
                   choices=['split', 'anom', 'subgroup', 'subgroup_op',
                            'subgroup_op3', 'subgroup_op5', 'rule_out',
                            'g3out', 'normout', 'normg3', 'ifout',
                            'surr', 'all'])
    p.add_argument('--out_prefix', default='outputs_fair/purpose')
    args = p.parse_args()
    if args.part in ('split', 'all'):
        eval_split(args.out_prefix)
    if args.part in ('anom', 'all'):
        eval_anom(args.out_prefix)
    if args.part in ('subgroup', 'all'):
        eval_subgroup(args.out_prefix)
    if args.part == 'subgroup_op':
        eval_subgroup(args.out_prefix, variant='opposite')
    if args.part == 'subgroup_op3':
        eval_subgroup(args.out_prefix, variant='opposite', rules=3)
    if args.part == 'subgroup_op5':
        eval_subgroup(args.out_prefix, variant='opposite', rules=5)
    if args.part == 'rule_out':
        eval_rule_out(args.out_prefix)
    if args.part == 'g3out':
        eval_g3out(args.out_prefix)
    if args.part == 'normout':
        eval_normout(args.out_prefix)
    if args.part == 'normg3':
        eval_normout(args.out_prefix, rules=5, tag='mi_gat_normg3',
                     out_suffix='_normg3', objective='g3')
    if args.part == 'ifout':
        eval_ifout(args.out_prefix)
    if args.part in ('surr', 'all'):
        surrogate_report()


if __name__ == '__main__':
    main()
