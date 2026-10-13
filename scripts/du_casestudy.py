"""Case study: the lambda knob yields a readable simplicity<->usefulness spectrum.

For an interpretable dataset, sweep lambda in {0,.25,.5,.75,1} and show the top
selected B-MAFE features as human-readable expressions (real feature names) with
their K and MI. Then validate one interpretable-and-useful feature as a held-out
threshold rule (fixes the paper's 12-row Example 1 fragility).
"""
from __future__ import annotations
import argparse
import numpy as np
import sys
sys.path.insert(0, '.')
from scripts.du_common import get_env, construct_values, feat_K, feat_MI, to_infix, load_pool
from autolearn.feat_selection.nfo.metrics import composite_from_pool

FAIR = 'outputs_fair'
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]
NAMES = {
    'pima': {0: 'Pregnancies', 1: 'Glucose', 2: 'BloodPressure', 3: 'SkinThickness',
             4: 'Insulin', 5: 'BMI', 6: 'DPF', 7: 'Age'},
    'airfoil': {0: 'freq', 1: 'angle', 2: 'chord', 3: 'velocity', 4: 'disp_thick'},
    'housing': {0: 'CRIM', 1: 'ZN', 2: 'INDUS', 3: 'CHAS', 4: 'NOX', 5: 'RM',
                6: 'AGE', 7: 'DIS', 8: 'RAD', 9: 'TAX', 10: 'PTRATIO', 11: 'B', 12: 'LSTAT'},
}


def sweep(dataset, topk=3):
    env = get_env(dataset + '__tr'); names = NAMES[dataset]
    pool, _ = load_pool(FAIR, dataset + '__tr')
    feats = list(pool.keys())
    MI = np.array([pool[f] for f in feats], float)
    K = np.array([feat_K(env, f) for f in feats], float)
    print(f"\n===== {dataset} (pool {len(feats)}) : lambda-swept top-{topk} =====")
    print(f"  {'lambda':>7}  top features  (K bits, MI)")
    for lam in LAMS:
        comp = composite_from_pool(MI, K, lam)
        order = [i for i in np.argsort(comp)[::-1] if np.isfinite(K[i])][:topk]
        items = '; '.join(f"{to_infix(feats[i],names)} [K={K[i]:.1f}, MI={MI[i]:.3f}]" for i in order)
        print(f"  {lam:>7}  {items}")


def threshold_rule(dataset, postfix, label):
    env_tr, env_te = get_env(dataset + '__tr'), get_env(dataset + '__te')
    ytr = np.asarray(env_tr.dataset.labels); yte = np.asarray(env_te.dataset.labels)
    vtr = construct_values(env_tr, postfix); vte = construct_values(env_te, postfix)
    vtr = np.where(np.isfinite(vtr), vtr, np.nanmedian(vtr[np.isfinite(vtr)]))
    vte = np.where(np.isfinite(vte), vte, np.nanmedian(vtr[np.isfinite(vtr)]))
    best = (None, -1, 1)
    for thr in np.quantile(vtr, np.linspace(0.05, 0.95, 91)):
        for s in (1, -1):
            pred = ((vtr > thr) if s > 0 else (vtr <= thr)).astype(int)
            acc = (pred == ytr).mean()
            if acc > best[1]:
                best = (thr, acc, s)
    thr, tracc, s = best
    pr = ((vte > thr) if s > 0 else (vte <= thr)).astype(int)
    teacc = (pr == yte).mean()
    base = max(np.bincount(yte)) / len(yte)
    op = '>' if s > 0 else '<='
    print(f"\n  RULE: [{label}] {op} {thr:.3g}  =>  train {tracc:.3f} | HELD-OUT TEST {teacc:.3f} "
          f"(base rate {base:.3f}, lift {teacc-base:+.3f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['pima', 'airfoil', 'housing'])
    args = p.parse_args()
    for d in args.datasets:
        sweep(d)
    # Pima threshold rules: simple base vs an engineered compound
    print("\n===== held-out threshold rules (Pima) =====")
    threshold_rule('pima', '1', 'Glucose')                 # simplest
    threshold_rule('pima', '1,5,*', 'Glucose x BMI')       # engineered compound
    threshold_rule('pima', '1,5,+', 'Glucose + BMI')


if __name__ == '__main__':
    main()
