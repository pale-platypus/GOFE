"""E-AFE baseline entry point.

Usage:
    python -m autolearn.feat_selection.eafe_baseline \\
        --data pima --eval_model RF \\
        --fpe_model models/fpe_classifier.pt \\
        --feat_pool outputs/nfo/RF/eafe/iter/pima

Writes one row to <feat_pool>/results.csv in the same iter,score,top_k,method
shape as run_iter.sh's output so run_compare.sh-style scripts ingest it.

If no FPE checkpoint exists, the run falls back to "Stage 2 only" mode which
is equivalent to NFS in PyTorch (slower, but still produces a meaningful
baseline number).
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.extend(["."])

from autolearn.data import Dataset
from autolearn.utils import log
from autolearn.feat_selection.eafe.fpe_model import load_fpe
from autolearn.feat_selection.eafe.two_stage_train import EAFEConfig, two_stage_train


def _columns_as_numpy(df):
    cols = []
    for c in df.columns:
        v = df[c].to_numpy()
        try:
            v = v.astype(np.float64)
        except (TypeError, ValueError):
            # categorical/string column — label-encode to ints
            uniques = {val: i for i, val in enumerate(pd.unique(v))}
            v = np.array([uniques[val] for val in v], dtype=np.float64)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(v)
    return cols


def run_eafe(args):
    dataset = Dataset(args.data)
    X_df = dataset.instances.copy()
    X_df.columns = X_df.columns.astype(str)
    y = dataset.labels
    task = dataset.task

    log(f"[E-AFE] dataset={args.data} task={task} n={X_df.shape[0]} d={X_df.shape[1]}")

    original_cols = _columns_as_numpy(X_df)

    # Load FPE if available; warn (not fail) if missing
    fpe = None
    fpe_meta = None
    if args.fpe_model and os.path.exists(args.fpe_model):
        try:
            device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
            fpe, fpe_meta = load_fpe(args.fpe_model, device=device)
            log(f"[E-AFE] loaded FPE from {args.fpe_model}  metadata={fpe_meta}")
        except Exception as e:
            log(f"[E-AFE] failed to load FPE at {args.fpe_model}: {e}")
            log(f"[E-AFE] continuing in Stage-2-only mode (equivalent to NFS-PyTorch)")
            fpe = None
    elif args.fpe_model:
        log(f"[E-AFE] FPE model not found at {args.fpe_model} — Stage-2-only mode")

    cfg = EAFEConfig(
        max_order=args.max_order,
        batch_size=args.batch_size,
        stage1_epochs=args.stage1_epochs if fpe is not None else 0,
        stage2_epochs=args.stage2_epochs,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        gamma=args.gamma,
        lambd=args.lambd,
        fpe_filter_thresh=args.fpe_filter_thresh if fpe is not None else 0.0,
        cv=args.cv,
        n_estimators=args.n_estimators,
        seed=args.seed,
        device='cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu',
    )

    t0 = time.time()
    best_score, best_epoch, info = two_stage_train(
        original_cols=original_cols,
        y=np.asarray(y),
        task=task,
        fpe_predictor=fpe,
        config=cfg,
        scoring=args.c_scoring,
        log_fn=log,
    )
    total_time = time.time() - t0

    log(f"[E-AFE] best score {best_score:.6f} at stage2 epoch {best_epoch} "
        f"(baseline {info['baseline_score']:.6f}, "
        f"delta {best_score - info['baseline_score']:+.6f}, "
        f"total_time {total_time:.1f}s)")

    if args.feat_pool:
        os.makedirs(args.feat_pool, exist_ok=True)
        results_path = os.path.join(args.feat_pool, 'results.csv')
        if not os.path.exists(results_path):
            with open(results_path, 'w') as f:
                f.write("iter,score,top_k,method\n")
        # top_k for E-AFE is the number of original features (each gets one
        # transformed output column) — kept for schema parity with other baselines.
        with open(results_path, 'a') as f:
            f.write(f"0,{best_score},{X_df.shape[1]},eafe\n")
        log(f"[E-AFE] appended result to {results_path}")

        # Always write eval_manifest.json — even when E-AFE finds no
        # useful columns. Otherwise run_quality_compare.sh's skip-if-cached
        # check (which looks for eval_manifest.json) keeps re-running E-AFE
        # every sweep on datasets where the RL policy converges to "produce
        # nothing better than baseline".
        try:
            best_cols = info.get('best_columns') or []
            actions_arr = info.get('best_actions')
            n_feats_dataset = int(X_df.shape[1])
            if best_cols:
                derived = pd.DataFrame({
                    f"new_{i}": np.asarray(c, dtype=np.float64).reshape(-1)
                    for i, c in enumerate(best_cols)
                })
                formulas = []
                feature_strs = []
                if actions_arr is not None:
                    formulas = [f"action_seq_{i}={list(a)}" for i, a in enumerate(actions_arr)]
                    # Translate the E-AFE action sequence for each output column
                    # into a DIFER post-order feature string so eval_quality
                    # can build trees and compute Kext / DBF the same way it
                    # does for DIFER / B-MAFE features.
                    feature_strs = [
                        _action_seq_to_postorder(actions_arr[i], i, n_feats_dataset)
                        for i in range(min(len(actions_arr), len(best_cols)))
                    ]
            else:
                derived = pd.DataFrame()
                formulas = []
                feature_strs = []
                log("[E-AFE] no new columns beat baseline; writing empty manifest "
                    "so the sweep records 'ran, found nothing' instead of retrying")
            _write_baseline_manifest(
                args.feat_pool, dataset=args.data, method='eafe',
                derived=derived,
                formulas=formulas[:derived.shape[1]],
                feature_strs=feature_strs[:derived.shape[1]] if feature_strs else None,
            )
        except Exception as e:
            log(f"[E-AFE] failed to write eval_manifest: {type(e).__name__}: {e}")

    return best_score


def _action_seq_to_postorder(action_seq, agent_idx, num_features):
    """Translate an E-AFE action sequence into a DIFER post-order feature string.

    E-AFE encodes per-feature transformations as integer action IDs against an
    action space defined in ``eafe/operations.py`` (4 unary + 5 binary ops).
    eval_quality.py only knows how to compute Kext from DIFER-style post-order
    strings (e.g. "0,5,mod_column,log"). This walker decodes each action,
    accumulates the current expression as a post-order token list, and emits a
    comma-joined string that env.generate_tree_from_str can parse.

    The seed leaf is the agent's own base feature (index ``agent_idx``). Each
    non-noop action wraps the running expression:
      unary:  cur = cur + "," + <unary_op>
      binary: cur = cur + "," + <other_feat> + "," + <binary_op>

    Returns the post-order string. For an all-noop sequence (or empty), just
    returns the bare seed feature index (which is a valid one-leaf tree).
    """
    from autolearn.feat_selection.eafe.operations import decode_action
    # E-AFE op name -> DIFER op name (both must exist in env.ops or
    # generate_tree_from_str returns an invalid tree). All five binary + four
    # unary ops have direct DIFER counterparts in search_space.num_op_info.
    UNARY_DIFER = ['log', 'sqrt_abs', 'reciprocal', 'min_max']
    BINARY_DIFER = ['+', '-', '*', 'division', 'mod_column']
    tokens = [str(int(agent_idx))]
    for action in action_seq:
        decoded = decode_action(int(action), agent_idx, num_features)
        kind = decoded[0]
        if kind == 'noop':
            continue
        if kind == 'unary':
            tokens.append(UNARY_DIFER[decoded[1]])
        else:  # binary
            other_feat = decoded[2]
            tokens.append(str(int(other_feat)))
            tokens.append(BINARY_DIFER[decoded[1]])
    return ','.join(tokens)


def _write_baseline_manifest(out_dir, *, dataset, method, derived, formulas,
                              feature_strs=None):
    """Writer used by all baselines. ``feature_strs`` (optional) carries DIFER
    post-order tree strings so eval_quality can compute Kext / DBF on the
    symbolic tree (used by E-AFE; other baselines pass formulas only).
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
    if feature_strs:
        manifest['feature_strs'] = feature_strs
    manifest_path = os.path.join(out_dir, 'eval_manifest.json')
    with open(manifest_path, 'w') as f:
        _json.dump(manifest, f, indent=2)
    log(f"[{method}] wrote eval_manifest to {manifest_path} "
        f"(n_features={n_features})")


def parse_args():
    p = argparse.ArgumentParser(description='E-AFE baseline (PyTorch reimplementation)')
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--eval_model', type=str, default='RF',
                   help='kept for interface symmetry; only RF is used here')
    p.add_argument('--c_scoring', type=str, default='f1_micro')
    p.add_argument('--cv', type=int, default=5)
    p.add_argument('--n_estimators', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--use_cuda', action='store_true')

    p.add_argument('--fpe_model', type=str, default='models/fpe_classifier.pt',
                   help='path to pre-trained FPE classifier; if missing, Stage 1 is skipped')
    p.add_argument('--stage1_epochs', type=int, default=30)
    p.add_argument('--stage2_epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--max_order', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--entropy_coef', type=float, default=1e-3)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lambd', type=float, default=0.4)
    p.add_argument('--fpe_filter_thresh', type=float, default=0.5)

    p.add_argument('--feat_pool', type=str, default=None,
                   help='directory to write results.csv; if None, only logs')
    return p.parse_args()


if __name__ == '__main__':
    run_eafe(parse_args())
