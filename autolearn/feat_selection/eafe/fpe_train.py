"""FPE training pipeline.

Labeling protocol (from E-AFE paper, Section III-B):
  For each training dataset D^i with m^i features:
    A_0 = cv-score on all features
    For each feature j:
      A_j = cv-score after removing feature j
      L_j = 1 if (A_0 - A_j) > thre else 0
    Each feature also produces a CCWS signature H_j (d floats)
  Train an MLP on {(H_j, L_j)} across all i, j.

Two training modes:
  --mode bootstrap  : use the 25 local datasets in data/ as the training corpus.
                       Fast, runs in <1 hour. Use this for a smoke-test FPE.
  --mode openml     : download N public datasets from OpenML and train on those.
                       Faithful to the paper but requires ~hours of cv runs.

Usage (bootstrap):
  python -m autolearn.feat_selection.eafe.fpe_train --mode bootstrap \
      --output models/fpe_classifier.pt --signature_dim 48

Usage (OpenML, optional):
  python -m autolearn.feat_selection.eafe.fpe_train --mode openml \
      --n_datasets 50 --output models/fpe_classifier.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.extend(["."])

from autolearn.data.dataset import Dataset
from autolearn.feat_selection.eafe.fpe_model import FPEClassifier, save_fpe
from autolearn.feat_selection.eafe.minhash import column_signature


def _project_root():
    return Path(__file__).resolve().parents[3]


def _list_local_datasets():
    data_dir = _project_root() / "data"
    csvs = sorted(data_dir.glob("*.csv"))
    return [p.stem for p in csvs]


def _safe_cv_score(X, y, task, cv=3, n_estimators=10):
    """Quick cross-val score. Lower n_estimators / cv than the eval protocol —
    we're doing it once per feature, so speed matters."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer

    X = X.copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(-1, inplace=True)
    X.columns = X.columns.astype(str)

    if task == 'regression':
        from autolearn.metrics import r2_score as r2
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=0, n_jobs=1)
        try:
            return float(cross_val_score(model, X, y, scoring=make_scorer(r2),
                                         cv=cv, n_jobs=1).mean())
        except Exception:
            return float('nan')
    else:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=0, n_jobs=1)
        try:
            return float(cross_val_score(model, X, y, scoring='f1_micro',
                                         cv=cv, n_jobs=1).mean())
        except Exception:
            return float('nan')


def build_training_set(dataset_names, signature_dim=48, thre=0.01,
                       cv=3, n_estimators=10, verbose=True):
    """Run the leave-one-feature-out labeling protocol over a list of datasets.

    Returns (X, y) where X is (total_features, signature_dim) and y is (total_features,) in {0, 1}.
    """
    all_sigs = []
    all_labels = []

    for ds_idx, name in enumerate(dataset_names):
        try:
            ds = Dataset(name)
        except Exception as e:
            if verbose:
                print(f"[FPE-train] skip {name}: {e}", flush=True)
            continue
        X_df = ds.instances.copy()
        X_df.columns = X_df.columns.astype(str)
        y = ds.labels
        task = ds.task

        t0 = time.time()
        A0 = _safe_cv_score(X_df, y, task, cv=cv, n_estimators=n_estimators)
        if not np.isfinite(A0):
            if verbose:
                print(f"[FPE-train] skip {name}: baseline cv failed", flush=True)
            continue

        n_pos, n_neg = 0, 0
        for col in X_df.columns:
            col_values = X_df[col].to_numpy()
            try:
                col_numeric = col_values.astype(np.float64)
            except (TypeError, ValueError):
                continue  # skip non-numeric

            X_drop = X_df.drop(columns=[col])
            if X_drop.shape[1] == 0:
                continue  # single-feature dataset; can't drop

            Aj = _safe_cv_score(X_drop, y, task, cv=cv, n_estimators=n_estimators)
            if not np.isfinite(Aj):
                continue

            label = 1 if (A0 - Aj) > thre else 0
            sig = column_signature(col_numeric, d=signature_dim, seed=0)
            all_sigs.append(sig)
            all_labels.append(label)
            n_pos += int(label == 1)
            n_neg += int(label == 0)

        elapsed = time.time() - t0
        if verbose:
            print(f"[FPE-train] {ds_idx + 1}/{len(dataset_names)} {name:24s} "
                  f"task={task} n_feats={X_df.shape[1]} "
                  f"A0={A0:.4f} pos={n_pos} neg={n_neg} t={elapsed:.1f}s",
                  flush=True)

    if not all_sigs:
        raise RuntimeError("No FPE training data was generated. Check dataset availability.")
    X = np.stack(all_sigs).astype(np.float32)
    y = np.asarray(all_labels, dtype=np.float32)
    if verbose:
        print(f"[FPE-train] total examples: {len(y)} (pos={int(y.sum())}, neg={int((1 - y).sum())})",
              flush=True)
    return X, y


def train_classifier(X, y, signature_dim=48, hidden=128, epochs=200,
                     batch_size=128, lr=1e-3, weight_decay=1e-4, val_frac=0.2,
                     seed=0, device='cpu'):
    torch.manual_seed(seed)
    np.random.seed(seed)

    idx = np.arange(len(y))
    np.random.shuffle(idx)
    split = int(len(idx) * (1 - val_frac))
    train_idx, val_idx = idx[:split], idx[split:]
    Xt = torch.from_numpy(X[train_idx]).to(device)
    yt = torch.from_numpy(y[train_idx]).to(device)
    Xv = torch.from_numpy(X[val_idx]).to(device)
    yv = torch.from_numpy(y[val_idx]).to(device)

    model = FPEClassifier(signature_dim=signature_dim, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Reweight positives for class imbalance
    n_pos = max(1.0, float(yt.sum().item()))
    n_neg = max(1.0, float((1 - yt).sum().item()))
    pos_weight = torch.tensor(n_neg / n_pos, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = float('inf')
    best_state = None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(yt), device=device)
        total = 0.0
        for i in range(0, len(yt), batch_size):
            sel = perm[i: i + batch_size]
            logits = model(Xt[sel])
            loss = loss_fn(logits, yt[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(sel)

        model.eval()
        with torch.no_grad():
            vlogits = model(Xv)
            vloss = loss_fn(vlogits, yv).item()
            vpred = (torch.sigmoid(vlogits) > 0.5).float()
            vacc = (vpred == yv).float().mean().item()
            vrec = (vpred[yv == 1] == 1).float().mean().item() if (yv == 1).any() else 0.0

        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"[FPE-train] epoch {epoch + 1}/{epochs} "
                  f"train_loss={total / len(yt):.4f} "
                  f"val_loss={vloss:.4f} val_acc={vacc:.4f} val_recall={vrec:.4f}",
                  flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Train the E-AFE FPE classifier')
    parser.add_argument('--mode', choices=['bootstrap', 'openml'], default='bootstrap')
    parser.add_argument('--output', type=str, default='models/fpe_classifier.pt')
    parser.add_argument('--signature_dim', type=int, default=48)
    parser.add_argument('--thre', type=float, default=0.01)
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--n_estimators', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_datasets', type=int, default=50,
                        help='[openml mode] how many OpenML datasets to fetch')
    parser.add_argument('--exclude', type=str, nargs='*', default=[],
                        help='dataset names to exclude (e.g. the eval-set datasets)')
    return parser.parse_args()


def _openml_dataset_names(n):
    try:
        import openml
    except ImportError as e:
        raise ImportError(
            "openml mode requires `pip install openml`. "
            "Or use --mode bootstrap to train on local datasets."
        ) from e
    # Curated list of small tabular benchmarks — kept short for speed
    suite = openml.study.get_suite('OpenML-CC18')
    task_ids = suite.tasks[:n]
    out = []
    for tid in task_ids:
        try:
            task = openml.tasks.get_task(tid, download_data=False)
            ds = task.get_dataset()
            out.append(ds.name)
        except Exception:
            continue
    return out


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[FPE-train] device={device} mode={args.mode}", flush=True)

    if args.mode == 'bootstrap':
        names = [n for n in _list_local_datasets() if n not in set(args.exclude)]
        print(f"[FPE-train] bootstrap from {len(names)} local datasets: {names}", flush=True)
    else:
        names = _openml_dataset_names(args.n_datasets)
        print(f"[FPE-train] openml mode: {len(names)} datasets", flush=True)

    X, y = build_training_set(
        names, signature_dim=args.signature_dim, thre=args.thre,
        cv=args.cv, n_estimators=args.n_estimators,
    )
    model = train_classifier(
        X, y,
        signature_dim=args.signature_dim, hidden=args.hidden,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, seed=args.seed, device=device,
    )

    metadata = {
        'mode': args.mode,
        'n_datasets_used': len(names),
        'n_examples': int(len(y)),
        'pos_ratio': float(y.mean()),
        'signature_dim': args.signature_dim,
        'hidden': args.hidden,
        'thre': args.thre,
    }
    save_fpe(model, args.output, metadata=metadata)
    print(f"[FPE-train] saved {args.output}  metadata={metadata}", flush=True)


if __name__ == '__main__':
    main()
