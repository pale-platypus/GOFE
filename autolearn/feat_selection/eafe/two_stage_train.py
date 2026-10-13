"""Two-stage E-AFE policy training.

Stage 1 (Quick Initialization with FPE Model):
    - REINFORCE the controller using FPE-only reward (Eq. 8 of paper).
    - Promising features (FPE prob > 0.5) get pushed into a replay buffer.
    - No real cv evaluation in this stage; should be very fast.

Stage 2 (Formal Training):
    - REINFORCE on real cv-score reward.
    - FPE filters candidates: features with FPE prob below ``fpe_filter_thresh``
      are dropped before cv evaluation (this is the "reduce candidate features"
      step from Section III-B).
    - Best score across all evaluated feature sets is tracked.

If no FPE model is available, ``stage1_epochs=0`` and ``fpe_filter_thresh=0``
recover vanilla NFS in PyTorch.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from autolearn.feat_selection.eafe import operations
from autolearn.feat_selection.eafe.controller import NFSController
from autolearn.feat_selection.eafe.fpe_model import FPEPredictor


_EPS = 1e-9


@dataclass
class EAFEConfig:
    max_order: int = 5
    batch_size: int = 32
    stage1_epochs: int = 30
    stage2_epochs: int = 100
    lr: float = 1e-2
    entropy_coef: float = 1e-3
    gamma: float = 0.99
    lambd: float = 0.4
    replay_buffer_capacity: int = 256
    fpe_filter_thresh: float = 0.5
    fpe_train_thre: float = 0.01  # only used to scale Stage 1 reward (Eq. 8)
    cv: int = 5
    n_estimators: int = 10
    seed: int = 0
    device: str = 'cpu'


# ----- feature column construction -----

def build_transformed_columns(actions_np, original_cols):
    """Apply an (N, T) action matrix and return a (N,) list of resulting columns.

    actions_np: numpy int array shape (num_features, max_order)
    original_cols: list of 1D numpy arrays for the N original features
    """
    N = len(original_cols)
    T = actions_np.shape[1]
    result = []
    for agent_idx in range(N):
        col = original_cols[agent_idx].astype(np.float64, copy=True)
        for t in range(T):
            col = operations.apply_action(
                int(actions_np[agent_idx, t]), agent_idx, col, original_cols
            )
        result.append(col)
    return result


# ----- cv evaluation -----

def _cv_score_on_columns(columns, y, task, cv=5, n_estimators=10, scoring='f1_micro'):
    """Cross-val on an augmented feature matrix: original cols + provided cols."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer

    X = pd.DataFrame(np.stack(columns, axis=1))
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(-1, inplace=True)
    X.columns = X.columns.astype(str)

    if task == 'regression':
        from autolearn.metrics import r2_score as r2
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=0, n_jobs=1)
        return float(cross_val_score(model, X, y, scoring=make_scorer(r2),
                                     cv=cv, n_jobs=1).mean())
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=0, n_jobs=1)
    return float(cross_val_score(model, X, y, scoring=scoring, cv=cv, n_jobs=1).mean())


# ----- Stage 1 reward (paper Eq. 8) -----

def _stage1_reward_per_agent(fpe_probs, A_orig, delta_A_max, delta_A_min, thre):
    """Given FPE probabilities for the N generated features, return per-agent
    reward (the score for that feature subset as constructed in paper Eq. 8).

    fpe_probs: shape (N,) in [0, 1]
    Returns shape (N,) floats.
    """
    p = np.asarray(fpe_probs, dtype=np.float64)
    # paper eq. 8: piecewise-linear in p around 0.5
    # branch A: p in [0, 0.5)
    branch_a = A_orig + (0.5 - p) / 0.5 * (delta_A_max - thre)
    # branch B: p in [0.5, 1]
    branch_b = A_orig + (0.5 - p) / 0.5 * (thre - delta_A_min)
    return np.where(p < 0.5, branch_a, branch_b)


# ----- main training entry -----

def two_stage_train(
    original_cols: List[np.ndarray],
    y: np.ndarray,
    task: str,
    fpe_predictor: Optional[FPEPredictor],
    config: EAFEConfig,
    scoring: str = 'f1_micro',
    log_fn=print,
) -> Tuple[float, int, dict]:
    """Run Stage 1 (optional, if FPE available) + Stage 2 REINFORCE training.

    Returns (best_score, best_epoch, info_dict).
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    N = len(original_cols)
    device = torch.device(config.device)

    controller = NFSController(num_features=N, max_order=config.max_order).to(device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=config.lr)

    # baseline cv score (no engineered features added — just originals)
    baseline_score = _cv_score_on_columns(
        original_cols, y, task,
        cv=config.cv, n_estimators=config.n_estimators, scoring=scoring,
    )
    log_fn(f"[E-AFE] baseline cv score (original features only): {baseline_score:.6f}")

    # rolling moving average for variance reduction in REINFORCE
    reward_baseline = baseline_score

    best_score = baseline_score
    best_epoch = -1
    best_columns = [c.copy() for c in original_cols]  # fallback if nothing beats baseline
    best_actions = None
    replay_buffer = []  # list of (action_seqs np array, fpe_mean)

    # -------- Stage 1 --------
    use_stage1 = (fpe_predictor is not None) and (config.stage1_epochs > 0)
    delta_A_max = 0.05  # initial guesses, refined as Stage 1 proceeds
    delta_A_min = -0.05
    if use_stage1:
        log_fn(f"[E-AFE] Stage 1: {config.stage1_epochs} epochs, FPE-only reward")
        t0 = time.time()
        for epoch in range(config.stage1_epochs):
            batch_loss = 0.0
            batch_fpe_mean = 0.0
            for _ in range(config.batch_size):
                actions, log_probs, entropies = controller.sample()
                actions_np = actions.detach().cpu().numpy()
                new_cols = build_transformed_columns(actions_np, original_cols)
                fpe_probs = fpe_predictor.score_columns(new_cols)
                rewards = _stage1_reward_per_agent(
                    fpe_probs, baseline_score, delta_A_max, delta_A_min,
                    config.fpe_train_thre,
                )
                batch_fpe_mean += float(fpe_probs.mean())

                # advantage = rewards - moving baseline
                advantages = torch.from_numpy(
                    (rewards - reward_baseline).astype(np.float32)
                ).to(device).unsqueeze(-1)  # (N, 1) broadcasts over max_order
                # spread per-agent advantage across its sequence length
                advantages = advantages.expand_as(log_probs)

                loss = controller.reinforce_loss(
                    log_probs, advantages, entropies=entropies,
                    entropy_coef=config.entropy_coef,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(controller.parameters(), 5.0)
                optimizer.step()
                batch_loss += float(loss.item())

                # remember promising features in replay buffer
                if (fpe_probs > 0.5).any():
                    replay_buffer.append((actions_np.copy(), float(fpe_probs.mean())))
                    if len(replay_buffer) > config.replay_buffer_capacity:
                        # keep the most promising
                        replay_buffer.sort(key=lambda x: -x[1])
                        replay_buffer = replay_buffer[:config.replay_buffer_capacity]

                # refine delta_A_max/min using FPE-derived rewards
                delta_A_max = max(delta_A_max, float(rewards.max() - baseline_score))
                delta_A_min = min(delta_A_min, float(rewards.min() - baseline_score))

            avg_loss = batch_loss / config.batch_size
            avg_fpe = batch_fpe_mean / config.batch_size
            if (epoch + 1) % max(1, config.stage1_epochs // 5) == 0 or epoch == 0:
                log_fn(f"[E-AFE] Stage1 epoch {epoch + 1}/{config.stage1_epochs} "
                       f"loss={avg_loss:.4f} E[FPE]={avg_fpe:.3f} "
                       f"|replay|={len(replay_buffer)}")
        log_fn(f"[E-AFE] Stage 1 done in {time.time() - t0:.1f}s, "
               f"replay buffer holds {len(replay_buffer)} promising action sequences")
    else:
        log_fn("[E-AFE] Stage 1 skipped (no FPE model or stage1_epochs=0)")

    # -------- Stage 2 --------
    log_fn(f"[E-AFE] Stage 2: {config.stage2_epochs} epochs, real cv reward "
           f"(FPE filter thresh = {config.fpe_filter_thresh if fpe_predictor else 'off'})")
    t0 = time.time()
    n_real_evals = 0
    n_filtered = 0

    for epoch in range(config.stage2_epochs):
        epoch_best = -np.inf
        actions, log_probs, entropies = controller.sample()
        actions_np = actions.detach().cpu().numpy()
        new_cols = build_transformed_columns(actions_np, original_cols)

        # FPE filtering: keep agents whose generated column passes threshold
        if fpe_predictor is not None and config.fpe_filter_thresh > 0:
            fpe_probs = fpe_predictor.score_columns(new_cols)
            keep_mask = fpe_probs >= config.fpe_filter_thresh
            n_filtered += int((~keep_mask).sum())
        else:
            keep_mask = np.ones(N, dtype=bool)

        if not keep_mask.any():
            # nothing passed the filter — fall back to evaluating originals
            score = baseline_score
        else:
            kept_cols = [c if keep else original_cols[i]
                         for i, (c, keep) in enumerate(zip(new_cols, keep_mask))]
            try:
                score = _cv_score_on_columns(
                    kept_cols, y, task,
                    cv=config.cv, n_estimators=config.n_estimators, scoring=scoring,
                )
            except Exception as e:
                log_fn(f"[E-AFE] cv failed: {type(e).__name__}: {e}")
                score = baseline_score
            n_real_evals += 1

        if score > best_score:
            best_score = score
            best_epoch = epoch
            # cache the columns at the best epoch for downstream re-evaluation
            best_columns = [c.copy() for c in (kept_cols if keep_mask.any() else original_cols)]
            best_actions = actions_np.copy()
        epoch_best = max(epoch_best, score)

        # Reward: gain over running baseline, broadcast across all (N, T) positions.
        reward = score - reward_baseline
        advantages = torch.full_like(log_probs, float(reward))
        loss = controller.reinforce_loss(
            log_probs, advantages, entropies=entropies, entropy_coef=config.entropy_coef
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 5.0)
        optimizer.step()

        # update moving baseline (exponential)
        reward_baseline = 0.9 * reward_baseline + 0.1 * score

        if (epoch + 1) % max(1, config.stage2_epochs // 10) == 0 or epoch == 0:
            log_fn(f"[E-AFE] Stage2 epoch {epoch + 1}/{config.stage2_epochs} "
                   f"score={score:.4f} best={best_score:.4f} "
                   f"filtered={n_filtered}")

    log_fn(f"[E-AFE] Stage 2 done in {time.time() - t0:.1f}s, "
           f"{n_real_evals} cv evals performed, {n_filtered} agents filtered by FPE")

    info = {
        'baseline_score': baseline_score,
        'best_score': best_score,
        'best_epoch': best_epoch,
        'n_real_evals': n_real_evals,
        'n_filtered': n_filtered,
        'stage1_used': use_stage1,
        'best_columns': best_columns,         # list[np.ndarray] — for downstream eval
        'best_actions': best_actions,         # np.ndarray | None — controller actions
    }
    return best_score, best_epoch, info
