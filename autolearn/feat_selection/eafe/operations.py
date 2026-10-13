"""Feature transformation operators for E-AFE.

Mirrors the paper's choice: 4 unary + 5 binary operators (Section IV-A4).
Each operator is numerically safe — handles zeros/negatives/nans without
producing inf or nan in normal operation.

Action encoding for a feature with N original features:
    0                                            -> no-op (keep current)
    [1, num_unary]                               -> apply unary op (1..num_unary)
    [num_unary + 1, num_unary + (N-1) * num_binary]
                                                 -> apply binary op with another original feature
Total = 1 + num_unary + (N - 1) * num_binary

Per-feature sequence: max_order actions applied in order.
"""

import numpy as np

_EPS = 1e-9
# Clip the per-step output to a finite range. Without this, cascading mul/div/
# inv across max_order steps (default 5) on datasets with large feature
# magnitudes (e.g. credit_default: raw amounts up to ~1e6) overflows float32
# during the downstream sklearn cv cast → ValueError("contains infinity") →
# RL training scores every epoch as failed and best_actions stays None.
# 1e15 leaves ~7 orders of magnitude of headroom under float32's ~3.4e38 cap.
_CLIP = 1e15


def _sanitize(x):
    """Replace NaN/Inf and clip to a finite range so downstream sklearn /
    pandas casts don't see non-finite values. Idempotent and cheap.
    """
    out = np.nan_to_num(x, nan=0.0, posinf=_CLIP, neginf=-_CLIP)
    return np.clip(out, -_CLIP, _CLIP)


# ------------------------ unary ops ------------------------

def op_log(x):
    return _sanitize(np.log(np.abs(x) + _EPS))


def op_sqrt(x):
    return _sanitize(np.sqrt(np.abs(x) + _EPS))


def op_inv(x):
    # Avoid division by ~0 producing huge magnitudes that overflow downstream.
    denom = np.where(np.abs(x) < _EPS, np.sign(x) * _EPS + _EPS, x)
    return _sanitize(1.0 / denom)


def op_minmax(x):
    lo = np.min(x)
    hi = np.max(x)
    rng = hi - lo
    if rng < _EPS:
        return np.zeros_like(x)
    return _sanitize((x - lo) / rng)


UNARY_OPS = [op_log, op_sqrt, op_inv, op_minmax]
UNARY_NAMES = ['log', 'sqrt', 'inv', 'minmax']
NUM_UNARY = len(UNARY_OPS)


# ------------------------ binary ops ------------------------

def op_add(a, b):
    return _sanitize(a + b)


def op_sub(a, b):
    return _sanitize(a - b)


def op_mul(a, b):
    return _sanitize(a * b)


def op_div(a, b):
    denom = np.where(np.abs(b) < _EPS, np.sign(b) * _EPS + _EPS, b)
    return _sanitize(a / denom)


def op_mod(a, b):
    bb = np.where(np.abs(b) < _EPS, _EPS, b)
    return _sanitize(np.mod(a, bb))


BINARY_OPS = [op_add, op_sub, op_mul, op_div, op_mod]
BINARY_NAMES = ['add', 'sub', 'mul', 'div', 'mod']
NUM_BINARY = len(BINARY_OPS)


# ------------------------ action / operator dispatch ------------------------

def num_actions(num_features, num_unary=NUM_UNARY, num_binary=NUM_BINARY):
    """Total action space size per agent (per original feature)."""
    return 1 + num_unary + (num_features - 1) * num_binary


def decode_action(action_idx, agent_idx, num_features,
                  num_unary=NUM_UNARY, num_binary=NUM_BINARY):
    """Translate a controller action index into ('noop' | 'unary' | 'binary', payload).

    Returns one of:
      ('noop', None)
      ('unary', op_index)       where op_index in [0, num_unary)
      ('binary', op_index, other_feat_index)
                                 where op_index in [0, num_binary),
                                       other_feat_index in [0, num_features) and != agent_idx
    """
    if action_idx == 0:
        return ('noop', None)
    action_idx -= 1
    if action_idx < num_unary:
        return ('unary', action_idx)
    action_idx -= num_unary
    op_index = action_idx % num_binary
    slot = action_idx // num_binary  # 0..num_features-2
    other_feat = slot if slot < agent_idx else slot + 1
    return ('binary', op_index, other_feat)


def apply_action(action_idx, agent_idx, current_col, original_cols,
                 num_unary=NUM_UNARY, num_binary=NUM_BINARY):
    """Apply a decoded action and return the resulting feature column (np array).

    current_col: the agent's current feature value after prior actions in the sequence.
    original_cols: the original feature matrix as a list of 1D arrays (used for binary ops).
    """
    decoded = decode_action(action_idx, agent_idx, num_features=len(original_cols),
                            num_unary=num_unary, num_binary=num_binary)
    kind = decoded[0]
    if kind == 'noop':
        return current_col
    if kind == 'unary':
        return UNARY_OPS[decoded[1]](current_col)
    # binary
    other = original_cols[decoded[2]]
    return BINARY_OPS[decoded[1]](current_col, other)


def action_name(action_idx, agent_idx, num_features,
                num_unary=NUM_UNARY, num_binary=NUM_BINARY):
    """Pretty-printed action label, used in logs."""
    decoded = decode_action(action_idx, agent_idx, num_features, num_unary, num_binary)
    if decoded[0] == 'noop':
        return 'noop'
    if decoded[0] == 'unary':
        return UNARY_NAMES[decoded[1]]
    return f"{BINARY_NAMES[decoded[1]]}(f{agent_idx},f{decoded[2]})"
