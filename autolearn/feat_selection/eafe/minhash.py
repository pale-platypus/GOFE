"""MinHash-family column signatures for E-AFE's FPE pre-evaluator.

Given a numeric feature column of arbitrary length M, produce a fixed-size
d-dimensional signature that:
  * captures distributional structure of the column,
  * is consistent across runs (same column -> same signature given seed),
  * is comparable between columns (similar distributions -> similar signatures).

We implement Canonical Consistent Weighted Sampling (CCWS, Wu et al. 2016) as
the default — this is what the E-AFE paper uses. The mathematical core: treat
each column as a weighted multiset where weights are |value|, and produce d
weighted MinHash samples. We return the log-weight at the sampled position
as the float feature for the classifier (this is a standard 'CWS scalar'
representation suitable for downstream MLPs).

For columns containing NaN/inf we substitute zeros; the FPE classifier handles
the resulting low-information signatures gracefully.
"""

import numpy as np


_EPS = 1e-9


def _sanitize(values):
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return v


def ccws_signature(values, d=48, seed=0):
    """Canonical Consistent Weighted Sampling signature.

    Returns a (d,) float64 array. Deterministic given (values, seed, d).

    Implementation notes:
      * CCWS treats the input as a weighted set. We use |value| as weights.
      * For each of d hash functions, sample per-element gamma random variables
        and find the argmin of c_i * exp(-r_i * t_i) where t_i = floor(log(w_i)/r_i + beta_i).
      * Signature entry k = log of weight at argmin position (a scalar form
        usable directly as a real-valued classifier feature).
    """
    w = np.abs(_sanitize(values)) + _EPS
    n = w.shape[0]
    if n == 0:
        return np.zeros(d, dtype=np.float64)

    rng = np.random.RandomState(int(seed))
    # (d, n) random matrices — Gamma(2, 1) for r, c; Uniform(0, 1) for beta.
    r = rng.gamma(shape=2.0, scale=1.0, size=(d, n)).astype(np.float64) + _EPS
    c = rng.gamma(shape=2.0, scale=1.0, size=(d, n)).astype(np.float64) + _EPS
    beta = rng.uniform(low=0.0, high=1.0, size=(d, n)).astype(np.float64)

    log_w = np.log(w)[np.newaxis, :]  # (1, n)
    t = np.floor(log_w / r + beta)    # (d, n)
    a = np.log(c) - r * t              # (d, n) — log of the CCWS score
    argmins = np.argmin(a, axis=1)     # (d,)
    sig = np.log(w[argmins] + _EPS)
    return sig.astype(np.float64)


def column_signature(values, d=48, seed=0, method='ccws'):
    """Public entry point. Currently only CCWS is supported; the method arg is
    kept so paper variants (PCWS/ICWS/LICWS) can be added later without API
    changes.
    """
    if method != 'ccws':
        raise ValueError(f"unsupported method {method!r}; only 'ccws' is implemented")
    return ccws_signature(values, d=d, seed=seed)
