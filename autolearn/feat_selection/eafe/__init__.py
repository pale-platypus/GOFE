"""E-AFE: Efficient Automated Feature Engineering.

PyTorch reimplementation of "Toward Efficient Automated Feature Engineering"
(Wang et al., ICDE 2023). Builds on NFS's RNN-controller + REINFORCE pipeline
with two additions:

  1. Feature Pre-Evaluation (FPE) Model — a binary classifier trained offline
     on public datasets that predicts whether a generated feature would help
     downstream performance. Uses MinHash signatures (CCWS-style) over feature
     columns so the classifier sees a fixed-size representation per feature.

  2. Two-stage policy training — Stage 1 uses FPE-only reward to warm-start
     the controller (cheap), Stage 2 uses real cross-val reward but filters
     candidates through FPE first (fast).

This implementation lives in the main PyTorch env (no TF1 / Keras / Java
dependencies), unlike the vendored NFS baseline.
"""
