"""Feature Pre-Evaluation (FPE) classifier — inference side.

A small PyTorch MLP that takes a column's CCWS signature (d floats) and
predicts whether the feature is "useful" — i.e., whether including it would
raise downstream cv score by more than `thre`.

The classifier is trained offline (see fpe_train.py) and the resulting
state_dict + metadata is saved to a single .pt file. At search time we load
this file once per dataset, then call `predict_proba` on every candidate
feature column in batches.

If no pre-trained FPE checkpoint exists, run E-AFE with `--use_fpe=0` to
skip Stages 1 / FPE-filtering entirely (the controller then trains end-to-end
on real cv scores, equivalent to vanilla NFS in PyTorch).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from autolearn.feat_selection.eafe.minhash import column_signature


class FPEClassifier(nn.Module):
    """3-layer MLP: signature (d) -> 1 logit (sigmoid -> probability)."""

    def __init__(self, signature_dim=48, hidden=128, dropout=0.1):
        super().__init__()
        self.signature_dim = signature_dim
        self.net = nn.Sequential(
            nn.Linear(signature_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, sig):
        return self.net(sig).squeeze(-1)  # logits


class FPEPredictor:
    """Stateful wrapper that pairs an FPEClassifier with the CCWS signature
    generator. Use ``score_columns`` on a list of numpy 1D arrays to get
    probabilities in [0, 1].
    """

    def __init__(self, model, signature_dim=48, ccws_seed=0, device='cpu'):
        self.model = model.to(device).eval()
        self.signature_dim = signature_dim
        self.ccws_seed = ccws_seed
        self.device = device

    @torch.no_grad()
    def score_columns(self, columns):
        """columns: iterable of 1D numpy arrays. Returns 1D numpy array of probabilities."""
        if len(columns) == 0:
            return np.zeros(0, dtype=np.float32)
        sigs = np.stack([
            column_signature(c, d=self.signature_dim, seed=self.ccws_seed)
            for c in columns
        ]).astype(np.float32)
        x = torch.from_numpy(sigs).to(self.device)
        logits = self.model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        return probs


def save_fpe(model, path, metadata=None):
    payload = {
        'state_dict': model.state_dict(),
        'signature_dim': model.signature_dim,
        'metadata': metadata or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_fpe(path, device='cpu'):
    """Returns (FPEPredictor, metadata_dict)."""
    payload = torch.load(path, map_location=device)
    dim = payload['signature_dim']
    # hidden size isn't stored explicitly — try the default and fall back
    metadata = payload.get('metadata', {})
    hidden = metadata.get('hidden', 128)
    model = FPEClassifier(signature_dim=dim, hidden=hidden)
    model.load_state_dict(payload['state_dict'])
    predictor = FPEPredictor(model, signature_dim=dim, device=device)
    return predictor, metadata
