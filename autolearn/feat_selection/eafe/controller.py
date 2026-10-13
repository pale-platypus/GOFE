"""RNN-based controller for E-AFE / NFS.

One LSTM rolls forward for ``max_order`` time steps per agent. We have N
agents (one per original feature) and batch them as the LSTM's batch
dimension — so one forward call yields all agents' action distributions in
parallel. This is a faithful reimplementation of NFS's architecture in
PyTorch.

Per-agent action space:
    1 (no-op) + num_unary + (num_features - 1) * num_binary
Per-agent sequence length: max_order

Returns ``(actions, log_probs)`` of shape ``(num_features, max_order)``
during sampling; ``loss = -mean(log_probs * advantage)`` for REINFORCE.
"""

import torch
import torch.nn as nn

from autolearn.feat_selection.eafe import operations


class NFSController(nn.Module):
    def __init__(self, num_features, max_order=5, hidden=64,
                 num_unary=operations.NUM_UNARY, num_binary=operations.NUM_BINARY):
        super().__init__()
        self.num_features = num_features
        self.max_order = max_order
        self.num_unary = num_unary
        self.num_binary = num_binary
        self.num_actions = operations.num_actions(num_features, num_unary, num_binary)
        self.hidden = hidden

        # Per-agent learnable "start" token (one row per agent, shared input embedding for noop slot).
        self.agent_init = nn.Parameter(torch.randn(num_features, hidden) * 0.01)
        # Embedding for previous-action -> input at next timestep.
        self.action_embed = nn.Embedding(self.num_actions, hidden)
        self.rnn = nn.LSTMCell(hidden, hidden)
        self.head = nn.Linear(hidden, self.num_actions)

    def _zero_state(self, batch, device):
        h = torch.zeros(batch, self.hidden, device=device)
        c = torch.zeros(batch, self.hidden, device=device)
        return h, c

    def sample(self, deterministic=False):
        """Roll the LSTM forward for ``max_order`` steps with ``num_features`` agents
        in parallel. Returns:
            actions:   LongTensor (num_features, max_order)
            log_probs: FloatTensor (num_features, max_order)
            entropies: FloatTensor (num_features, max_order)   — useful for entropy regularization
        """
        device = self.agent_init.device
        N = self.num_features
        T = self.max_order

        h, c = self._zero_state(N, device)
        x = self.agent_init  # (N, hidden)

        actions = torch.zeros(N, T, dtype=torch.long, device=device)
        log_probs = torch.zeros(N, T, device=device)
        entropies = torch.zeros(N, T, device=device)

        for t in range(T):
            h, c = self.rnn(x, (h, c))
            logits = self.head(h)  # (N, num_actions)
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                a = logits.argmax(dim=-1)
            else:
                a = dist.sample()
            actions[:, t] = a
            log_probs[:, t] = dist.log_prob(a)
            entropies[:, t] = dist.entropy()
            x = self.action_embed(a)

        return actions, log_probs, entropies

    def reinforce_loss(self, log_probs, advantages, entropies=None, entropy_coef=0.0):
        """Standard REINFORCE policy loss.

        log_probs:  (B, ...) per-step log-prob of taken actions
        advantages: (B, ...) advantage estimates (typically rewards-to-go - baseline)
        Returns scalar tensor to minimize.
        """
        loss = -(log_probs * advantages.detach()).mean()
        if entropies is not None and entropy_coef > 0.0:
            loss = loss - entropy_coef * entropies.mean()
        return loss
