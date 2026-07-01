"""Shared policy/value network (spec §7 architecture requirement).

This is the single network architecture used by both the imitation baseline
(behavior cloning, M4) and the PPO agent (M5). Because they share the exact
module definition, "warm start" is a literal weight copy: PPO loads this net's
trained state_dict rather than starting from random init.

Layout:
    obs (OBS_SIZE) -> shared MLP trunk
      -> policy head: one set of logits per active slot (N_ACTIVE_SLOTS x
         PER_SLOT_ACTIONS), matching the MultiDiscrete action space
      -> value head: scalar state value (used by PPO's critic and, exposed at
         inference, as the win-estimate readout in the turn-analysis tool §11)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..env.actions import N_ACTIVE_SLOTS, PER_SLOT_ACTIONS
from ..env.features import OBS_SIZE

HIDDEN = (256, 256)


class PolicyValueNet(nn.Module):
    def __init__(self, obs_size: int = OBS_SIZE, hidden: tuple[int, ...] = HIDDEN):
        super().__init__()
        self.obs_size = obs_size
        self.n_slots = N_ACTIVE_SLOTS
        self.per_slot = PER_SLOT_ACTIONS

        layers: list[nn.Module] = []
        prev = obs_size
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, N_ACTIVE_SLOTS * PER_SLOT_ACTIONS)
        self.value_head = nn.Linear(prev, 1)

    def forward(self, obs: torch.Tensor):
        """Return (policy_logits [B, n_slots, per_slot], value [B])."""
        z = self.trunk(obs)
        logits = self.policy_head(z).view(-1, self.n_slots, self.per_slot)
        value = self.value_head(z).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor | None = None,
            greedy: bool = True) -> torch.Tensor:
        """Choose a per-slot action index for a single observation.

        ``mask`` (n_slots, per_slot bool) zeroes out illegal actions before the
        argmax/sample. Returns a (n_slots,) long tensor.
        """
        logits, _ = self.forward(obs.unsqueeze(0))
        logits = logits.squeeze(0)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        if greedy:
            return logits.argmax(dim=-1)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    @torch.no_grad()
    def action_probs(self, obs: torch.Tensor, mask: torch.Tensor | None = None):
        """Per-slot legal-action probability distributions + value estimate.

        This backs the "analyze this turn" tool (spec §11 / plan §1.2): the
        move %s are these policy probabilities, the win estimate is the value.
        """
        logits, value = self.forward(obs.unsqueeze(0))
        logits = logits.squeeze(0)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        return probs, value.squeeze(0)
