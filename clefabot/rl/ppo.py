"""Compact PPO trainer over the shared PolicyValueNet (spec §8 / plan M5).

Deviation from spec §3 (documented): the spec lists stable-baselines3's PPO.
SB3 cannot natively express this setup — a PettingZoo two-agent battle env,
MultiDiscrete action masking (sb3-contrib only), and warm-starting from an
externally trained network all require substantial custom-policy surgery.
Rather than adapt three layers of SB3 internals, this module implements the
PPO-clip algorithm directly (~150 lines) on the exact same PolicyValueNet the
imitation baseline trained — which makes the §7 warm start a literal
``load_state_dict`` and keeps action masking and per-game logging first-class.
The algorithm is standard PPO (GAE, clipped surrogate, entropy bonus, value
loss), so nothing conceptual is lost relative to SB3's implementation.

Self-play data collection lives in ``selfplay.py``; this file is pure
optimization: (obs, action, mask, logprob, value, reward, done) batches in,
updated network out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ..imitation.net import PolicyValueNet


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 256


@dataclass
class RolloutBuffer:
    """Flat storage for one training iteration's transitions."""

    obs: list = field(default_factory=list)          # [T, OBS]
    actions: list = field(default_factory=list)      # [T, 2]
    masks: list = field(default_factory=list)        # [T, 2, A]
    logprobs: list = field(default_factory=list)     # [T]
    values: list = field(default_factory=list)       # [T]
    rewards: list = field(default_factory=list)      # [T]
    dones: list = field(default_factory=list)        # [T]

    def add(self, obs, action, mask, logprob, value, reward, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.masks.append(mask)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.obs)


def masked_dist(logits: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
    """Per-slot categorical over legal actions. logits/mask: [B, slots, A]."""
    return torch.distributions.Categorical(
        logits=logits.masked_fill(~mask, float("-inf")))


def evaluate_actions(net: PolicyValueNet, obs: torch.Tensor, mask: torch.Tensor,
                     actions: torch.Tensor):
    """Return (joint logprob, entropy, value) for stored transitions."""
    logits, value = net(obs)
    dist = masked_dist(logits, mask)
    logprob = dist.log_prob(actions).sum(-1)       # sum over the 2 slots
    entropy = dist.entropy().sum(-1)
    return logprob, entropy, value


@torch.no_grad()
def sample_action(net: PolicyValueNet, obs: torch.Tensor, mask: torch.Tensor):
    """Sample a (2,)-action; return (action, joint logprob, value)."""
    logits, value = net(obs.unsqueeze(0))
    dist = masked_dist(logits, mask.unsqueeze(0))
    action = dist.sample()                          # [1, 2]
    logprob = dist.log_prob(action).sum(-1)         # [1]
    return action.squeeze(0), logprob.squeeze(0), value.squeeze(0)


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """Standard GAE-lambda advantages + returns (numpy in, numpy out)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + np.asarray(values, dtype=np.float32)
    return adv, returns


def ppo_update(net: PolicyValueNet, optimizer: torch.optim.Optimizer,
               buffer: RolloutBuffer, cfg: PPOConfig) -> dict:
    """Run PPO-clip epochs over the buffer; return loss metrics."""
    obs = torch.as_tensor(np.asarray(buffer.obs), dtype=torch.float32)
    actions = torch.as_tensor(np.asarray(buffer.actions), dtype=torch.long)
    masks = torch.as_tensor(np.asarray(buffer.masks), dtype=torch.bool)
    old_logprobs = torch.as_tensor(np.asarray(buffer.logprobs), dtype=torch.float32)
    values = np.asarray(buffer.values, dtype=np.float32)

    adv, returns = compute_gae(
        buffer.rewards, values, buffer.dones, last_value=0.0,
        gamma=cfg.gamma, lam=cfg.gae_lambda)
    adv_t = torch.as_tensor(adv)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    returns_t = torch.as_tensor(returns)

    n = len(buffer)
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0}
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, cfg.minibatch_size):
            idx = perm[i:i + cfg.minibatch_size]
            logprob, entropy, value = evaluate_actions(
                net, obs[idx], masks[idx], actions[idx])
            ratio = torch.exp(logprob - old_logprobs[idx])
            surr1 = ratio * adv_t[idx]
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(value, returns_t[idx])
            loss = (policy_loss + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy.mean())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += entropy.mean().item()
            metrics["updates"] += 1

    u = max(1, metrics.pop("updates"))
    return {k: v / u for k, v in metrics.items()}
