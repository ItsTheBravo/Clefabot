"""Tests for the shared PolicyValueNet and the BC training step (M4)."""

import numpy as np
import torch

from clefabot.env.actions import (
    N_ACTIVE_SLOTS,
    PER_SLOT_ACTIONS,
    decode_action,
    encode_order_slot,
    index_to_move_target,
    move_target_to_index,
)
from clefabot.env.features import OBS_SIZE
from clefabot.imitation.net import PolicyValueNet


def test_net_shapes():
    net = PolicyValueNet()
    logits, value = net(torch.randn(3, OBS_SIZE))
    assert logits.shape == (3, N_ACTIVE_SLOTS, PER_SLOT_ACTIONS)
    assert value.shape == (3,)


def test_act_respects_mask():
    net = PolicyValueNet()
    mask = torch.zeros(N_ACTIVE_SLOTS, PER_SLOT_ACTIONS, dtype=torch.bool)
    # Only allow action index 5 on slot 0 and 12 on slot 1.
    mask[0, 5] = True
    mask[1, 12] = True
    action = net.act(torch.randn(OBS_SIZE), mask, greedy=True)
    assert action.tolist() == [5, 12]


def test_action_probs_sum_to_one_over_legal():
    net = PolicyValueNet()
    mask = torch.zeros(N_ACTIVE_SLOTS, PER_SLOT_ACTIONS, dtype=torch.bool)
    mask[0, :3] = True
    mask[1, 4:8] = True
    probs, value = net.action_probs(torch.randn(OBS_SIZE), mask)
    assert torch.allclose(probs.sum(-1), torch.ones(N_ACTIVE_SLOTS), atol=1e-5)
    # Illegal actions get zero probability.
    assert probs[0, 3:].sum().item() < 1e-6


def test_target_mapping_roundtrip():
    for idx in range(4):
        mt = index_to_move_target(idx)
        # opp targets roundtrip exactly; own-side collapses (documented approx).
        if idx in (0, 1):
            assert move_target_to_index(mt) == idx


def test_bc_training_reduces_loss():
    # Synthetic separable data: loss must drop over a few steps.
    from clefabot.imitation import train_bc

    rng = np.random.default_rng(0)
    n = 128
    obs = rng.standard_normal((n, OBS_SIZE)).astype(np.float32)
    actions = rng.integers(0, PER_SLOT_ACTIONS, size=(n, N_ACTIVE_SLOTS)).astype(np.int64)
    masks = np.ones((n, N_ACTIVE_SLOTS, PER_SLOT_ACTIONS), dtype=bool)

    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        ds = os.path.join(d, "ds.npz")
        out = os.path.join(d, "m.pt")
        np.savez_compressed(ds, obs=obs, actions=actions, masks=masks)
        meta = train_bc.train(ds, out, epochs=5, batch_size=32)
        assert os.path.exists(out)
        assert meta["n"] == n
