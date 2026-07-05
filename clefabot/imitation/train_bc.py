"""Behavior-cloning trainer (spec §7 / M4).

Supervised training of the shared PolicyValueNet to imitate the teacher's
per-slot action choices. Illegal actions are masked out of the loss. The saved
checkpoint is the warm-start parent for PPO (M5): same architecture, so PPO
loads these weights directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..env.features import OBS_SIZE
from .net import PolicyValueNet


def train(dataset_path: str, out_path: str, epochs: int = 20,
          batch_size: int = 256, lr: float = 1e-3, val_frac: float = 0.1,
          seed: int = 0) -> dict:
    torch.manual_seed(seed)
    data = np.load(dataset_path)
    obs = torch.from_numpy(data["obs"]).float()
    actions = torch.from_numpy(data["actions"]).long()      # [N, n_slots]
    masks = torch.from_numpy(data["masks"]).bool()          # [N, n_slots, per_slot]

    # Defensive: drop any rows with negative (default/forfeit sentinel) labels.
    keep = (actions >= 0).all(dim=1)
    if (~keep).any():
        print(f"dropping {(~keep).sum().item()} rows with sentinel labels")
        obs, actions, masks = obs[keep], actions[keep], masks[keep]

    n = obs.shape[0]
    if n == 0:
        raise ValueError("empty dataset")
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    obs, actions, masks = obs[perm], actions[perm], masks[perm]
    n_val = max(1, int(n * val_frac))
    tr = slice(n_val, n)
    va = slice(0, n_val)

    net = PolicyValueNet(obs_size=OBS_SIZE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    def accuracy(o, a, m):
        # Inference masks illegal actions, so measure accuracy the same way.
        net.eval()
        with torch.no_grad():
            logits, _ = net(o)
            pred = logits.masked_fill(~m, float("-inf")).argmax(-1)
            return (pred == a).float().mean().item()

    history = []
    for epoch in range(epochs):
        net.train()
        idx = torch.randperm(obs[tr].shape[0])
        o_tr, a_tr, m_tr = obs[tr][idx], actions[tr][idx], masks[tr][idx]
        total = 0.0
        for i in range(0, o_tr.shape[0], batch_size):
            ob, ac = o_tr[i:i + batch_size], a_tr[i:i + batch_size]
            # Plain behavior cloning toward the teacher's action on raw logits;
            # legality masking is applied at inference, not in the loss.
            logits, _ = net(ob)               # [B, n_slots, per_slot]
            loss = loss_fn(
                logits.reshape(-1, logits.shape[-1]), ac.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * ob.shape[0]
        tr_loss = total / max(1, o_tr.shape[0])
        val_acc = accuracy(obs[va], actions[va], masks[va])
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_acc": val_acc})
        print(f"epoch {epoch:2d}  train_loss={tr_loss:.4f}  val_acc={val_acc:.3f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": net.state_dict(), "obs_size": OBS_SIZE,
         "arch": "PolicyValueNet", "history": history},
        out_path,
    )
    meta = {"dataset": dataset_path, "n": n, "epochs": epochs,
            "final_val_acc": history[-1]["val_acc"], "out": out_path}
    Path(out_path).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"saved BC checkpoint -> {out_path}  (val_acc={meta['final_val_acc']:.3f})")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Behavior-clone the teacher (M4)")
    ap.add_argument("--dataset", default="data/bc_dataset.npz")
    ap.add_argument("--out", default="checkpoints/bc_baseline.pt")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    train(args.dataset, args.out, epochs=args.epochs)


if __name__ == "__main__":
    main()
