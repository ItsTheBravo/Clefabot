"""Self-play rollout collection + the M5 training loop (spec §8).

Rollouts are collected through poke-env's ``Player.choose_move`` path (the same
battle_against machinery the imitation pipeline uses — proven stable across
hundreds of local games), not through PokeEnv's synchronous step loop, which
showed start-of-battle mask races and a stalled episode in smoke testing.
``RolloutActor`` samples from the current policy, records (obs, action, mask,
logprob, value) at every decision, and converts battle-state deltas into the
same shaped reward as ``battle_env.ClefabotDoublesEnv`` defines.

Every finished game is logged through the shared data layer (spec §8: one
pipeline for training and analytics). Checkpoints are saved every
``checkpoint_every`` games and registered in the ``checkpoints`` table.

Opponents per game are drawn from:
- self-play mirror (current policy vs itself)
- a frozen recent checkpoint (when one exists)
- the scripted MegaTeacher (anchor opponent, keeps play grounded)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import Player

from ..data_layer.writer import GameLogger
from ..env.actions import action_to_order, legal_action_mask
from ..env.features import encode_battle
from ..env.players import REG_MB_FORMAT
from ..imitation.collect import MegaTeacher
from ..imitation.net import PolicyValueNet
from ..team.parser import parse_team
from .battle_env import W_FAINT, W_HP, W_WIN
from .ppo import PPOConfig, RolloutBuffer, ppo_update, sample_action


def _potential(battle) -> float:
    my_hp = sum(m.current_hp_fraction or 0.0 for m in battle.team.values())
    opp_hp = sum(m.current_hp_fraction or 0.0 for m in battle.opponent_team.values())
    my_f = sum(1 for m in battle.team.values() if m.fainted)
    opp_f = sum(1 for m in battle.opponent_team.values() if m.fainted)
    return W_HP * (my_hp - opp_hp) / 4.0 + W_FAINT * (opp_f - my_f)


class RolloutActor(Player):
    """Samples from the policy and records transitions for PPO."""

    def __init__(self, *args, net: PolicyValueNet, greedy: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net
        self.greedy = greedy
        # battle_tag -> list of pending transition dicts
        self._traj: dict[str, list[dict]] = {}
        self._last_pot: dict[str, float] = {}

    def choose_move(self, battle):
        try:
            obs = encode_battle(battle)
            mask = legal_action_mask(battle)
            obs_t = torch.from_numpy(obs).float()
            mask_t = torch.from_numpy(mask).bool()
            if self.greedy:
                action = self.net.act(obs_t, mask_t, greedy=True).numpy()
                logprob, value = 0.0, 0.0
            else:
                a, lp, v = sample_action(self.net, obs_t, mask_t)
                action, logprob, value = a.numpy(), float(lp), float(v)

            # Shaped step reward: potential difference since our last decision.
            pot = _potential(battle)
            traj = self._traj.setdefault(battle.battle_tag, [])
            if traj:
                traj[-1]["reward"] += pot - self._last_pot[battle.battle_tag]
            self._last_pot[battle.battle_tag] = pot

            traj.append({
                "obs": obs, "action": action.astype(np.int64), "mask": mask,
                "logprob": logprob, "value": value, "reward": 0.0, "done": False,
                "turn": battle.turn,
                "active_self": "+".join(m.species for m in battle.active_pokemon if m),
                "active_opp": "+".join(
                    m.species for m in battle.opponent_active_pokemon if m),
            })
            return action_to_order(
                np.asarray(action, dtype=np.int64), battle, strict=False)
        except Exception:
            return self.choose_random_move(battle)

    def finish_battle(self, battle_tag: str, won: bool | None) -> list[dict]:
        """Close a trajectory with the terminal reward; return transitions."""
        traj = self._traj.pop(battle_tag, [])
        self._last_pot.pop(battle_tag, None)
        if traj:
            if won is True:
                traj[-1]["reward"] += W_WIN
            elif won is False:
                traj[-1]["reward"] -= W_WIN
            traj[-1]["done"] = True
        return traj


async def run_session(team_text: str, db_path: str, *,
                      total_games: int = 200, games_per_iter: int = 20,
                      checkpoint_every: int = 100,
                      warm_start: str | None = "checkpoints/bc_baseline.pt",
                      out_dir: str = "checkpoints",
                      teacher_mix: float = 0.34,
                      seed: int = 0) -> dict:
    """The M5 loop: collect games -> PPO update -> repeat, all logged."""
    random.seed(seed)
    torch.manual_seed(seed)

    team_spec = parse_team(team_text)
    team = team_spec.to_showdown_team()
    logger = GameLogger(db_path)
    team_version = logger.register_team(team_spec, label="current")

    net = PolicyValueNet()
    if warm_start and Path(warm_start).exists():
        ckpt = torch.load(warm_start, map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt["state_dict"])
        print(f"warm-started from {warm_start}")
    frozen = PolicyValueNet()
    frozen.load_state_dict(net.state_dict())

    cfg = PPOConfig()
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    common = dict(
        battle_format=REG_MB_FORMAT,
        server_configuration=LocalhostServerConfiguration,
        team=team,
        max_concurrent_battles=5,
    )

    run_id = uuid.uuid4().hex[:6]
    games_done, wins = 0, 0
    session = {"iters": [], "checkpoints": []}

    def save_checkpoint(tag: str) -> str:
        cid = f"ppo_{run_id}_{tag}"
        path = Path(out_dir) / f"{cid}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": net.state_dict(), "obs_size": net.obs_size,
                    "arch": "PolicyValueNet"}, path)
        logger.conn.execute(
            "INSERT OR REPLACE INTO checkpoints"
            "(checkpoint_id, created_at, team_version, parent_id, path, is_champion)"
            " VALUES(?,?,?,?,?,0)",
            (cid, datetime.now(timezone.utc).isoformat(), team_version,
             warm_start, str(path)))
        logger.conn.commit()
        session["checkpoints"].append(cid)
        return cid

    iter_idx = 0
    while games_done < total_games:
        iter_idx += 1
        n = min(games_per_iter, total_games - games_done)
        actor = RolloutActor(
            account_configuration=AccountConfiguration(f"RL{run_id}{iter_idx}", None),
            net=net, **common)

        r = random.random()
        if r < teacher_mix:
            opp = MegaTeacher(account_configuration=AccountConfiguration(
                f"T{run_id}{iter_idx}", None), **common)
            source = "local_opponent_pool"  # scripted anchor opponent
        elif r < teacher_mix + (1 - teacher_mix) / 2:
            opp = RolloutActor(account_configuration=AccountConfiguration(
                f"F{run_id}{iter_idx}", None), net=frozen, greedy=True, **common)
            source = "local_selfplay"
        else:
            opp = RolloutActor(account_configuration=AccountConfiguration(
                f"S{run_id}{iter_idx}", None), net=net, **common)
            source = "local_selfplay"

        t0 = time.time()
        await actor.battle_against(opp, n_battles=n)

        buffer = RolloutBuffer()
        for tag, battle in actor.battles.items():
            traj = actor.finish_battle(tag, battle.won)
            result = "win" if battle.won else ("tie" if battle.won is None else "loss")
            logger.log_game(
                team_version=team_version, result=result,
                turn_count=battle.turn, source=source,
                checkpoint_id=f"ppo_{run_id}_live",
                game_format=REG_MB_FORMAT,
                turns=[{
                    "turn_number": t["turn"],
                    "win_probability": None if t["value"] == 0.0 else
                        float(torch.tanh(torch.tensor(t["value"]))),
                    "active_self": t["active_self"],
                    "active_opponent": t["active_opp"],
                    "action_taken": json.dumps(t["action"].tolist()),
                } for t in traj],
            )
            for t in traj:
                buffer.add(t["obs"], t["action"], t["mask"],
                           t["logprob"], t["value"], t["reward"], t["done"])

        games_done += actor.n_finished_battles
        wins += actor.n_won_battles
        metrics = ppo_update(net, optimizer, buffer, cfg) if len(buffer) else {}
        dt = time.time() - t0
        print(f"iter {iter_idx}: {n} games vs {source} in {dt:.0f}s | "
              f"transitions={len(buffer)} | wr_so_far={wins/max(1,games_done):.3f} | "
              + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
        session["iters"].append({"iter": iter_idx, "games": n, "source": source,
                                 "transitions": len(buffer), **metrics})

        # Refresh the frozen opponent and checkpoint on schedule.
        if games_done % checkpoint_every < games_per_iter:
            frozen.load_state_dict(net.state_dict())
            cid = save_checkpoint(f"g{games_done}")
            print(f"checkpoint saved: {cid}", flush=True)

    cid = save_checkpoint("final")
    print(f"final checkpoint: {cid} | overall wr={wins/max(1,games_done):.3f}")
    logger.close()
    return session


def main() -> None:
    ap = argparse.ArgumentParser(description="M5 PPO self-play session")
    ap.add_argument("--team", default="config/current_team.txt")
    ap.add_argument("--db", default="data/clefabot.sqlite")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--games-per-iter", type=int, default=20)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--warm-start", default="checkpoints/bc_baseline.pt")
    args = ap.parse_args()
    asyncio.run(run_session(
        Path(args.team).read_text(), args.db,
        total_games=args.games, games_per_iter=args.games_per_iter,
        checkpoint_every=args.checkpoint_every, warm_start=args.warm_start))


if __name__ == "__main__":
    main()
