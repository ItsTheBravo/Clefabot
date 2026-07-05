"""Evaluation gate + promotion rule (spec §9 / plan M7).

Any two checkpoints can be pitted against each other, team held constant on
both sides. The challenger is promoted to champion only if its win rate over
the fixed evaluation set clears an explicit threshold (default >0.55 — a rule,
not a judgment call). Every evaluation is logged to the ``evaluations`` table
and champion status is updated in ``checkpoints``.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from poke_env import AccountConfiguration, LocalhostServerConfiguration

from ..data_layer.writer import GameLogger
from ..env.players import REG_MB_FORMAT
from ..imitation.evaluate import BCPlayer, load_net
from ..team.parser import parse_team

DEFAULT_THRESHOLD = 0.55
DEFAULT_GAMES = 100


async def run_gate(champion_ckpt: str, challenger_ckpt: str, team_text: str,
                   db_path: str, *, n_games: int = DEFAULT_GAMES,
                   threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Play challenger vs champion; log the result; return the decision."""
    team = parse_team(team_text).to_showdown_team()
    common = dict(
        battle_format=REG_MB_FORMAT,
        server_configuration=LocalhostServerConfiguration,
        team=team,
        max_concurrent_battles=10,
    )
    tag = datetime.now(timezone.utc).strftime("%H%M%S")
    challenger = BCPlayer(
        account_configuration=AccountConfiguration(f"Chal{tag}", None),
        net=load_net(challenger_ckpt), **common)
    champion = BCPlayer(
        account_configuration=AccountConfiguration(f"Champ{tag}", None),
        net=load_net(champion_ckpt), **common)

    await challenger.battle_against(champion, n_battles=n_games)

    games = challenger.n_finished_battles
    win_rate = challenger.n_won_battles / max(1, games)
    promoted = win_rate > threshold

    logger = GameLogger(db_path)
    logger.conn.execute(
        "INSERT INTO evaluations"
        "(timestamp, champion_id, challenger_id, n_games, win_rate, threshold,"
        " promoted) VALUES(?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), Path(champion_ckpt).stem,
         Path(challenger_ckpt).stem, games, win_rate, threshold, int(promoted)))
    if promoted:
        logger.conn.execute("UPDATE checkpoints SET is_champion=0")
        logger.conn.execute(
            "UPDATE checkpoints SET is_champion=1 WHERE checkpoint_id=?",
            (Path(challenger_ckpt).stem,))
    logger.conn.commit()
    logger.close()

    decision = {"challenger": challenger_ckpt, "champion": champion_ckpt,
                "games": games, "win_rate": win_rate, "threshold": threshold,
                "promoted": promoted}
    print(f"gate: challenger wr={win_rate:.3f} over {games} games "
          f"(threshold {threshold}) -> {'PROMOTED' if promoted else 'retained'}")
    return decision


def main() -> None:
    ap = argparse.ArgumentParser(description="Checkpoint evaluation gate (M7)")
    ap.add_argument("--champion", required=True)
    ap.add_argument("--challenger", required=True)
    ap.add_argument("--team", default="config/current_team.txt")
    ap.add_argument("--db", default="data/clefabot.sqlite")
    ap.add_argument("--games", type=int, default=DEFAULT_GAMES)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = ap.parse_args()
    asyncio.run(run_gate(args.champion, args.challenger,
                         Path(args.team).read_text(), args.db,
                         n_games=args.games, threshold=args.threshold))


if __name__ == "__main__":
    main()
