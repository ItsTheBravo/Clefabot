"""M1 entry point: play one full local game and log it end to end (spec §13.1).

Runs the current team against a scripted opponent (same team both sides, held
constant per spec §9) on the local Showdown server and writes the game + turns
into the shared SQLite store.

Usage:
    python -m clefabot.cli.play --team config/current_team.txt --db data/clefabot.sqlite
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from poke_env import AccountConfiguration, LocalhostServerConfiguration

from ..data_layer.writer import GameLogger
from ..env.players import REG_MB_FORMAT, RecordingPlayer
from ..team.parser import parse_team


async def play_one_game(team_text: str, db_path: str, n_battles: int = 1) -> None:
    team = parse_team(team_text)
    team_string = team.to_showdown_team()

    logger = GameLogger(db_path)
    team_version = logger.register_team(team, label="current")

    common = dict(
        battle_format=REG_MB_FORMAT,
        server_configuration=LocalhostServerConfiguration,
        team=team_string,
        max_concurrent_battles=1,
    )
    agent = RecordingPlayer(
        account_configuration=AccountConfiguration("Clefabot", None), **common
    )
    opponent = RecordingPlayer(
        account_configuration=AccountConfiguration("Sparring", None), **common
    )

    await agent.battle_against(opponent, n_battles=n_battles)

    for battle_tag, battle in agent.battles.items():
        result = "win" if battle.won else ("tie" if battle.won is None else "loss")
        turns = agent.turn_log_for(battle_tag)
        game_id = logger.log_game(
            team_version=team_version,
            result=result,
            turn_count=battle.turn,
            source="local_scripted",
            game_format=REG_MB_FORMAT,
            turns=turns,
        )
        print(f"logged {game_id}: result={result} turns={battle.turn} "
              f"snapshots={len(turns)}")

    print(f"agent record: {agent.n_won_battles}W-"
          f"{agent.n_lost_battles}L over {agent.n_finished_battles} game(s)")
    logger.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Play and log one local game (M1)")
    ap.add_argument("--team", default="config/current_team.txt")
    ap.add_argument("--db", default="data/clefabot.sqlite")
    ap.add_argument("--games", type=int, default=1)
    args = ap.parse_args()

    team_text = Path(args.team).read_text()
    asyncio.run(play_one_game(team_text, args.db, n_battles=args.games))


if __name__ == "__main__":
    main()
