"""poke-env players for Phase 1 local play (spec §5).

`RecordingPlayer` plays a legal move each turn while snapshotting per-turn state
into a log we hand to the data layer. For M1 the policy is random-legal (the
real policy arrives with the PPO agent at M5); the point of M1 is proving the
full spine — server -> battle -> structured log — works end to end.
"""

from __future__ import annotations

from poke_env.player import Player

# Format id confirmed against the local server's format list (plan §0).
REG_MB_FORMAT = "gen9championsvgc2026regmb"


def _active_names(mons) -> str:
    names = [m.species for m in mons if m is not None]
    return "+".join(names) if names else ""


class RecordingPlayer(Player):
    """Random-legal doubles player that records a per-turn snapshot."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # battle_tag -> {turn_number -> snapshot dict}
        self._turn_logs: dict[str, dict[int, dict]] = {}

    def choose_move(self, battle):
        log = self._turn_logs.setdefault(battle.battle_tag, {})
        if battle.turn not in log:
            log[battle.turn] = {
                "turn_number": battle.turn,
                "active_self": _active_names(battle.active_pokemon),
                "active_opponent": _active_names(battle.opponent_active_pokemon),
                "action_taken": "move",
            }
        return self.choose_random_move(battle)

    def turn_log_for(self, battle_tag: str) -> list[dict]:
        log = self._turn_logs.get(battle_tag, {})
        return [log[k] for k in sorted(log)]

    def last_battle(self):
        """Return the most recently finished battle object."""
        if not self.battles:
            return None
        return list(self.battles.values())[-1]
