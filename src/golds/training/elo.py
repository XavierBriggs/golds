"""Elo rating tracker for self-play opponent snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal


class EloTracker:
    """Track Elo ratings for self-play opponent snapshots.

    Used to measure relative strength of policy snapshots over time
    and optionally to weight opponent sampling.
    """

    def __init__(
        self,
        k_factor: float = 32.0,
        initial_elo: float = 1200.0,
        save_path: Path | str | None = None,
    ) -> None:
        self.k_factor = k_factor
        self.initial_elo = initial_elo
        self.save_path = Path(save_path) if save_path else None
        self.ratings: dict[str, float] = {}
        self.history: list[dict[str, Any]] = []
        if self.save_path and self.save_path.exists():
            self._load()

    def _load(self) -> None:
        """Load ratings from disk."""
        with open(self.save_path) as f:
            data = json.load(f)
        self.ratings = data.get("ratings", {})
        self.history = data.get("history", [])

    def _save(self) -> None:
        """Persist ratings to disk."""
        if self.save_path is None:
            return
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "w") as f:
            json.dump({"ratings": self.ratings, "history": self.history}, f, indent=2)

    def get_rating(self, player_id: str) -> float:
        """Get current Elo rating for a player."""
        return self.ratings.get(player_id, self.initial_elo)

    def update(self, winner_id: str, loser_id: str) -> None:
        """Update Elo ratings after a match."""
        r_w = self.get_rating(winner_id)
        r_l = self.get_rating(loser_id)

        # Expected scores
        e_w = 1 / (1 + 10 ** ((r_l - r_w) / 400))
        e_l = 1 - e_w

        # Update ratings
        self.ratings[winner_id] = r_w + self.k_factor * (1 - e_w)
        self.ratings[loser_id] = r_l + self.k_factor * (0 - e_l)

        self.history.append(
            {
                "winner": winner_id,
                "loser": loser_id,
                "winner_elo": self.ratings[winner_id],
                "loser_elo": self.ratings[loser_id],
            }
        )
        self._save()

    def record_draw(self, player1_id: str, player2_id: str) -> None:
        """Update ratings for a draw."""
        r1 = self.get_rating(player1_id)
        r2 = self.get_rating(player2_id)

        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 - e1

        self.ratings[player1_id] = r1 + self.k_factor * (0.5 - e1)
        self.ratings[player2_id] = r2 + self.k_factor * (0.5 - e2)
        self._save()

    def sample_opponent(
        self,
        candidates: list[str],
        method: Literal["uniform", "proportional", "pfsp"] = "uniform",
        current_player_id: str | None = None,
    ) -> str:
        """Sample an opponent from candidates.

        Args:
            candidates: List of opponent IDs to choose from.
            method: Sampling strategy.
                - "uniform": Equal probability for all candidates.
                - "proportional": Weight by Elo (higher rated = more likely).
                - "pfsp": Prioritized Fictitious Self-Play (weight by win probability).
            current_player_id: ID of the current player (required for pfsp).

        Returns:
            Selected opponent ID.
        """
        import random

        if not candidates:
            raise ValueError("No candidates to sample from")

        if len(candidates) == 1 or method == "uniform":
            return random.choice(candidates)

        if method == "proportional":
            ratings = [self.get_rating(c) for c in candidates]
            min_r = min(ratings)
            weights = [r - min_r + 1 for r in ratings]  # shift to positive
            return random.choices(candidates, weights=weights, k=1)[0]

        if method == "pfsp":
            if current_player_id is None:
                return random.choice(candidates)
            r_current = self.get_rating(current_player_id)
            # Weight by win probability against current player
            weights = []
            for c in candidates:
                r_c = self.get_rating(c)
                win_prob = 1 / (1 + 10 ** ((r_current - r_c) / 400))
                # PFSP prefers opponents we're likely to beat slightly more often
                # f(x) = x * (1 - x) peaks at 0.5, weighting balanced matchups
                weights.append(win_prob * (1 - win_prob) + 1e-6)
            return random.choices(candidates, weights=weights, k=1)[0]

        return random.choice(candidates)

    def get_leaderboard(self) -> list[tuple[str, float]]:
        """Get sorted leaderboard (highest Elo first)."""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
