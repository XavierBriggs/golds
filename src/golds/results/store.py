"""JSON-backed results storage."""

from __future__ import annotations

import json
from pathlib import Path

from golds.results.schema import TrainingResult


class ResultStore:
    """Stores training results in a JSON file."""

    def __init__(self, path: Path | str = "results.json") -> None:
        self.path = Path(path)
        self._results: list[TrainingResult] = []
        self._load()

    def _load(self) -> None:
        """Load results from disk."""
        if self.path.exists():
            with open(self.path) as f:
                data = json.load(f)
            self._results = [TrainingResult(**r) for r in data]
        else:
            self._results = []

    def _save(self) -> None:
        """Persist results to disk."""
        data = [r.model_dump(mode="json") for r in self._results]
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def add_result(self, result: TrainingResult) -> None:
        """Add a training result and save."""
        self._results.append(result)
        self._save()

    def get_results(self, game_id: str | None = None) -> list[TrainingResult]:
        """Get all results, optionally filtered by game."""
        if game_id is None:
            return list(self._results)
        return [r for r in self._results if r.game_id == game_id]

    def get_latest(self, game_id: str) -> TrainingResult | None:
        """Get the most recent result for a game."""
        game_results = self.get_results(game_id)
        if not game_results:
            return None
        return max(game_results, key=lambda r: r.started_at)

    def get_best(self, game_id: str) -> TrainingResult | None:
        """Get the best result for a game (by best_eval_reward)."""
        game_results = [r for r in self.get_results(game_id) if r.best_eval_reward is not None]
        if not game_results:
            return None
        return max(game_results, key=lambda r: r.best_eval_reward)

    def get_leaderboard(self) -> list[TrainingResult]:
        """Get best result per game, sorted by human_normalized_score or best_eval_reward."""
        best_per_game: dict[str, TrainingResult] = {}
        for r in self._results:
            game = r.game_id
            if game not in best_per_game:
                best_per_game[game] = r
            elif (r.best_eval_reward or 0) > (best_per_game[game].best_eval_reward or 0):
                best_per_game[game] = r
        return sorted(best_per_game.values(), key=lambda r: r.best_eval_reward or 0, reverse=True)
