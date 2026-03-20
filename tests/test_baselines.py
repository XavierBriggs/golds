"""Tests for the baselines module."""

import pytest

from golds.results.baselines import BASELINES, human_normalized_score


def test_baselines_exist():
    expected_games = [
        "space_invaders",
        "breakout",
        "pong",
        "qbert",
        "seaquest",
        "asteroids",
        "ms_pacman",
        "montezuma_revenge",
        "enduro",
        "frostbite",
    ]
    for game in expected_games:
        assert game in BASELINES, f"Missing baseline for {game}"


def test_human_normalized_score_pong():
    # Pong: human=14.6, random=-20.7
    # HNS = (agent - random) / (human - random)
    # For agent_score = 14.6 (human level): HNS = (14.6 - (-20.7)) / (14.6 - (-20.7)) = 1.0
    score = human_normalized_score("pong", 14.6)
    assert score == pytest.approx(1.0)

    # For agent_score = -20.7 (random level): HNS = 0.0
    score = human_normalized_score("pong", -20.7)
    assert score == pytest.approx(0.0)


def test_human_normalized_score_unknown_game():
    result = human_normalized_score("nonexistent_game_xyz", 100.0)
    assert result is None


def test_human_scores_positive():
    """All human scores should be strictly greater than random scores."""
    for game_id, baseline in BASELINES.items():
        assert baseline.human > baseline.random, (
            f"{game_id}: human ({baseline.human}) should be > random ({baseline.random})"
        )
