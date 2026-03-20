"""Tests for the Elo rating tracker."""

import pytest

from golds.training.elo import EloTracker


def test_initial_rating():
    tracker = EloTracker(initial_elo=1500.0)
    assert tracker.get_rating("new_player") == 1500.0


def test_update_winner_gains():
    tracker = EloTracker()
    tracker.ratings["player_a"] = 1200.0
    tracker.ratings["player_b"] = 1200.0

    tracker.update(winner_id="player_a", loser_id="player_b")
    assert tracker.get_rating("player_a") > 1200.0


def test_update_loser_loses():
    tracker = EloTracker()
    tracker.ratings["player_a"] = 1200.0
    tracker.ratings["player_b"] = 1200.0

    tracker.update(winner_id="player_a", loser_id="player_b")
    assert tracker.get_rating("player_b") < 1200.0


def test_draw_balanced():
    tracker = EloTracker()
    tracker.ratings["player_a"] = 1200.0
    tracker.ratings["player_b"] = 1200.0

    tracker.record_draw("player_a", "player_b")
    # Equal players drawing should barely change ratings
    assert abs(tracker.get_rating("player_a") - 1200.0) < 1.0
    assert abs(tracker.get_rating("player_b") - 1200.0) < 1.0


def test_elo_conservation():
    tracker = EloTracker()
    tracker.ratings["player_a"] = 1200.0
    tracker.ratings["player_b"] = 1200.0

    total_before = tracker.get_rating("player_a") + tracker.get_rating("player_b")
    tracker.update(winner_id="player_a", loser_id="player_b")
    total_after = tracker.get_rating("player_a") + tracker.get_rating("player_b")

    assert total_before == pytest.approx(total_after, abs=0.01)


def test_sample_uniform():
    tracker = EloTracker()
    candidates = ["a", "b", "c"]
    result = tracker.sample_opponent(candidates, method="uniform")
    assert result in candidates


def test_sample_proportional():
    tracker = EloTracker()
    tracker.ratings["a"] = 1500.0
    tracker.ratings["b"] = 1000.0
    candidates = ["a", "b"]
    # Just verify it returns a valid candidate (sampling is stochastic)
    for _ in range(10):
        result = tracker.sample_opponent(candidates, method="proportional")
        assert result in candidates


def test_get_leaderboard():
    tracker = EloTracker()
    tracker.ratings["player_a"] = 1300.0
    tracker.ratings["player_b"] = 1500.0
    tracker.ratings["player_c"] = 1100.0

    leaderboard = tracker.get_leaderboard()
    assert leaderboard[0] == ("player_b", 1500.0)
    assert leaderboard[1] == ("player_a", 1300.0)
    assert leaderboard[2] == ("player_c", 1100.0)


def test_persistence(tmp_path):
    save_path = tmp_path / "elo.json"

    tracker1 = EloTracker(save_path=save_path)
    tracker1.ratings["player_a"] = 1200.0
    tracker1.ratings["player_b"] = 1200.0
    tracker1.update(winner_id="player_a", loser_id="player_b")

    rating_a = tracker1.get_rating("player_a")
    rating_b = tracker1.get_rating("player_b")

    # Reload from disk
    tracker2 = EloTracker(save_path=save_path)
    assert tracker2.get_rating("player_a") == pytest.approx(rating_a)
    assert tracker2.get_rating("player_b") == pytest.approx(rating_b)
    assert len(tracker2.history) == 1
