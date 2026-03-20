"""Tests for results storage and schemas."""

from datetime import datetime, timedelta

from golds.results.schema import EvalResult, TrainingResult
from golds.results.store import ResultStore


def _make_eval_result(**overrides) -> EvalResult:
    defaults = dict(
        mean_reward=100.0,
        std_reward=10.0,
        min_reward=50.0,
        max_reward=150.0,
        mean_length=500.0,
        std_length=50.0,
        n_episodes=10,
    )
    defaults.update(overrides)
    return EvalResult(**defaults)


def _make_training_result(**overrides) -> TrainingResult:
    defaults = dict(
        game_id="breakout",
        experiment_name="test_run",
        config_hash="abc123",
        total_timesteps_completed=1_000_000,
        total_timesteps_target=10_000_000,
        wall_time_seconds=3600.0,
        best_eval_reward=100.0,
        final_eval_reward=95.0,
    )
    defaults.update(overrides)
    return TrainingResult(**defaults)


def test_create_eval_result():
    result = _make_eval_result()
    assert result.mean_reward == 100.0
    assert result.n_episodes == 10
    assert result.deterministic is True


def test_create_training_result():
    result = _make_training_result()
    assert result.game_id == "breakout"
    assert result.round == 1
    assert result.exit_code == 0
    assert result.reward_regime == "clipped"


def test_store_add_and_get(tmp_path):
    store = ResultStore(path=tmp_path / "results.json")
    result = _make_training_result()
    store.add_result(result)

    results = store.get_results()
    assert len(results) == 1
    assert results[0].game_id == "breakout"


def test_store_filter_by_game(tmp_path):
    store = ResultStore(path=tmp_path / "results.json")
    store.add_result(_make_training_result(game_id="breakout"))
    store.add_result(_make_training_result(game_id="pong"))
    store.add_result(_make_training_result(game_id="breakout"))

    breakout_results = store.get_results(game_id="breakout")
    assert len(breakout_results) == 2

    pong_results = store.get_results(game_id="pong")
    assert len(pong_results) == 1


def test_store_get_latest(tmp_path):
    store = ResultStore(path=tmp_path / "results.json")
    now = datetime.now()
    store.add_result(
        _make_training_result(
            game_id="breakout",
            started_at=now - timedelta(hours=2),
            experiment_name="old_run",
        )
    )
    store.add_result(
        _make_training_result(
            game_id="breakout",
            started_at=now,
            experiment_name="new_run",
        )
    )

    latest = store.get_latest("breakout")
    assert latest is not None
    assert latest.experiment_name == "new_run"


def test_store_get_best(tmp_path):
    store = ResultStore(path=tmp_path / "results.json")
    store.add_result(_make_training_result(game_id="breakout", best_eval_reward=50.0))
    store.add_result(_make_training_result(game_id="breakout", best_eval_reward=200.0))
    store.add_result(_make_training_result(game_id="breakout", best_eval_reward=100.0))

    best = store.get_best("breakout")
    assert best is not None
    assert best.best_eval_reward == 200.0


def test_store_leaderboard(tmp_path):
    store = ResultStore(path=tmp_path / "results.json")
    store.add_result(_make_training_result(game_id="breakout", best_eval_reward=200.0))
    store.add_result(_make_training_result(game_id="breakout", best_eval_reward=50.0))
    store.add_result(_make_training_result(game_id="pong", best_eval_reward=300.0))

    leaderboard = store.get_leaderboard()
    assert len(leaderboard) == 2
    # Sorted by best_eval_reward descending
    assert leaderboard[0].game_id == "pong"
    assert leaderboard[1].game_id == "breakout"
    assert leaderboard[1].best_eval_reward == 200.0


def test_store_persistence(tmp_path):
    path = tmp_path / "results.json"
    store1 = ResultStore(path=path)
    store1.add_result(_make_training_result(game_id="breakout", best_eval_reward=42.0))

    # Create a new store instance pointing to the same file
    store2 = ResultStore(path=path)
    results = store2.get_results()
    assert len(results) == 1
    assert results[0].best_eval_reward == 42.0


def test_store_empty(tmp_path):
    store = ResultStore(path=tmp_path / "results.json")
    assert store.get_results() == []
    assert store.get_latest("breakout") is None
    assert store.get_best("breakout") is None
    assert store.get_leaderboard() == []
