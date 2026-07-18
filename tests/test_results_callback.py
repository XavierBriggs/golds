"""Tests for ResultsCallback: timestamp fix (T1), git provenance (T2), and the
R7 eval-metric population (eval_100ep, human_score, human_normalized_score,
published_ppo_score)."""

from __future__ import annotations

import subprocess
import time

import pytest

from golds.results.schema import EvalResult
from golds.results.store import ResultStore
from golds.training.callbacks import ResultsCallback


def _make_callback(tmp_path, **overrides):
    defaults = dict(
        game_id="breakout",
        experiment_name="ts_test",
        config_hash="abc123",
        round=1,
        total_timesteps_target=1000,
        device="cpu",
        n_envs=1,
        output_dir=tmp_path,
        results_path=tmp_path / "results.json",
    )
    defaults.update(overrides)
    return ResultsCallback(**defaults)


def test_started_at_before_completed_at_with_distinct_values(tmp_path):
    """T1: started_at must be captured at training start, not at the end.

    Before the fix, both started_at and completed_at were set to
    datetime.now() inside _on_training_end, making them identical (or at
    best a coin-flip on ordering). After the fix, started_at is captured
    in _on_training_start, so a real gap between the two calls must show
    up as started_at < completed_at.
    """
    cb = _make_callback(tmp_path)

    cb._on_training_start()
    time.sleep(0.05)
    cb._on_training_end()

    store = ResultStore(tmp_path / "results.json")
    result = store.get_latest("breakout")

    assert result is not None
    assert result.started_at < result.completed_at
    assert result.started_at != result.completed_at
    assert (result.completed_at - result.started_at).total_seconds() >= 0.04


def test_git_provenance_populated_on_result(tmp_path):
    """T2: the built TrainingResult carries git_sha/git_dirty from the repo.

    Cross-checked against `git rev-parse HEAD` / `git status --porcelain`
    run directly, so the expectation is hand-computed rather than assumed.
    """
    expected_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    expected_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )

    cb = _make_callback(tmp_path, experiment_name="git_test")
    cb._on_training_start()
    cb._on_training_end()

    store = ResultStore(tmp_path / "results.json")
    result = store.get_latest("breakout")

    assert result is not None
    assert result.git_sha == expected_sha
    assert len(result.git_sha) == 40
    assert result.git_dirty == expected_dirty


def _synthetic_eval_result(mean_reward: float, n_episodes: int = 100) -> EvalResult:
    return EvalResult(
        mean_reward=mean_reward,
        std_reward=2.0,
        min_reward=mean_reward - 5.0,
        max_reward=mean_reward + 5.0,
        median_reward=mean_reward,
        mean_length=500.0,
        std_length=25.0,
        n_episodes=n_episodes,
        deterministic=True,
    )


def test_eval_100ep_populated_and_human_normalized_score_computed(tmp_path, monkeypatch):
    """R7: a synthetic final eval populates eval_100ep, human_score,
    published_ppo_score, and a correctly-computed human_normalized_score.

    Uses breakout's published baselines (human=30.5, random=1.7) and a
    synthetic agent eval mean of 16.1, so HNS = (16.1-1.7)/(30.5-1.7) = 0.5.
    """
    synthetic = _synthetic_eval_result(mean_reward=16.1)
    monkeypatch.setattr(
        ResultsCallback, "_run_final_eval", lambda self: synthetic
    )

    cb = _make_callback(tmp_path, game_id="breakout", experiment_name="eval_test")
    cb._on_training_start()
    cb._on_training_end()

    store = ResultStore(tmp_path / "results.json")
    result = store.get_latest("breakout")

    assert result is not None
    assert result.eval_100ep is not None
    assert result.eval_100ep.mean_reward == pytest.approx(16.1)
    assert result.eval_100ep.n_episodes == 100
    assert result.human_score == pytest.approx(30.5)
    assert result.published_ppo_score == pytest.approx(274.8)
    assert result.human_normalized_score == pytest.approx(0.5)


def test_missing_baseline_entry_no_crash(tmp_path, monkeypatch):
    """A game with no baselines.py entry => human_score/published_ppo_score/
    human_normalized_score are all None, and building the result never raises."""
    synthetic = _synthetic_eval_result(mean_reward=42.0)
    monkeypatch.setattr(
        ResultsCallback, "_run_final_eval", lambda self: synthetic
    )

    cb = _make_callback(
        tmp_path, game_id="nonexistent_game_xyz", experiment_name="no_baseline_test"
    )
    cb._on_training_start()
    cb._on_training_end()

    store = ResultStore(tmp_path / "results.json")
    result = store.get_latest("nonexistent_game_xyz")

    assert result is not None
    assert result.eval_100ep is not None
    assert result.eval_100ep.mean_reward == pytest.approx(42.0)
    assert result.human_score is None
    assert result.published_ppo_score is None
    assert result.human_normalized_score is None


def test_backward_compat_no_eval_env_still_constructs_with_none_fields(tmp_path):
    """No eval_env wired up (e.g. eval_freq=0, or older call sites): eval_100ep
    and human_normalized_score stay None, but the result still constructs and
    saves cleanly (no crash), and baseline lookups that don't need an agent
    score still populate (human_score/published_ppo_score for a known game)."""
    cb = _make_callback(tmp_path, game_id="breakout", experiment_name="no_eval_test")
    cb._on_training_start()
    cb._on_training_end()

    store = ResultStore(tmp_path / "results.json")
    result = store.get_latest("breakout")

    assert result is not None
    assert result.eval_100ep is None
    assert result.human_normalized_score is None
    # Baseline scores for a known game are independent of whether an eval ran.
    assert result.human_score == pytest.approx(30.5)
    assert result.published_ppo_score == pytest.approx(274.8)
