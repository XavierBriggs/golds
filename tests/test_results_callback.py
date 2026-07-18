"""Tests for ResultsCallback: timestamp fix (T1) and git provenance (T2)."""

from __future__ import annotations

import subprocess
import time

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
