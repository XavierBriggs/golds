"""Tests for `golds diagnose` (R9): the binary broken/healthy verdict."""

from __future__ import annotations

import typer
from typer.testing import CliRunner

from golds.cli.diagnose import diagnose, is_broken
from golds.results.schema import TrainingResult
from golds.results.store import ResultStore

runner = CliRunner()


def _make_result(**overrides) -> TrainingResult:
    defaults = dict(
        game_id="breakout",
        experiment_name="run_a",
        config_hash="abc123",
        total_timesteps_completed=1_000_000,
        total_timesteps_target=10_000_000,
        wall_time_seconds=3600.0,
        best_eval_reward=100.0,
    )
    defaults.update(overrides)
    return TrainingResult(**defaults)


def _diagnose_app() -> typer.Typer:
    app = typer.Typer()
    app.command()(diagnose)
    return app


# --- predicate unit tests -------------------------------------------------


def test_is_broken_true_when_reward_is_none():
    result = _make_result(best_eval_reward=None)
    assert is_broken(result) is True


def test_is_broken_true_when_reward_at_or_below_epsilon():
    result = _make_result(best_eval_reward=1e-10)
    assert is_broken(result) is True


def test_is_broken_false_when_healthy():
    result = _make_result(best_eval_reward=50.0)
    assert is_broken(result) is False


# --- CLI end-to-end tests --------------------------------------------------


def test_diagnose_cli_healthy_run_exits_zero(tmp_path):
    results_file = tmp_path / "results.json"
    store = ResultStore(path=results_file)
    store.add_result(_make_result(experiment_name="healthy_run", best_eval_reward=50.0))

    app = _diagnose_app()
    result = runner.invoke(app, ["healthy_run", "--file", str(results_file)])

    assert result.exit_code == 0
    assert "HEALTHY" in result.stdout


def test_diagnose_cli_broken_run_exits_nonzero(tmp_path):
    results_file = tmp_path / "results.json"
    store = ResultStore(path=results_file)
    store.add_result(_make_result(experiment_name="broken_run", best_eval_reward=None))

    app = _diagnose_app()
    result = runner.invoke(app, ["broken_run", "--file", str(results_file)])

    assert result.exit_code != 0
    assert "BROKEN" in result.stdout


def test_diagnose_cli_unknown_run_exits_nonzero(tmp_path):
    results_file = tmp_path / "results.json"
    ResultStore(path=results_file)  # creates an empty store/file

    app = _diagnose_app()
    result = runner.invoke(app, ["does_not_exist", "--file", str(results_file)])

    assert result.exit_code != 0
