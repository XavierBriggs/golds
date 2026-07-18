"""CLI command for diagnosing a training run's local health status (R9)."""

from __future__ import annotations

import typer
from rich.console import Console

from golds.results.schema import TrainingResult

console = Console()

EPSILON = 1e-9


def is_broken(result: TrainingResult, epsilon: float = EPSILON) -> bool:
    """Return True if a result's eval reward flags the run as broken.

    Broken predicate (R9): ``best_eval_reward is None or best_eval_reward <= epsilon``.
    """
    return result.best_eval_reward is None or result.best_eval_reward <= epsilon


def diagnose(
    run: str = typer.Argument(
        ...,
        help=(
            "Run to diagnose, addressed by TrainingResult.experiment_name from the local "
            "results.json (the simplest addressable key -- no game_id/path lookup). If "
            "multiple rows share the name, the most recently started row is used."
        ),
    ),
    results_file: str = typer.Option(
        "results.json",
        "--file",
        "-f",
        help="Path to the local results.json file. No network access; no W&B dependency.",
    ),
) -> None:
    """Print a binary health verdict (HEALTHY/BROKEN) for a run's local results.json row.

    Reads only the local results store. Exit code 0 for a healthy run,
    non-zero for a broken or not-found run.
    """
    from golds.results.store import ResultStore

    store = ResultStore(results_file)
    matches = [r for r in store.get_results() if r.experiment_name == run]

    if not matches:
        console.print(f"[red]No result found for run '{run}' in {results_file}[/red]")
        raise typer.Exit(2)

    result = max(matches, key=lambda r: r.started_at)

    if is_broken(result):
        console.print(f"[red]BROKEN[/red] {run}: best_eval_reward={result.best_eval_reward!r}")
        raise typer.Exit(1)

    console.print(f"[green]HEALTHY[/green] {run}: best_eval_reward={result.best_eval_reward!r}")
