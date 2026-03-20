"""CLI commands for results management."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

console = Console()
results_app = typer.Typer(help="Training results management")


@results_app.command("show")
def results_show(
    game: str | None = typer.Option(None, "--game", "-g", help="Filter by game ID"),
    results_file: str = typer.Option("results.json", "--file", "-f", help="Results file path"),
) -> None:
    """Show training results."""
    from golds.results.store import ResultStore

    store = ResultStore(results_file)
    results = store.get_results(game_id=game)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title="Training Results")
    table.add_column("Game", style="cyan")
    table.add_column("Experiment", style="green")
    table.add_column("Round")
    table.add_column("Timesteps", justify="right")
    table.add_column("Best Reward", justify="right")
    table.add_column("Wall Time", justify="right")
    table.add_column("Reward Regime")

    for r in results:
        hours = r.wall_time_seconds / 3600 if r.wall_time_seconds else 0
        table.add_row(
            r.game_id,
            r.experiment_name,
            str(r.round),
            f"{r.total_timesteps_completed:,}",
            f"{r.best_eval_reward:.1f}" if r.best_eval_reward is not None else "N/A",
            f"{hours:.1f}h",
            r.reward_regime,
        )

    console.print(table)


@results_app.command("leaderboard")
def results_leaderboard(
    results_file: str = typer.Option("results.json", "--file", "-f", help="Results file path"),
) -> None:
    """Show cross-game leaderboard."""
    from golds.results.baselines import human_normalized_score
    from golds.results.store import ResultStore

    store = ResultStore(results_file)
    leaders = store.get_leaderboard()

    if not leaders:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title="GOLDS Leaderboard")
    table.add_column("Rank", justify="right")
    table.add_column("Game", style="cyan")
    table.add_column("Best Reward", justify="right")
    table.add_column("HNS", justify="right", style="green")
    table.add_column("Timesteps", justify="right")

    for i, r in enumerate(leaders, 1):
        hns = human_normalized_score(r.game_id, r.best_eval_reward or 0)
        hns_str = f"{hns:.2f}" if hns is not None else "N/A"
        table.add_row(
            str(i),
            r.game_id,
            f"{r.best_eval_reward:.1f}" if r.best_eval_reward is not None else "N/A",
            hns_str,
            f"{r.total_timesteps_completed:,}",
        )

    console.print(table)


@results_app.command("export")
def results_export(
    output: str = typer.Option("results_export.csv", "--output", "-o", help="Output file path"),
    format: str = typer.Option("csv", "--format", help="Export format (csv or json)"),
    results_file: str = typer.Option("results.json", "--file", "-f", help="Results file path"),
) -> None:
    """Export results to CSV or JSON."""
    import json

    from golds.results.store import ResultStore

    store = ResultStore(results_file)
    results = store.get_results()

    if not results:
        console.print("[yellow]No results to export.[/yellow]")
        return

    if format == "csv":
        import csv

        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].model_dump().keys()))
            writer.writeheader()
            for r in results:
                writer.writerow({k: str(v) for k, v in r.model_dump().items()})
    else:
        with open(output, "w") as f:
            json.dump([r.model_dump(mode="json") for r in results], f, indent=2, default=str)

    console.print(f"[green]Exported {len(results)} results to {output}[/green]")
