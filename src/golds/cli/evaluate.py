"""Evaluation CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

eval_app = typer.Typer(help="Evaluation commands")
console = Console()


@eval_app.command("model")
def eval_model(
    model: Path = typer.Argument(..., help="Path to trained model (.zip)"),
    game: str = typer.Option(..., "--game", "-g", help="Game ID"),
    episodes: int = typer.Option(10, "--episodes", "-e", help="Number of episodes"),
    deterministic: bool = typer.Option(
        True, "--deterministic/--stochastic", help="Use deterministic policy"
    ),
) -> None:
    """Evaluate a trained model."""
    from golds.evaluation.evaluator import Evaluator

    if not model.exists():
        # Try adding .zip extension
        if not model.suffix:
            model = model.with_suffix(".zip")
        if not model.exists():
            console.print(f"[red]Model not found: {model}[/red]")
            raise typer.Exit(1)

    console.print(f"[bold]Evaluating model: {model}[/bold]")
    console.print(f"Game: {game}")
    console.print(f"Episodes: {episodes}")
    console.print(f"Deterministic: {deterministic}")
    console.print()

    try:
        evaluator = Evaluator(model, game)
        results = evaluator.evaluate(
            n_episodes=episodes,
            deterministic=deterministic,
            verbose=True,
        )
        console.print()
        evaluator.print_results(results)

    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        raise typer.Exit(1)


@eval_app.command("compare")
def eval_compare(
    models: list[Path] = typer.Argument(..., help="Paths to models to compare"),
    game: str = typer.Option(..., "--game", "-g", help="Game ID"),
    episodes: int = typer.Option(10, "--episodes", "-e", help="Number of episodes"),
) -> None:
    """Compare multiple trained models."""
    from rich.table import Table

    from golds.evaluation.evaluator import Evaluator

    if len(models) < 2:
        console.print("[red]Need at least 2 models to compare[/red]")
        raise typer.Exit(1)

    results = []
    for model_path in models:
        if not model_path.exists():
            console.print(f"[yellow]Skipping missing model: {model_path}[/yellow]")
            continue

        console.print(f"Evaluating: {model_path.name}...")
        evaluator = Evaluator(model_path, game)
        result = evaluator.evaluate(n_episodes=episodes, verbose=False)
        result["model"] = model_path.name
        results.append(result)

    if not results:
        console.print("[red]No models could be evaluated[/red]")
        raise typer.Exit(1)

    # Sort by mean reward
    results.sort(key=lambda x: x["mean_reward"], reverse=True)

    table = Table(title=f"Model Comparison - {game}")
    table.add_column("Rank", style="cyan")
    table.add_column("Model", style="yellow")
    table.add_column("Mean Reward", style="green")
    table.add_column("Std", style="dim")
    table.add_column("Max", style="green")

    for i, result in enumerate(results, 1):
        table.add_row(
            str(i),
            result["model"],
            f"{result['mean_reward']:.2f}",
            f"{result['std_reward']:.2f}",
            f"{result['max_reward']:.2f}",
        )

    console.print()
    console.print(table)


@eval_app.command("benchmark")
def eval_benchmark(
    model_path: str = typer.Argument(..., help="Path to model .zip file"),
    game: str = typer.Option(..., "--game", "-g", help="Game ID"),
    episodes: int = typer.Option(100, "--episodes", "-n", help="Episodes per seed"),
    seeds: str = typer.Option("42,123,456", "--seeds", help="Comma-separated seeds"),
) -> None:
    """Run standardized benchmark evaluation (multi-seed)."""
    from golds.evaluation.evaluator import Evaluator

    seed_list = [int(s.strip()) for s in seeds.split(",")]

    if not model_path.endswith(".zip"):
        model_path += ".zip"

    console.print(
        f"[cyan]Running benchmark: {episodes} episodes \u00d7 {len(seed_list)} seeds[/cyan]"
    )

    evaluator = Evaluator(model_path=model_path, game_id=game)
    results = evaluator.benchmark(n_episodes=episodes, seeds=seed_list)
    evaluator.print_results(results)
