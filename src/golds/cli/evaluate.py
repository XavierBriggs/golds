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


def eval_completion(
    config: Path = typer.Option(..., "--config", "-c", help="Path to experiment config file"),
    model: Path = typer.Option(..., "--model", "-m", help="Path to trained model (.zip)"),
    episodes: int = typer.Option(
        100, "--episodes", "-e", help="Number of eval episodes (north-star protocol: 100)"
    ),
    deterministic: bool = typer.Option(
        True, "--deterministic/--stochastic", help="Use deterministic policy (north-star: True)"
    ),
) -> None:
    """Run the north-star completion-rate eval (R11 / goal G6).

    Builds the eval env via the same factory path training uses
    (``EnvironmentFactory.create_eval_env``, so ``PlatformerRewardWrapper``
    and its ``info["level_complete"]`` signal are present), runs
    ``episodes`` deterministic episodes, and reports the fraction of
    episodes in which the agent reached the level-completion signpost.
    """
    from stable_baselines3 import PPO

    from golds.config.loader import ConfigLoader
    from golds.environments.factory import EnvironmentFactory
    from golds.evaluation.completion import evaluate_completion_rate

    if not config.exists():
        console.print(f"[red]Config file not found: {config}[/red]")
        raise typer.Exit(1)

    if not model.exists():
        if not model.suffix:
            model = model.with_suffix(".zip")
        if not model.exists():
            console.print(f"[red]Model not found: {model}[/red]")
            raise typer.Exit(1)

    loader = ConfigLoader()
    exp_config = loader.load(config)
    env_cfg = exp_config.environment

    self_play_2p = env_cfg.players == 2 and env_cfg.opponent == "self_play"

    console.print(f"[bold]Completion-rate eval: {model}[/bold]")
    console.print(f"Config: {config}")
    console.print(f"Episodes: {episodes}  Deterministic: {deterministic}")
    console.print()

    eval_env = None
    try:
        eval_env = EnvironmentFactory.create_eval_env(
            game_id=env_cfg.game_id,
            frame_stack=env_cfg.frame_stack,
            seed=exp_config.training.seed,
            state=env_cfg.state,
            reward_regime=env_cfg.reward_regime,
            players=env_cfg.players,
            opponent_mode="noop" if self_play_2p else env_cfg.opponent,
            opponent_model_path=None if self_play_2p else env_cfg.opponent_model_path,
            opponent_snapshot_dir=None,
            wrapper_kwargs={
                "terminal_on_life_loss": False,
                "clip_reward": (
                    env_cfg.eval_clip_reward
                    if env_cfg.eval_clip_reward is not None
                    else env_cfg.clip_reward
                ),
                "action_set": env_cfg.action_set,
                "sticky_action_prob": 0.0,
            },
        )
        model_obj = PPO.load(model)

        results = evaluate_completion_rate(
            model_obj,
            eval_env,
            n_episodes=episodes,
            deterministic=deterministic,
        )
    except Exception as e:
        console.print(f"[red]Completion eval failed: {e}[/red]")
        raise typer.Exit(1)
    finally:
        if eval_env is not None:
            eval_env.close()

    n = results["n_episodes"]
    n_completed = results["n_completed"]
    rate = results["completion_rate"]
    console.print(
        f"[bold green]GHZ Act 1 completion: {n_completed}/{n} = {rate * 100:.1f}%[/bold green]"
    )
    console.print(f"Mean reward: {results['mean_reward']:.2f}")
    if results.get("mean_max_x") is not None:
        console.print(f"Mean max x: {results['mean_max_x']:.1f}")

    goal_met = rate >= 0.8
    status = "[green]MET[/green]" if goal_met else "[yellow]NOT MET[/yellow]"
    console.print(f"Goal G6 (>= 80%): {status}")


eval_app.command("completion")(eval_completion)


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
