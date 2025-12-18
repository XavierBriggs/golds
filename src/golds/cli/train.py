"""Training CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

train_app = typer.Typer(help="Training commands")
console = Console()


@train_app.command("run")
def train_run(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to experiment config file"
    ),
    output: Path = typer.Option(
        Path("outputs"), "--output", "-o", help="Output directory"
    ),
    resume: Optional[Path] = typer.Option(
        None, "--resume", "-r", help="Resume from checkpoint"
    ),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Override random seed"),
    device: str = typer.Option("auto", "--device", "-d", help="Device: auto, cuda, cpu"),
) -> None:
    """Run training with a configuration file."""
    from golds.config.loader import ConfigLoader
    from golds.training.trainer import Trainer

    if not config.exists():
        console.print(f"[red]Config file not found: {config}[/red]")
        raise typer.Exit(1)

    # Load configuration
    loader = ConfigLoader()
    exp_config = loader.load(config)

    # Apply CLI overrides
    if seed is not None:
        exp_config.training.seed = seed
    if device != "auto":
        exp_config.training.device = device

    # Create output directory for this experiment
    exp_output = output / exp_config.name
    exp_output.mkdir(parents=True, exist_ok=True)

    # Run training
    trainer = Trainer(exp_config, exp_output, resume)
    trainer.train()


@train_app.command("game")
def train_game(
    game: str = typer.Argument(..., help="Game ID (e.g., space_invaders)"),
    timesteps: int = typer.Option(
        10_000_000, "--timesteps", "-t", help="Total timesteps"
    ),
    envs: int = typer.Option(8, "--envs", "-n", help="Number of parallel environments"),
    output: Path = typer.Option(
        Path("outputs"), "--output", "-o", help="Output directory"
    ),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    device: str = typer.Option("auto", "--device", "-d", help="Device: auto, cuda, cpu"),
) -> None:
    """Quick training for a specific game with default settings."""
    from golds.config.loader import ConfigLoader
    from golds.environments.registry import GameRegistry
    from golds.training.trainer import Trainer

    # Verify game exists
    try:
        game_info = GameRegistry.get(game)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        console.print("\nUse 'golds list-games' to see available games.")
        raise typer.Exit(1)

    console.print(f"[bold]Training {game_info.display_name}[/bold]")
    console.print(f"Platform: {game_info.platform}")

    # For retro games, check if ROM is available
    if game_info.platform == "retro":
        try:
            import retro
            if game_info.env_id not in retro.data.list_games():
                console.print(
                    f"[red]ROM not found for {game_info.env_id}[/red]"
                )
                console.print(
                    "\nTo import ROMs, place them in the 'roms' directory and run:"
                )
                console.print("  golds rom import ./roms")
                raise typer.Exit(1)
        except ImportError:
            console.print("[red]stable-retro is not installed[/red]")
            console.print("Install with: pip install stable-retro")
            raise typer.Exit(1)

    # Create configuration
    loader = ConfigLoader()
    exp_config = loader.create_from_args(
        game_id=game,
        platform=game_info.platform,
        n_envs=envs,
        total_timesteps=timesteps,
        seed=seed,
        device=device,
    )

    # Create output directory
    exp_output = output / game
    exp_output.mkdir(parents=True, exist_ok=True)

    # Run training
    trainer = Trainer(exp_config, exp_output)
    trainer.train()


@train_app.command("list-configs")
def list_configs(
    config_dir: Path = typer.Option(
        Path("configs/games"), "--dir", "-d", help="Config directory"
    ),
) -> None:
    """List available configuration files."""
    from rich.table import Table

    if not config_dir.exists():
        console.print(f"[yellow]Config directory does not exist: {config_dir}[/yellow]")
        return

    configs = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))

    if not configs:
        console.print("[yellow]No configuration files found.[/yellow]")
        return

    table = Table(title="Available Configurations")
    table.add_column("File", style="cyan")
    table.add_column("Path", style="dim")

    for cfg in sorted(configs):
        table.add_row(cfg.stem, str(cfg))

    console.print(table)
