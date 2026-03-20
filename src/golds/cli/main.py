"""Main CLI interface for GOLDS."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from golds import __version__
from golds.cli.evaluate import eval_app
from golds.cli.results import results_app
from golds.cli.roms import rom_app
from golds.cli.shortcuts import go, setup, status, train_all
from golds.cli.train import train_app

console = Console()

app = typer.Typer(
    name="golds",
    help="GOLDS - Multi-Environment RL Training System",
    add_completion=False,
    no_args_is_help=True,
)

# Add subcommand groups
app.add_typer(train_app, name="train", help="Training commands")
app.add_typer(eval_app, name="eval", help="Evaluation commands")
app.add_typer(rom_app, name="rom", help="ROM management commands")
app.add_typer(results_app, name="results", help="Training results management")

# Shortcut commands
app.command("go")(go)
app.command("train-all")(train_all)
app.command("setup")(setup)
app.command("status")(status)


@app.command("list-games")
def list_games(
    platform: str | None = typer.Option(
        None, "--platform", "-p", help="Filter by platform (atari/retro)"
    ),
) -> None:
    """List all registered games."""
    from golds.environments.registry import GameRegistry

    table = Table(title="Registered Games")
    table.add_column("Game ID", style="cyan")
    table.add_column("Platform", style="green")
    table.add_column("Display Name", style="yellow")
    table.add_column("Env ID", style="dim")

    for game_id in GameRegistry.list_games(platform):
        game = GameRegistry.get(game_id)
        table.add_row(game_id, game.platform, game.display_name, game.env_id)

    console.print(table)


@app.command("info")
def info() -> None:
    """Show system information."""
    from golds.utils.device import get_device_info

    device_info = get_device_info()

    table = Table(title="GOLDS System Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Version", __version__)
    table.add_row("CUDA Available", str(device_info["cuda_available"]))
    table.add_row("GPU Count", str(device_info["device_count"]))

    if device_info.get("cuda_device_name"):
        table.add_row("GPU Name", str(device_info["cuda_device_name"]))
        table.add_row("CUDA Version", str(device_info.get("cuda_version", "N/A")))

    # Check for stable-retro
    try:
        import retro

        retro_available = True
        retro_games = len(retro.data.list_games())
    except ImportError:
        retro_available = False
        retro_games = 0

    table.add_row("Stable-Retro", "Available" if retro_available else "Not installed")
    if retro_available:
        table.add_row("Retro Games", str(retro_games))

    console.print(table)


@app.command("tensorboard")
def tensorboard(
    logdir: Path = typer.Option(Path("outputs/logs"), "--logdir", "-l", help="Log directory"),
    port: int = typer.Option(6006, "--port", "-p", help="Port number"),
) -> None:
    """Launch TensorBoard for viewing training logs."""
    import subprocess

    if not logdir.exists():
        console.print(f"[yellow]Warning: Log directory does not exist: {logdir}[/yellow]")
        console.print("Creating directory...")
        logdir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Launching TensorBoard at http://localhost:{port}[/bold]")
    console.print(f"Log directory: {logdir}")
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    try:
        subprocess.run(
            ["tensorboard", "--logdir", str(logdir), "--port", str(port)],
            check=True,
        )
    except FileNotFoundError:
        console.print("[red]TensorBoard not found. Install with: pip install tensorboard[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]TensorBoard stopped.[/yellow]")


@app.command("version")
def version() -> None:
    """Show version information."""
    console.print(f"GOLDS version {__version__}")


@app.command()
def doctor() -> None:
    """Check system dependencies and configuration."""
    from golds.utils.device import get_device_info

    console.print("[bold]GOLDS System Check[/bold]\n")

    # CUDA
    info = get_device_info()
    if info["cuda_available"]:
        console.print(
            f"[green]\u2713[/green] CUDA available: {info.get('cuda_device_name', 'unknown')}"
        )
    elif info.get("mps_available"):
        console.print(
            f"[green]\u2713[/green] MPS available: {info.get('mps_device_name', 'Apple Silicon GPU')}"
        )
    else:
        console.print("[yellow]\u2717[/yellow] No GPU available (will use CPU)")

    # stable-retro
    try:
        import retro

        games = retro.data.list_games()
        console.print(
            f"[green]\u2713[/green] stable-retro installed ({len(games)} games available)"
        )
    except ImportError:
        console.print("[red]\u2717[/red] stable-retro not installed")

    # ale-py
    try:
        import ale_py  # noqa: F401

        console.print("[green]\u2713[/green] ale-py installed")
    except ImportError:
        console.print("[red]\u2717[/red] ale-py not installed")

    # sb3-contrib
    try:
        import sb3_contrib  # noqa: F401

        console.print("[green]\u2713[/green] sb3-contrib installed (RecurrentPPO available)")
    except ImportError:
        console.print(
            "[yellow]\u2717[/yellow] sb3-contrib not installed (RecurrentPPO unavailable)"
        )

    # Disk space
    import shutil

    usage = shutil.disk_usage(".")
    free_gb = usage.free / (1024**3)
    if free_gb > 5:
        console.print(f"[green]\u2713[/green] Disk space: {free_gb:.1f} GB free")
    else:
        console.print(f"[yellow]\u2717[/yellow] Disk space: {free_gb:.1f} GB free")

    # Results file
    if Path("results.json").exists():
        from golds.results.store import ResultStore

        store = ResultStore()
        console.print(f"[green]\u2713[/green] Results store: {len(store.get_results())} results")
    else:
        console.print("[yellow]\u2717[/yellow] No results.json found")

    console.print()


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
