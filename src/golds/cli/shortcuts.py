"""Shortcut commands for streamlined training workflow."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _auto_import_roms() -> None:
    """Import ROMs from the ``roms/`` directory if present.

    This compensates for ``uv sync`` wiping site-packages (and therefore
    previously-imported ROMs). The call is fast and idempotent.
    """
    roms_dir = Path("roms")
    if not roms_dir.exists():
        return
    has_roms = (
        any(roms_dir.glob("*.nes"))
        or any(roms_dir.glob("*.gen"))
        or any(roms_dir.glob("*.gb"))
        or any(roms_dir.glob("*.sfc"))
        or any(roms_dir.glob("*.smc"))
    )
    if not has_roms:
        return
    import subprocess

    try:
        subprocess.run(
            ["python", "-m", "retro.import", str(roms_dir)],
            capture_output=True,
            timeout=30,
        )
    except Exception:
        pass  # Best-effort; setup command handles diagnostics


def find_game_config(game_id: str, config_dir: Path | None = None) -> Path | None:
    """Find the YAML config file for a game.

    Args:
        game_id: Game identifier (e.g., 'pong').
        config_dir: Config directory. Defaults to 'configs'.

    Returns:
        Path to config file, or None if not found.
    """
    config_dir = config_dir or Path("configs")
    config_path = config_dir / "games" / f"{game_id}.yaml"
    if config_path.exists():
        return config_path
    return None


def make_output_dir(game_id: str, base: Path | None = None) -> Path:
    """Create a timestamped output directory for a training run.

    Args:
        game_id: Game identifier.
        base: Base output directory. Defaults to 'outputs'.

    Returns:
        Path to the output directory (not yet created on disk).
    """
    base = base or Path("outputs")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    return base / f"{game_id}_{timestamp}"


def _find_latest_checkpoint(game_id: str) -> Path | None:
    """Find the latest checkpoint for a game across all output directories.

    Supports both the unified structure (``outputs/{game}_{timestamp}/best/``)
    and the legacy nested structure (``outputs/{game}_{timestamp}/{game}/best/``).
    """
    outputs = Path("outputs")
    if not outputs.exists():
        return None

    candidates: list[tuple[float, Path]] = []
    for d in outputs.iterdir():
        if not d.is_dir() or not d.name.startswith(game_id):
            continue

        # Unified structure: outputs/{game}_{timestamp}/best/best_model.zip
        best = d / "best" / "best_model.zip"
        if best.exists():
            candidates.append((best.stat().st_mtime, best))
            continue
        # Legacy nested: outputs/{game}_{timestamp}/{game}/best/best_model.zip
        best_legacy = d / game_id / "best" / "best_model.zip"
        if best_legacy.exists():
            candidates.append((best_legacy.stat().st_mtime, best_legacy))
            continue

        # Final model (unified then legacy)
        final = d / "models" / "final_model.zip"
        if final.exists():
            candidates.append((final.stat().st_mtime, final))
            continue
        final_legacy = d / game_id / "models" / "final_model.zip"
        if final_legacy.exists():
            candidates.append((final_legacy.stat().st_mtime, final_legacy))
            continue

        # Latest checkpoint (unified then legacy)
        for ckpt_dir in [d / "models" / "checkpoints", d / game_id / "models" / "checkpoints"]:
            if ckpt_dir.exists():
                ckpts = sorted(ckpt_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
                if ckpts:
                    candidates.append((ckpts[-1].stat().st_mtime, ckpts[-1]))
                    break

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def go(
    game: str = typer.Argument(..., help="Game ID (e.g., pong, breakout, space_invaders)"),
    timesteps: int | None = typer.Option(
        None, "--timesteps", "-t", help="Override total timesteps"
    ),
    device: str = typer.Option("auto", "--device", "-d", help="Device: auto, cuda, mps, cpu"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Override output directory"),
    resume: bool = typer.Option(
        False, "--resume", "-r", help="Resume from latest run for this game"
    ),
) -> None:
    """Train a game with one command. Picks config automatically."""
    from golds.config.loader import ConfigLoader
    from golds.training.trainer import Trainer
    from golds.utils.device import get_device

    _auto_import_roms()

    # Find config
    config_path = find_game_config(game)
    if config_path is None:
        from golds.environments.registry import GameRegistry

        if not GameRegistry.is_registered(game):
            console.print(f"[red]Unknown game: {game}[/red]")
            console.print("Run [cyan]golds list-games[/cyan] to see available games.")
            raise typer.Exit(1)

        game_info = GameRegistry.get(game)
        console.print(
            f"[yellow]No config file for {game}, using defaults for {game_info.platform}[/yellow]"
        )
        loader = ConfigLoader()
        config = loader.create_from_args(
            game_id=game,
            platform=game_info.platform,
            total_timesteps=timesteps or 10_000_000,
            device=device,
        )
    else:
        loader = ConfigLoader()
        config = loader.load(config_path)

    # Override timesteps if specified
    if timesteps is not None:
        config.training.total_timesteps = timesteps

    # Override device
    if device != "auto":
        config.training.device = device

    # Determine output directory
    if output is not None:
        out_dir = output
    else:
        out_dir = make_output_dir(game)

    # Check for resume
    resume_path = None
    if resume:
        resume_path = _find_latest_checkpoint(game)
        if resume_path:
            console.print(f"[yellow]Resuming from: {resume_path}[/yellow]")
        else:
            console.print("[yellow]No previous run found, starting fresh.[/yellow]")

    # Print summary
    detected_device = get_device(config.training.device)
    console.print(
        Panel(
            f"[bold]{game}[/bold]\n"
            f"Config: {config_path or 'defaults'}\n"
            f"Output: {out_dir}\n"
            f"Device: {detected_device}\n"
            f"Timesteps: {config.training.total_timesteps:,}\n"
            f"Envs: {config.environment.n_envs}",
            title="Training",
            border_style="green",
        )
    )

    # Train
    trainer = Trainer(config, out_dir, resume_from=resume_path)
    trainer.train()

    console.print(f"\n[bold green]Done![/bold green] Results in: {out_dir}")
    console.print(
        f"Evaluate with: [cyan]golds eval model {out_dir}/{game}/best/best_model.zip --game {game}[/cyan]"
    )


def train_all(
    games: str | None = typer.Option(
        None, "--games", "-g", help="Comma-separated game list (default: all configured)"
    ),
    device: str = typer.Option("auto", "--device", "-d", help="Device: auto, cuda, mps, cpu"),
    skip_completed: bool = typer.Option(
        True, "--skip-completed/--no-skip", help="Skip games that already hit their timestep target"
    ),
) -> None:
    """Train all configured games sequentially."""
    from golds.config.loader import ConfigLoader
    from golds.results.store import ResultStore
    from golds.training.trainer import Trainer

    _auto_import_roms()

    # Determine game list
    configs_dir = Path("configs/games")
    if games:
        game_list = [g.strip() for g in games.split(",")]
    else:
        if not configs_dir.exists():
            console.print("[red]No configs/games/ directory found.[/red]")
            raise typer.Exit(1)
        game_list = sorted(p.stem for p in configs_dir.glob("*.yaml"))

    if not game_list:
        console.print("[yellow]No games to train.[/yellow]")
        return

    console.print(f"[bold]Training {len(game_list)} games:[/bold] {', '.join(game_list)}\n")

    store = ResultStore()
    loader = ConfigLoader()
    completed = 0
    failed = 0

    for i, game_id in enumerate(game_list, 1):
        console.print(f"\n[bold cyan]({i}/{len(game_list)}) {game_id}[/bold cyan]")

        config_path = find_game_config(game_id)
        if config_path is None:
            console.print("  [yellow]No config file, skipping.[/yellow]")
            continue

        config = loader.load(config_path)
        if device != "auto":
            config.training.device = device

        # Check if already completed
        if skip_completed:
            latest = store.get_latest(game_id)
            if latest and latest.total_timesteps_completed >= config.training.total_timesteps:
                console.print(
                    f"  [green]Already complete ({latest.total_timesteps_completed:,} steps). Skipping.[/green]"
                )
                continue

        out_dir = make_output_dir(game_id)

        # Check for resume
        resume_path = _find_latest_checkpoint(game_id)
        if resume_path:
            console.print(f"  [yellow]Resuming from: {resume_path}[/yellow]")

        try:
            trainer = Trainer(config, out_dir, resume_from=resume_path)
            trainer.train()
            completed += 1
            console.print("  [green]Complete![/green]")
        except Exception as e:
            failed += 1
            console.print(f"  [red]Failed: {e}[/red]")
            continue

    skipped = len(game_list) - completed - failed
    console.print(f"\n[bold]Done![/bold] {completed} completed, {failed} failed, {skipped} skipped")


def setup() -> None:
    """Set up GOLDS: install deps, import ROMs, verify pipeline."""
    import shutil
    import subprocess

    console.print(Panel("[bold]GOLDS Setup[/bold]", border_style="cyan"))
    console.print()

    issues: list[str] = []

    # 1. Install dependencies
    console.print("[cyan]1. Installing dependencies...[/cyan]")
    try:
        result = subprocess.run(
            ["uv", "sync", "--dev"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            console.print("[green]   Dependencies installed[/green]")
        else:
            console.print(f"[yellow]   uv sync had issues: {result.stderr.strip()[:200]}[/yellow]")
    except FileNotFoundError:
        console.print("[yellow]   uv not found, skipping dependency install[/yellow]")
    except Exception as e:
        console.print(f"[yellow]   Dependency install skipped: {e}[/yellow]")

    # 2. Check device
    from golds.utils.device import get_device, get_device_info

    info = get_device_info()
    device = get_device()
    if device == "cuda":
        console.print(f"[green]2. GPU:[/green] CUDA - {info.get('cuda_device_name', 'unknown')}")
    elif device == "mps":
        console.print("[green]2. GPU:[/green] Apple Silicon (MPS)")
    else:
        console.print("[yellow]2. GPU:[/yellow] None detected, will use CPU (slow)")
        issues.append("No GPU - training will be slow")

    # 3. Check ale-py (Atari)
    try:
        import ale_py  # noqa: F401

        console.print("[green]3. Atari (ale-py):[/green] Installed")
    except ImportError:
        console.print("[red]3. Atari (ale-py):[/red] Not installed")
        issues.append("Install ale-py: uv pip install 'gymnasium[atari]' ale-py")

    # 4. Check stable-retro + auto-import ROMs
    retro_available = False
    try:
        import retro

        retro_available = True
        game_count = len(retro.data.list_games())
        console.print(f"[green]4. Retro (stable-retro):[/green] Installed - {game_count} games")

        # Auto-import ROMs from ./roms if they exist
        roms_dir = Path("roms")
        rom_files = []
        if roms_dir.exists():
            rom_files = (
                list(roms_dir.glob("*.nes"))
                + list(roms_dir.glob("*.gen"))
                + list(roms_dir.glob("*.gb"))
                + list(roms_dir.glob("*.sfc"))
                + list(roms_dir.glob("*.smc"))
            )

        if rom_files:
            console.print(
                f"   [cyan]Found {len(rom_files)} ROM files in roms/, importing...[/cyan]"
            )
            try:
                result = subprocess.run(
                    ["python", "-m", "retro.import", str(roms_dir)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                # Re-check game count after import
                new_count = len(retro.data.list_games())
                imported = new_count - game_count
                if imported > 0:
                    console.print(
                        f"   [green]Imported {imported} new games ({new_count} total)[/green]"
                    )
                else:
                    console.print(
                        "   [dim]No new games imported (ROMs may already be imported or SHAs don't match)[/dim]"
                    )
            except Exception as e:
                console.print(f"   [yellow]ROM import failed: {e}[/yellow]")
        elif game_count == 0:
            console.print(
                "   [yellow]No ROMs found. Place ROM files in roms/ and re-run setup.[/yellow]"
            )
    except ImportError:
        console.print(
            "[yellow]4. Retro (stable-retro):[/yellow] Not installed"
            " (NES/SNES/Genesis games unavailable)"
        )
        issues.append("For retro games: uv pip install stable-retro")

    # 5. Check sb3-contrib
    try:
        import sb3_contrib  # noqa: F401

        console.print("[green]5. sb3-contrib:[/green] Installed (RecurrentPPO available)")
    except ImportError:
        console.print("[yellow]5. sb3-contrib:[/yellow] Not installed (optional)")

    # 6. Check configs
    configs_dir = Path("configs/games")
    if configs_dir.exists():
        config_count = len(list(configs_dir.glob("*.yaml")))
        console.print(f"[green]6. Configs:[/green] {config_count} game configs found")
    else:
        console.print("[red]6. Configs:[/red] No configs/games/ directory")
        issues.append("Missing configs directory")

    # 7. Disk space
    usage = shutil.disk_usage(".")
    free_gb = usage.free / (1024**3)
    if free_gb > 5:
        console.print(f"[green]7. Disk:[/green] {free_gb:.1f} GB free")
    else:
        console.print(f"[yellow]7. Disk:[/yellow] {free_gb:.1f} GB free (recommend >5 GB)")
        issues.append("Low disk space")

    # 8. Quick pipeline test
    console.print("\n[cyan]8. Running pipeline test (Pong preflight)...[/cyan]")
    try:
        from golds.environments.factory import EnvironmentFactory

        env = EnvironmentFactory.create(game_id="pong", n_envs=1, frame_stack=4, use_subproc=False)
        env.reset()
        env.step([env.action_space.sample()])
        env.close()
        console.print("[green]   Pipeline test: PASSED[/green]")
    except Exception as e:
        console.print(f"[red]   Pipeline test: FAILED - {e}[/red]")
        issues.append(f"Pipeline test failed: {e}")

    # 9. Telegram test
    console.print("\n[cyan]9. Testing Telegram notifications...[/cyan]")
    try:
        from golds.notifications.telegram import TelegramNotifier

        notifier = TelegramNotifier()
        if notifier.enabled:
            sent = notifier.send("GOLDS setup complete - notifications working!")
            if sent:
                console.print("[green]   Telegram: Connected[/green]")
            else:
                console.print(
                    "[yellow]   Telegram: Configured but couldn't send (network issue?)[/yellow]"
                )
        else:
            console.print("[dim]   Telegram: Not configured (set GOLDS_TELEGRAM_CHAT_ID)[/dim]")
    except Exception:
        console.print("[dim]   Telegram: Skipped[/dim]")

    # Summary
    console.print()

    # Show what's ready to train
    ready_games: list[str] = []
    atari_games = [
        "pong",
        "breakout",
        "space_invaders",
        "ms_pacman",
        "enduro",
        "frostbite",
        "montezuma_revenge",
    ]
    retro_games_map = {
        "super_mario_bros_2_japan": "SuperMarioBros2Japan-Nes-v0",
        "sonic_the_hedgehog": "SonicTheHedgehog-Genesis-v0",
        "mortal_kombat_ii": "MortalKombatII-Genesis-v0",
        "tetris": "Tetris-GameBoy-v0",
        "super_mario_bros": "SuperMarioBros-Nes-v0",
        "street_fighter_ii": "StreetFighterIISpecialChampionEdition-Genesis-v0",
        "mega_man_2": "MegaMan2-Nes-v0",
    }

    # Atari games are always ready if ale-py is installed
    try:
        import ale_py  # noqa: F401

        ready_games.extend(g for g in atari_games if (configs_dir / f"{g}.yaml").exists())
    except ImportError:
        pass

    # Retro games need ROM imported
    if retro_available:
        import retro as retro_mod

        available = set(retro_mod.data.list_games())
        for game_id, retro_name in retro_games_map.items():
            if retro_name in available and (configs_dir / f"{game_id}.yaml").exists():
                ready_games.append(game_id)

    if issues:
        console.print(
            Panel(
                "\n".join(f"  - {i}" for i in issues),
                title="Issues",
                border_style="yellow",
            )
        )

    if ready_games:
        games_str = ",".join(ready_games)
        console.print(
            Panel(
                f"[bold]{len(ready_games)} games ready to train:[/bold]\n"
                f"  {', '.join(ready_games)}\n\n"
                f"Start training:\n"
                f"  [cyan]golds go pong[/cyan]                          # Single game\n"
                f"  [cyan]golds train-all --games {games_str}[/cyan]",
                title="Ready!",
                border_style="green",
            )
        )
    else:
        console.print(
            "[yellow]No games ready to train. Fix the issues above and re-run setup.[/yellow]"
        )


def status() -> None:
    """Show training status and progress across all games."""
    from golds.results.baselines import human_normalized_score
    from golds.results.store import ResultStore
    from golds.utils.device import get_device

    store = ResultStore()
    results = store.get_results()
    device = get_device()

    # Header
    console.print(Panel(f"[bold]GOLDS Status[/bold]  |  Device: {device}", border_style="cyan"))
    console.print()

    # Available configs
    configs_dir = Path("configs/games")
    all_games: dict[str, Path] = {}
    if configs_dir.exists():
        for p in sorted(configs_dir.glob("*.yaml")):
            all_games[p.stem] = p

    # Build status per game
    trained_games: dict[str, dict] = {}
    for r in results:
        gid = r.game_id
        if gid not in trained_games or (r.best_eval_reward or 0) > (
            trained_games[gid].get("best_reward") or 0
        ):
            trained_games[gid] = {
                "best_reward": r.best_eval_reward,
                "timesteps": r.total_timesteps_completed,
                "target": r.total_timesteps_target,
                "wall_time": r.wall_time_seconds,
                "device": r.device,
            }

    # Table
    table = Table(title="Game Status")
    table.add_column("Game", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Progress", justify="right")
    table.add_column("Best Reward", justify="right")
    table.add_column("HNS", justify="right", style="green")
    table.add_column("Time", justify="right")

    for game_id in sorted(all_games.keys()):
        if game_id in trained_games:
            info = trained_games[game_id]
            pct = (info["timesteps"] / info["target"] * 100) if info["target"] else 0
            is_done = pct >= 100

            if is_done:
                status_str = "[green]Done[/green]"
            else:
                status_str = f"[yellow]{pct:.0f}%[/yellow]"

            reward_str = f"{info['best_reward']:.1f}" if info["best_reward"] is not None else "N/A"
            hns = human_normalized_score(game_id, info["best_reward"] or 0)
            hns_str = f"{hns:.2f}" if hns is not None else "-"
            hours = info["wall_time"] / 3600 if info["wall_time"] else 0
            time_str = f"{hours:.1f}h"
            progress_str = f"{info['timesteps']:,}/{info['target']:,}"
        else:
            status_str = "[dim]Not started[/dim]"
            reward_str = "-"
            hns_str = "-"
            time_str = "-"
            progress_str = "-"

        table.add_row(game_id, status_str, progress_str, reward_str, hns_str, time_str)

    console.print(table)

    # Summary stats
    if trained_games:
        total_steps = sum(t["timesteps"] for t in trained_games.values())
        total_time = sum(t["wall_time"] for t in trained_games.values())
        console.print(
            f"\n[dim]{len(trained_games)}/{len(all_games)} games trained"
            f"  |  {total_steps:,} total steps"
            f"  |  {total_time / 3600:.1f}h total training time[/dim]"
        )
    else:
        console.print(
            "\n[dim]No training runs yet. Get started with:[/dim] [cyan]golds go pong[/cyan]"
        )
