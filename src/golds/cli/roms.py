"""ROM management CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

rom_app = typer.Typer(help="ROM management commands")
console = Console()


@rom_app.command("import")
def rom_import(
    path: Path = typer.Argument(
        Path("roms"), help="Path to ROM directory or file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Import ROMs to stable-retro."""
    from golds.roms.manager import ROMManager

    if not path.exists():
        console.print(f"[red]Path does not exist: {path}[/red]")
        raise typer.Exit(1)

    manager = ROMManager(path if path.is_dir() else path.parent)

    console.print(f"[bold]Importing ROMs from: {path}[/bold]")

    if verbose and path.is_dir():
        # Scan and show found ROMs
        roms = manager.scan_roms()
        console.print(f"Found {len(roms)} ROM files:")
        for rom in roms:
            console.print(f"  - {rom.path.name} ({rom.platform})")
        console.print()

    # Import ROMs
    imported = manager.import_roms(path if path.is_dir() else path.parent)
    console.print(f"[green]Imported {imported} games to stable-retro[/green]")

    if imported > 0:
        console.print("\nTo verify, run: golds rom list")


@rom_app.command("list")
def rom_list(
    platform: Optional[str] = typer.Option(
        None, "--platform", "-p", help="Filter by platform (nes, snes, genesis)"
    ),
    search: Optional[str] = typer.Option(
        None, "--search", "-s", help="Search for game name"
    ),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results to show"),
) -> None:
    """List available games in stable-retro."""
    try:
        import retro
    except ImportError:
        console.print("[red]stable-retro is not installed[/red]")
        console.print("Install with: pip install stable-retro")
        raise typer.Exit(1)

    games = retro.data.list_games()

    # Filter by platform
    if platform:
        platform_suffix = f"-{platform.capitalize()}"
        games = [g for g in games if platform_suffix in g]

    # Filter by search term
    if search:
        search_lower = search.lower()
        games = [g for g in games if search_lower in g.lower()]

    games = sorted(games)

    if not games:
        console.print("[yellow]No games found matching criteria.[/yellow]")
        return

    table = Table(title=f"Available Retro Games ({len(games)} total)")
    table.add_column("Game ID", style="cyan")
    table.add_column("Platform", style="green")

    for game in games[:limit]:
        # Parse platform from game ID
        parts = game.rsplit("-", 1)
        plat = parts[-1] if len(parts) > 1 else "Unknown"
        table.add_row(game, plat)

    console.print(table)

    if len(games) > limit:
        console.print(f"[dim]... and {len(games) - limit} more[/dim]")


@rom_app.command("verify")
def rom_verify(
    game: str = typer.Argument(..., help="Game ID to verify"),
) -> None:
    """Verify a game is available and can be loaded."""
    from golds.roms.manager import ROMManager

    manager = ROMManager(Path("roms"))

    console.print(f"[bold]Verifying game: {game}[/bold]")

    if manager.verify_game_available(game):
        console.print(f"[green]Game '{game}' is available[/green]")

        # Try to get states
        try:
            import retro
            states = retro.data.list_states(game)
            if states:
                console.print(f"\nAvailable states:")
                for state in states[:10]:
                    console.print(f"  - {state}")
                if len(states) > 10:
                    console.print(f"  ... and {len(states) - 10} more")
        except Exception:
            pass

    else:
        console.print(f"[red]Game '{game}' is NOT available[/red]")
        console.print("\nTo import ROMs:")
        console.print("  1. Place ROM files in the 'roms' directory")
        console.print("  2. Run: golds rom import ./roms")
        raise typer.Exit(1)


@rom_app.command("info")
def rom_info() -> None:
    """Show ROM directory information and setup instructions."""
    console.print("[bold]ROM Setup Information[/bold]")
    console.print()

    console.print("[cyan]Atari ROMs:[/cyan]")
    console.print("  Atari environments use Gymnasium + ALE (`gymnasium[atari]` / `ale-py`).")
    console.print("  You also need an Atari ROM set (license restrictions apply).")
    console.print("  If ROMs are missing, use AutoROM (accepting the ROM license) or set `ALE_ROM_DIR`.")
    console.print()

    console.print("[cyan]Retro ROMs (NES/SNES/Genesis):[/cyan]")
    console.print("  Retro ROMs must be obtained legally and imported manually.")
    console.print()
    console.print("  Steps:")
    console.print("  1. Obtain ROMs legally (dump from cartridges you own)")
    console.print("  2. Create a 'roms' directory: mkdir roms")
    console.print("  3. Place ROM files in the directory")
    console.print("  4. Import: golds rom import ./roms")
    console.print("  5. Verify: golds rom verify SuperMarioBros-Nes")
    console.print()

    console.print("[cyan]Expected ROM Files:[/cyan]")
    console.print("  Super Mario Bros:")
    console.print("    - Super Mario Bros. (Japan, USA).nes")
    console.print("    - Game ID: SuperMarioBros-Nes")
    console.print()
    console.print("  Super Mario Bros. 2 Japan (Lost Levels):")
    console.print("    - Super Mario Bros. 2 (Japan).nes")
    console.print("    - Game ID: SuperMarioBros2Japan-Nes")
    console.print()

    # Show current ROM directory status
    rom_dir = Path("roms")
    if rom_dir.exists():
        roms = list(rom_dir.glob("*.nes")) + list(rom_dir.glob("*.sfc"))
        console.print(f"[green]ROM directory exists with {len(roms)} ROM files[/green]")
    else:
        console.print("[yellow]ROM directory does not exist yet[/yellow]")
