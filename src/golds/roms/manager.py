"""ROM management utilities for stable-retro."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ROMInfo:
    """ROM file information."""

    path: Path
    platform: str
    sha1_hash: str
    file_size: int

    @property
    def name(self) -> str:
        return self.path.name


class ROMManager:
    """Manage ROM files for stable-retro.

    Provides utilities for scanning, importing, and validating ROMs.
    """

    # File extension to platform mapping
    PLATFORM_MAP = {
        ".nes": "nes",
        ".sfc": "snes",
        ".smc": "snes",
        ".md": "genesis",
        ".gen": "genesis",
        ".bin": "genesis",
        ".a26": "atari2600",
        ".gb": "gb",
        ".gbc": "gbc",
        ".gba": "gba",
        ".pce": "pce",
    }

    def __init__(self, rom_directory: Path | str) -> None:
        """Initialize ROM manager.

        Args:
            rom_directory: Directory containing ROM files
        """
        self.rom_directory = Path(rom_directory)

    @staticmethod
    def compute_sha1(file_path: Path) -> str:
        """Compute SHA1 hash of a file.

        Args:
            file_path: Path to file

        Returns:
            SHA1 hash as hex string
        """
        sha1 = hashlib.sha1()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha1.update(chunk)
        return sha1.hexdigest()

    def _detect_platform(self, file_path: Path) -> str:
        """Detect platform from file extension.

        Args:
            file_path: Path to ROM file

        Returns:
            Platform name
        """
        ext = file_path.suffix.lower()
        return self.PLATFORM_MAP.get(ext, "unknown")

    def scan_roms(self) -> list[ROMInfo]:
        """Scan ROM directory for ROM files.

        Returns:
            List of ROMInfo objects
        """
        if not self.rom_directory.exists():
            return []

        roms = []
        extensions = set(self.PLATFORM_MAP.keys())

        for file_path in self.rom_directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    sha1 = self.compute_sha1(file_path)
                    platform = self._detect_platform(file_path)
                    roms.append(
                        ROMInfo(
                            path=file_path,
                            platform=platform,
                            sha1_hash=sha1,
                            file_size=file_path.stat().st_size,
                        )
                    )
                except Exception:
                    pass  # Skip files that can't be read

        return roms

    def import_roms(self, source_dir: Path | str | None = None) -> int:
        """Import ROMs to stable-retro.

        Args:
            source_dir: Directory containing ROMs. Defaults to rom_directory.

        Returns:
            Number of games imported
        """
        import_dir = Path(source_dir) if source_dir else self.rom_directory

        if not import_dir.exists():
            return 0

        try:
            # Use the current interpreter so we import into the same environment
            # (uv/venv installs stable-retro into this interpreter).
            result = subprocess.run(
                [sys.executable, "-m", "retro.import", str(import_dir)],
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            return 0

        output = result.stdout + result.stderr

        # Parse output for imported count
        # Example: "Imported 3 games"
        match = re.search(r"Imported (\d+) games?", output)
        if match:
            return int(match.group(1))

        # Check for individual game imports
        # Example: "Importing SuperMarioBros-Nes"
        imports = re.findall(r"Importing (\S+)", output)
        return len(imports)

    def verify_game_available(self, game_id: str) -> bool:
        """Check if a game is available in stable-retro.

        Args:
            game_id: Game identifier (e.g., 'SuperMarioBros-Nes')

        Returns:
            True if game is available
        """
        try:
            import retro

            return game_id in retro.data.list_games()
        except ImportError:
            return False

    def list_available_games(self, platform: str | None = None) -> list[str]:
        """List all available games in stable-retro.

        Args:
            platform: Filter by platform (optional)

        Returns:
            List of game IDs
        """
        try:
            import retro

            games = retro.data.list_games()

            if platform:
                suffix = f"-{platform.capitalize()}"
                games = [g for g in games if g.endswith(suffix)]

            return sorted(games)
        except ImportError:
            return []

    def get_game_states(self, game_id: str) -> list[str]:
        """Get available states for a game.

        Args:
            game_id: Game identifier

        Returns:
            List of state names
        """
        try:
            import retro

            return retro.data.list_states(game_id)
        except Exception:
            return []

    def ensure_directory(self) -> None:
        """Create ROM directory if it doesn't exist."""
        self.rom_directory.mkdir(parents=True, exist_ok=True)

    def get_instructions(self) -> str:
        """Get ROM setup instructions.

        Returns:
            Instructions as formatted string
        """
        return """
ROM Setup Instructions
======================

1. ATARI ROMs (Automatic)
   Atari environments require `gymnasium[atari]`/`ale-py` and a ROM set.

   Options:
   - If you already have ROMs: configure the ALE ROM path (e.g. `ALE_ROM_DIR`).
   - Automatic download (accepting the ROM license): use AutoROM if available in your setup.

2. NES/SNES/Genesis ROMs (Manual)

   a) Obtain ROMs legally:
      - Dump from cartridges you own using a ROM dumper
      - Some digital purchases include ROM files

   b) Place ROMs in the roms/ directory:
      mkdir roms
      cp /path/to/your/roms/*.nes roms/

   c) Import to stable-retro:
      golds rom import ./roms

   d) Verify import:
      golds rom verify SuperMarioBros-Nes

Expected Files for Super Mario Bros:
   - Super Mario Bros. (Japan, USA).nes
   - Super Mario Bros. (World).nes
   - Or similar named ROM files

The ROM filename doesn't have to match exactly - stable-retro
uses SHA1 hashes to identify ROMs.
"""
