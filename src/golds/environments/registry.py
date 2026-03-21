"""Game registry for managing supported games across platforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GameRegistration:
    """Registration entry for a game."""

    game_id: str
    platform: str  # 'atari' or 'retro'
    display_name: str
    env_id: str  # Platform-specific environment ID
    default_state: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GameRegistry:
    """Central registry for all supported games."""

    _games: dict[str, GameRegistration] = {}

    @classmethod
    def register(cls, registration: GameRegistration) -> None:
        """Register a game.

        Args:
            registration: Game registration entry
        """
        cls._games[registration.game_id] = registration

    @classmethod
    def get(cls, game_id: str) -> GameRegistration:
        """Get a game registration.

        Args:
            game_id: Game identifier

        Returns:
            Game registration

        Raises:
            ValueError: If game is not registered
        """
        if game_id not in cls._games:
            available = list(cls._games.keys())
            raise ValueError(f"Unknown game: '{game_id}'. Available games: {available}")
        return cls._games[game_id]

    @classmethod
    def list_games(cls, platform: str | None = None) -> list[str]:
        """List registered games.

        Args:
            platform: Filter by platform (optional)

        Returns:
            List of game IDs
        """
        if platform:
            return [g for g, r in cls._games.items() if r.platform == platform]
        return list(cls._games.keys())

    @classmethod
    def is_registered(cls, game_id: str) -> bool:
        """Check if a game is registered.

        Args:
            game_id: Game identifier

        Returns:
            True if registered
        """
        return game_id in cls._games


# Register Atari games
GameRegistry.register(
    GameRegistration(
        game_id="space_invaders",
        platform="atari",
        display_name="Space Invaders",
        env_id="SpaceInvadersNoFrameskip-v4",
    )
)

GameRegistry.register(
    GameRegistration(
        game_id="breakout",
        platform="atari",
        display_name="Breakout",
        env_id="BreakoutNoFrameskip-v4",
    )
)

GameRegistry.register(
    GameRegistration(
        game_id="pong",
        platform="atari",
        display_name="Pong",
        env_id="PongNoFrameskip-v4",
    )
)

GameRegistry.register(
    GameRegistration(
        game_id="qbert",
        platform="atari",
        display_name="Q*bert",
        env_id="QbertNoFrameskip-v4",
    )
)

GameRegistry.register(
    GameRegistration(
        game_id="seaquest",
        platform="atari",
        display_name="Seaquest",
        env_id="SeaquestNoFrameskip-v4",
    )
)

GameRegistry.register(
    GameRegistration(
        game_id="asteroids",
        platform="atari",
        display_name="Asteroids",
        env_id="AsteroidsNoFrameskip-v4",
    )
)

GameRegistry.register(
    GameRegistration(
        game_id="ms_pacman",
        platform="atari",
        display_name="Ms. Pac-Man",
        env_id="MsPacmanNoFrameskip-v4",
    )
)

GameRegistry.register(
    GameRegistration(
        game_id="montezuma_revenge",
        platform="atari",
        display_name="Montezuma's Revenge",
        env_id="MontezumaRevengeNoFrameskip-v4",
    )
)

GameRegistry.register(
    GameRegistration(
        game_id="enduro",
        platform="atari",
        display_name="Enduro",
        env_id="EnduroNoFrameskip-v4",
    )
)

GameRegistry.register(
    GameRegistration(
        game_id="frostbite",
        platform="atari",
        display_name="Frostbite",
        env_id="FrostbiteNoFrameskip-v4",
    )
)

# Register Retro games (NES)
GameRegistry.register(
    GameRegistration(
        game_id="super_mario_bros",
        platform="retro",
        display_name="Super Mario Bros. (NES)",
        env_id="SuperMarioBros-Nes-v0",
        default_state="Level1-1",
    )
)

GameRegistry.register(
    GameRegistration(
        game_id="super_mario_bros_2_japan",
        platform="retro",
        display_name="Super Mario Bros. 2 Japan (Lost Levels)",
        env_id="SuperMarioBros2Japan-Nes-v0",
        default_state="Level1-1",
    )
)

GameRegistry.register(
    GameRegistration(
        game_id="tetris",
        platform="retro",
        display_name="Tetris (Game Boy)",
        env_id="Tetris-GameBoy-v0",
        default_state="Level0.TypeA",
    )
)

GameRegistry.register(
    GameRegistration(
        game_id="mortal_kombat_ii",
        platform="retro",
        display_name="Mortal Kombat II (Genesis)",
        env_id="MortalKombatII-Genesis-v0",
    )
)

GameRegistry.register(
    GameRegistration(
        game_id="sonic_the_hedgehog",
        platform="retro",
        display_name="Sonic the Hedgehog (Genesis)",
        env_id="SonicTheHedgehog-Genesis-v0",
        default_state="GreenHillZone.Act1",
    )
)

GameRegistry.register(
    GameRegistration(
        game_id="street_fighter_ii",
        platform="retro",
        display_name="Street Fighter II (Genesis)",
        env_id="StreetFighterIISpecialChampionEdition-Genesis-v0",
        default_state=None,
    )
)

GameRegistry.register(
    GameRegistration(
        game_id="mega_man_2",
        platform="retro",
        display_name="Mega Man 2 (NES)",
        env_id="MegaMan2-Nes-v0",
        default_state=None,
    )
)
