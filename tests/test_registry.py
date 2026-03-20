"""Tests for the game registry."""

import pytest

from golds.environments.registry import GameRegistration, GameRegistry


def test_list_all_games():
    games = GameRegistry.list_games()
    assert len(games) > 0
    assert "space_invaders" in games
    assert "breakout" in games
    assert "pong" in games


def test_list_atari_games():
    atari_games = GameRegistry.list_games(platform="atari")
    assert "space_invaders" in atari_games
    assert "breakout" in atari_games
    # Retro games should not appear
    assert "super_mario_bros" not in atari_games


def test_list_retro_games():
    retro_games = GameRegistry.list_games(platform="retro")
    assert "super_mario_bros" in retro_games
    assert "tetris" in retro_games
    # Atari games should not appear
    assert "space_invaders" not in retro_games


def test_get_registered_game():
    reg = GameRegistry.get("breakout")
    assert isinstance(reg, GameRegistration)
    assert reg.game_id == "breakout"
    assert reg.platform == "atari"
    assert reg.env_id == "BreakoutNoFrameskip-v4"
    assert reg.display_name == "Breakout"


def test_get_unknown_game():
    with pytest.raises(ValueError, match="Unknown game"):
        GameRegistry.get("nonexistent_game_xyz")


def test_is_registered():
    assert GameRegistry.is_registered("pong") is True
    assert GameRegistry.is_registered("nonexistent_game_xyz") is False


def test_new_games_registered():
    """Verify that the Phase 3 games are registered."""
    for game_id in [
        "montezuma_revenge",
        "enduro",
        "frostbite",
        "street_fighter_ii",
        "mega_man_2",
    ]:
        assert GameRegistry.is_registered(game_id), f"{game_id} should be registered"
