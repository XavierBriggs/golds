"""Tests for CLI shortcut commands."""

from pathlib import Path


def test_find_game_config_exact_match(tmp_path):
    """find_game_config finds exact YAML match."""
    from golds.cli.shortcuts import find_game_config

    config = tmp_path / "configs" / "games" / "pong.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("name: pong\n")

    result = find_game_config("pong", config_dir=tmp_path / "configs")
    assert result == config


def test_find_game_config_not_found(tmp_path):
    """find_game_config returns None for missing game."""
    from golds.cli.shortcuts import find_game_config

    configs_dir = tmp_path / "configs" / "games"
    configs_dir.mkdir(parents=True)

    result = find_game_config("nonexistent", config_dir=tmp_path / "configs")
    assert result is None


def test_make_output_dir():
    """make_output_dir creates a timestamped path."""
    from golds.cli.shortcuts import make_output_dir

    path = make_output_dir("pong", base=Path("/tmp/test_outputs"))
    assert "pong" in str(path)
    assert str(path).startswith("/tmp/test_outputs/pong_")


def test_make_output_dir_default_base():
    """make_output_dir defaults to outputs/ base."""
    from golds.cli.shortcuts import make_output_dir

    path = make_output_dir("breakout")
    assert str(path).startswith("outputs/breakout_")
