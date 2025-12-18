"""Tests for Atari env-id compatibility."""

from golds.environments.atari.env_id import resolve_atari_env_id


def test_resolve_atari_env_id_passthrough_for_ale_ids() -> None:
    assert resolve_atari_env_id("ALE/SpaceInvadersNoFrameskip-v5") == "ALE/SpaceInvadersNoFrameskip-v5"


def test_resolve_atari_env_id_keeps_legacy_v4_ids() -> None:
    assert resolve_atari_env_id("SpaceInvadersNoFrameskip-v4") == "SpaceInvadersNoFrameskip-v4"


def test_resolve_atari_env_id_maps_legacy_ids_without_version() -> None:
    assert resolve_atari_env_id("PongNoFrameskip") == "PongNoFrameskip-v4"
