"""Tests for retro environment wrappers."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers — lightweight fake envs
# ---------------------------------------------------------------------------


class _FakeRetroEnv(gym.Env):
    """Minimal env that returns configurable info dicts (simulating retro)."""

    def __init__(self, info_sequence: list[dict] | None = None):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(7)
        self._info_seq = info_sequence or []
        self._step_idx = 0

    def reset(self, **kwargs):
        self._step_idx = 0
        info = self._info_seq[0] if self._info_seq else {}
        return np.zeros(self.observation_space.shape, dtype=np.uint8), info

    def step(self, action):
        self._step_idx += 1
        idx = min(self._step_idx, len(self._info_seq) - 1) if self._info_seq else 0
        info = self._info_seq[idx] if self._info_seq else {}
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        return obs, 1.0, False, False, info


class _FakeRetroEnvWithButtons(gym.Env):
    """Fake retro env with configurable buttons attr for action space testing."""

    def __init__(self, buttons):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8)
        self.action_space = gym.spaces.MultiBinary(len(buttons))
        self.buttons = buttons
        self._last_action = None

    def reset(self, **kwargs):
        self._last_action = None
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        self._last_action = action
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        return obs, 1.0, False, False, {}

    @property
    def unwrapped(self):
        return self


class _FakeRetroEnvWithState(gym.Env):
    """Fake retro env that simulates load_state/initial_state for level rotation."""

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(7)
        self.initial_state = b"default_state"
        self._loaded_states = []

        class _FakeEm:
            def __init__(self, parent):
                self._parent = parent

            def get_state(self):
                return self._parent.initial_state

        self.em = _FakeEm(self)

    def load_state(self, state_name):
        self._loaded_states.append(state_name)
        self.initial_state = f"state_{state_name}".encode()

    def reset(self, **kwargs):
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.uint8), 0.0, False, False, {}

    @property
    def unwrapped(self):
        return self


# ---------------------------------------------------------------------------
# DiscreteActionWrapper tests
# ---------------------------------------------------------------------------


class TestDiscreteActionWrapper:
    def test_platformer_nes_has_9_actions(self):
        from golds.environments.retro.wrappers import DiscreteActionWrapper

        nes_buttons = ["B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT"]
        env = DiscreteActionWrapper(
            _FakeRetroEnvWithButtons(nes_buttons), action_set="platformer"
        )
        assert env.action_space == gym.spaces.Discrete(9)

    def test_platformer_genesis_has_9_actions(self):
        from golds.environments.retro.wrappers import DiscreteActionWrapper

        genesis_buttons = [
            "B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT",
            "C", "Y", "X", "Z",
        ]
        env = DiscreteActionWrapper(
            _FakeRetroEnvWithButtons(genesis_buttons), action_set="platformer"
        )
        assert env.action_space == gym.spaces.Discrete(9)

    def test_fighter_has_12_actions(self):
        from golds.environments.retro.wrappers import DiscreteActionWrapper

        genesis_buttons = [
            "B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT",
            "C", "Y", "X", "Z",
        ]
        env = DiscreteActionWrapper(
            _FakeRetroEnvWithButtons(genesis_buttons), action_set="fighter"
        )
        assert env.action_space == gym.spaces.Discrete(12)

    def test_puzzle_has_5_actions(self):
        from golds.environments.retro.wrappers import DiscreteActionWrapper

        nes_buttons = ["B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT"]
        env = DiscreteActionWrapper(
            _FakeRetroEnvWithButtons(nes_buttons), action_set="puzzle"
        )
        assert env.action_space == gym.spaces.Discrete(5)

    def test_noop_maps_to_all_zeros(self):
        from golds.environments.retro.wrappers import DiscreteActionWrapper

        nes_buttons = ["B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT"]
        env = DiscreteActionWrapper(
            _FakeRetroEnvWithButtons(nes_buttons), action_set="platformer"
        )
        env.reset()
        env.step(0)  # NOOP
        assert list(env.env._last_action) == [0] * 8

    def test_jump_maps_to_a_button(self):
        from golds.environments.retro.wrappers import DiscreteActionWrapper

        nes_buttons = ["B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT"]
        env = DiscreteActionWrapper(
            _FakeRetroEnvWithButtons(nes_buttons), action_set="platformer"
        )
        env.reset()
        env.step(6)  # A (jump)
        a_idx = nes_buttons.index("A")
        assert env.env._last_action[a_idx] == 1
        assert sum(env.env._last_action) == 1

    def test_unknown_action_set_raises(self):
        from golds.environments.retro.wrappers import DiscreteActionWrapper

        nes_buttons = ["B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT"]
        with pytest.raises(ValueError, match="Unknown action_set"):
            DiscreteActionWrapper(
                _FakeRetroEnvWithButtons(nes_buttons), action_set="invalid"
            )


# ---------------------------------------------------------------------------
# StickyActionWrapper tests
# ---------------------------------------------------------------------------


class TestStickyActionWrapper:
    def test_no_stick_on_first_step(self):
        from golds.environments.retro.wrappers import StickyActionWrapper

        env = StickyActionWrapper(_FakeRetroEnv(), stickprob=1.0)
        env.reset()
        env.step(0)  # should not error

    def test_stickprob_zero_never_repeats(self):
        from golds.environments.retro.wrappers import StickyActionWrapper

        class _ActionTracker(_FakeRetroEnv):
            def __init__(self):
                super().__init__()
                self.actions = []

            def step(self, action):
                self.actions.append(action)
                return super().step(action)

        inner = _ActionTracker()
        env = StickyActionWrapper(inner, stickprob=0.0)
        env.reset()
        for i in range(10):
            env.step(i % 7)
        assert inner.actions == [i % 7 for i in range(10)]

    def test_stickprob_one_always_repeats(self):
        from golds.environments.retro.wrappers import StickyActionWrapper

        class _ActionTracker(_FakeRetroEnv):
            def __init__(self):
                super().__init__()
                self.actions = []

            def step(self, action):
                self.actions.append(action)
                return super().step(action)

        inner = _ActionTracker()
        env = StickyActionWrapper(inner, stickprob=1.0)
        env.reset()
        env.step(5)  # first step: action 5 (no stick, no previous)
        env.step(3)  # should stick to 5
        env.step(1)  # should stick to 5
        assert inner.actions == [5, 5, 5]

    def test_reset_clears_last_action(self):
        from golds.environments.retro.wrappers import StickyActionWrapper

        class _ActionTracker(_FakeRetroEnv):
            def __init__(self):
                super().__init__()
                self.actions = []

            def step(self, action):
                self.actions.append(action)
                return super().step(action)

        inner = _ActionTracker()
        env = StickyActionWrapper(inner, stickprob=1.0)
        env.reset()
        env.step(5)
        env.reset()
        env.step(3)  # after reset, no last_action → passes through
        assert inner.actions[-1] == 3


# ---------------------------------------------------------------------------
# MultiLevelWrapper tests
# ---------------------------------------------------------------------------


class TestMultiLevelWrapper:
    def test_picks_from_level_list(self):
        from golds.environments.retro.wrappers import MultiLevelWrapper

        inner = _FakeRetroEnvWithState()
        levels = ["GreenHillZone.Act1", "GreenHillZone.Act2", "MarbleZone.Act1"]
        env = MultiLevelWrapper(inner, levels=levels)

        for _ in range(10):
            env.reset()

        assert len(inner._loaded_states) == 10
        assert all(s in levels for s in inner._loaded_states)

    def test_sets_initial_state_before_reset(self):
        from golds.environments.retro.wrappers import MultiLevelWrapper

        inner = _FakeRetroEnvWithState()
        env = MultiLevelWrapper(inner, levels=["Level1-1"])
        env.reset()

        assert inner.initial_state == b"state_Level1-1"

    def test_covers_multiple_levels(self):
        from golds.environments.retro.wrappers import MultiLevelWrapper

        inner = _FakeRetroEnvWithState()
        levels = ["A", "B", "C"]
        env = MultiLevelWrapper(inner, levels=levels)

        for _ in range(100):
            env.reset()

        assert set(inner._loaded_states) == set(levels)

    def test_empty_levels_raises(self):
        from golds.environments.retro.wrappers import MultiLevelWrapper

        with pytest.raises(ValueError, match="non-empty"):
            MultiLevelWrapper(_FakeRetroEnvWithState(), levels=[])


# ---------------------------------------------------------------------------
# PlatformerRewardWrapper tests
# ---------------------------------------------------------------------------


class TestPlatformerRewardWrapper:
    def test_shaped_reward_increases_with_forward_progress(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 0}, {"x": 10}, {"x": 25}]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos), scale=0.1, game="SonicTheHedgehog-Genesis"
        )
        env.reset()

        _, reward1, _, _, info1 = env.step(0)
        assert reward1 == pytest.approx(2.0)
        assert info1["raw_reward"] == pytest.approx(1.0)

        _, reward2, _, _, _ = env.step(0)
        assert reward2 == pytest.approx(2.5)

    def test_backward_movement_gives_negative_shaping(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 100}, {"x": 90}]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos), scale=0.1, game="SonicTheHedgehog-Genesis"
        )
        env.reset()

        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx(0.0)

    def test_mario_extractor_uses_scroll_registers(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [
            {"xscrollLo": 0, "xscrollHi": 0},
            {"xscrollLo": 0, "xscrollHi": 1},
        ]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos), scale=0.01, game="SuperMarioBros-Nes"
        )
        env.reset()

        _, _, _, _, info = env.step(0)
        assert info["shaped_x_progress"] == pytest.approx(2.56)

    def test_zero_scale_disables_shaping(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 0}, {"x": 100}]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos), scale=0.0, game="SonicTheHedgehog-Genesis"
        )
        env.reset()

        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx(1.0)

    def test_reset_resets_x_old(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 50}, {"x": 60}]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos), scale=0.1, game="SonicTheHedgehog-Genesis"
        )
        env.reset()
        assert env._x_old == 50.0
        env.step(0)
        assert env._x_old == 60.0
        env.reset()
        assert env._x_old == 50.0


# ---------------------------------------------------------------------------
# Extended PlatformerRewardWrapper tests (death/collectible/time)
# ---------------------------------------------------------------------------


class TestPlatformerRewardWrapperExtended:
    def test_death_penalty_on_terminated(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        class _TerminatingEnv(_FakeRetroEnv):
            def __init__(self):
                super().__init__(info_sequence=[{"x": 0}, {"x": 0}])

            def step(self, action):
                obs, reward, _, truncated, info = super().step(action)
                return obs, reward, True, truncated, info

        env = PlatformerRewardWrapper(
            _TerminatingEnv(),
            scale=0.0,
            game="SonicTheHedgehog-Genesis",
            death_penalty=-1.0,
        )
        env.reset()
        _, reward, _, _, info = env.step(0)
        assert info["shaped_death"] == -1.0
        assert reward == pytest.approx(1.0 + (-1.0))

    def test_no_death_penalty_on_truncated(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        class _TruncatingEnv(_FakeRetroEnv):
            def __init__(self):
                super().__init__(info_sequence=[{"x": 0}, {"x": 0}])

            def step(self, action):
                obs, reward, terminated, _, info = super().step(action)
                return obs, reward, terminated, True, info

        env = PlatformerRewardWrapper(
            _TruncatingEnv(),
            scale=0.0,
            game="SonicTheHedgehog-Genesis",
            death_penalty=-1.0,
        )
        env.reset()
        _, reward, _, _, info = env.step(0)
        assert info["shaped_death"] == 0.0
        assert reward == pytest.approx(1.0)

    def test_collectible_bonus(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 0, "rings": 0}, {"x": 0, "rings": 5}]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos),
            scale=0.0,
            game="SonicTheHedgehog-Genesis",
            collectible_reward_scale=0.1,
        )
        env.reset()
        _, reward, _, _, info = env.step(0)
        assert info["shaped_collectible"] == pytest.approx(0.5)
        assert reward == pytest.approx(1.0 + 0.5)

    def test_time_penalty(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 0}, {"x": 0}]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos),
            scale=0.0,
            game="SonicTheHedgehog-Genesis",
            time_penalty=-0.01,
        )
        env.reset()
        _, reward, _, _, info = env.step(0)
        assert info["shaped_time"] == pytest.approx(-0.01)
        assert reward == pytest.approx(1.0 + (-0.01))

    def test_all_signals_combined(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 0, "rings": 0}, {"x": 10, "rings": 3}]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos),
            scale=0.1,
            game="SonicTheHedgehog-Genesis",
            death_penalty=-1.0,
            collectible_reward_scale=0.1,
            time_penalty=-0.001,
        )
        env.reset()
        _, reward, _, _, info = env.step(0)
        # raw=1.0, x=0.1*10=1.0, collect=0.1*3=0.3, time=-0.001, death=0
        expected = 1.0 + 1.0 + 0.3 + (-0.001) + 0.0
        assert reward == pytest.approx(expected)


# ---------------------------------------------------------------------------
# PlatformerRewardWrapper: progress_mode="delta_max_x" (ADR-004 / R3)
# ---------------------------------------------------------------------------


class TestPlatformerRewardWrapperDeltaMaxX:
    def test_backward_compat_default_is_delta_x(self):
        """progress_mode defaults to 'delta_x' so other games keep old behavior."""
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 100}, {"x": 90}]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos), scale=0.1, game="SonicTheHedgehog-Genesis"
        )
        assert env.progress_mode == "delta_x"
        env.reset()
        _, reward, _, _, _ = env.step(0)
        # Old behavior: raw(1.0) + scale*(90-100) = 1.0 - 1.0 = 0.0
        assert reward == pytest.approx(0.0)

    def test_delta_x_mode_explicit_matches_old_behavior(self):
        """Explicitly requesting delta_x still punishes backtracking (unchanged)."""
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 0}, {"x": 10}, {"x": 25}]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos),
            scale=0.1,
            game="SonicTheHedgehog-Genesis",
            progress_mode="delta_x",
        )
        env.reset()
        _, reward1, _, _, _ = env.step(0)
        assert reward1 == pytest.approx(2.0)
        _, reward2, _, _, _ = env.step(0)
        assert reward2 == pytest.approx(2.5)

    def test_delta_max_x_rewards_only_new_forward_progress(self):
        """Forward, then backward, then forward past the old max.

        x sequence: 0 -> 10 -> 25 -> 15 -> 5 -> 30 (reset -> steps 1..5)
        scale=0.1, raw reward is always 1.0 (per _FakeRetroEnv).

        step1: x=10, progress=max(0,10-0)=10  -> shaped=1.0 -> reward=2.0, max_x=10
        step2: x=25, progress=max(0,25-10)=15 -> shaped=1.5 -> reward=2.5, max_x=25
        step3: x=15 (backtrack), progress=max(0,15-25)=0 -> shaped=0.0 -> reward=1.0, max_x=25
        step4: x=5  (further back), progress=max(0,5-25)=0 -> shaped=0.0 -> reward=1.0, max_x=25
        step5: x=30 (new max, re-covers + exceeds), progress=max(0,30-25)=5 -> shaped=0.5 -> reward=1.5, max_x=30
        """
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 0}, {"x": 10}, {"x": 25}, {"x": 15}, {"x": 5}, {"x": 30}]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos),
            scale=0.1,
            game="SonicTheHedgehog-Genesis",
            progress_mode="delta_max_x",
        )
        env.reset()

        _, reward1, _, _, info1 = env.step(0)
        assert reward1 == pytest.approx(2.0)
        assert info1["shaped_x_progress"] == pytest.approx(1.0)

        _, reward2, _, _, info2 = env.step(0)
        assert reward2 == pytest.approx(2.5)
        assert info2["shaped_x_progress"] == pytest.approx(1.5)

        _, reward3, _, _, info3 = env.step(0)
        assert reward3 == pytest.approx(1.0)
        assert info3["shaped_x_progress"] == pytest.approx(0.0)

        _, reward4, _, _, info4 = env.step(0)
        assert reward4 == pytest.approx(1.0)
        assert info4["shaped_x_progress"] == pytest.approx(0.0)

        _, reward5, _, _, info5 = env.step(0)
        assert reward5 == pytest.approx(1.5)
        assert info5["shaped_x_progress"] == pytest.approx(0.5)

        progress = env.get_episode_progress()
        assert progress["max_x"] == pytest.approx(30.0)

    def test_delta_max_x_resets_max_on_episode_reset(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 50}, {"x": 80}]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos),
            scale=0.1,
            game="SonicTheHedgehog-Genesis",
            progress_mode="delta_max_x",
        )
        env.reset()
        env.step(0)
        assert env.get_episode_progress()["max_x"] == pytest.approx(80.0)

        env.reset()
        assert env.get_episode_progress()["max_x"] == pytest.approx(50.0)

    def test_invalid_progress_mode_raises(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        with pytest.raises(ValueError, match="progress_mode"):
            PlatformerRewardWrapper(
                _FakeRetroEnv(), game="SonicTheHedgehog-Genesis", progress_mode="bogus"
            )


# ---------------------------------------------------------------------------
# PlatformerRewardWrapper: completion detector (ADR-004 / R4)
# ---------------------------------------------------------------------------


class TestPlatformerRewardWrapperCompletion:
    def test_threshold_completion_emits_bonus_once(self):
        """x crossing level_end_x emits the completion bonus exactly once."""
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 0}, {"x": 50}, {"x": 100}, {"x": 150}]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos),
            scale=0.0,
            game="SonicTheHedgehog-Genesis",
            level_end_x=100.0,
            completion_bonus=50.0,
        )
        env.reset()

        _, _, _, _, info1 = env.step(0)  # x=50, below threshold
        assert info1["shaped_completion_bonus"] == pytest.approx(0.0)
        assert info1["level_complete"] is False

        _, _, _, _, info2 = env.step(0)  # x=100, crosses threshold
        assert info2["shaped_completion_bonus"] == pytest.approx(50.0)
        assert info2["level_complete"] is True

        _, _, _, _, info3 = env.step(0)  # x=150, still complete, bonus not repeated
        assert info3["shaped_completion_bonus"] == pytest.approx(0.0)
        assert info3["level_complete"] is True

        progress = env.get_episode_progress()
        assert progress["completed"] is True
        assert progress["max_x"] == pytest.approx(150.0)
        assert progress["level_end_x"] == pytest.approx(100.0)

    def test_below_threshold_never_completes(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 0}, {"x": 10}, {"x": 40}]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos),
            scale=0.0,
            game="SonicTheHedgehog-Genesis",
            level_end_x=100.0,
            completion_bonus=50.0,
        )
        env.reset()
        for _ in range(len(infos) - 1):
            _, _, _, _, info = env.step(0)
            assert info["shaped_completion_bonus"] == pytest.approx(0.0)
            assert info["level_complete"] is False

        progress = env.get_episode_progress()
        assert progress["completed"] is False
        assert progress["max_x"] == pytest.approx(40.0)

    def test_completion_terminates_episode_by_default(self):
        """Reaching the signpost ends the episode (objective = complete Act 1)."""
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 0}, {"x": 50}, {"x": 100}]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos),
            scale=0.0,
            game="SonicTheHedgehog-Genesis",
            level_end_x=100.0,
            completion_bonus=50.0,
        )
        env.reset()
        _, _, term1, _, _ = env.step(0)  # x=50, not yet
        assert term1 is False
        _, _, term2, _, info2 = env.step(0)  # x=100, crosses signpost
        assert info2["level_complete"] is True
        assert term2 is True

    def test_terminate_on_completion_false_keeps_episode_running(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 0}, {"x": 50}, {"x": 100}]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos),
            scale=0.0,
            game="SonicTheHedgehog-Genesis",
            level_end_x=100.0,
            completion_bonus=50.0,
            terminate_on_completion=False,
        )
        env.reset()
        env.step(0)  # x=50
        _, _, term2, _, info2 = env.step(0)  # x=100 completes
        assert info2["level_complete"] is True
        assert term2 is False

    def test_none_threshold_disables_threshold_completion(self):
        """level_end_x=None (the default) means completion-by-threshold is disabled."""
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 0}, {"x": 999999.0}]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos),
            scale=0.0,
            game="SonicTheHedgehog-Genesis",
            completion_bonus=50.0,
        )
        assert env.level_end_x is None
        env.reset()
        _, _, _, _, info = env.step(0)
        assert info["shaped_completion_bonus"] == pytest.approx(0.0)
        assert info["level_complete"] is False

        progress = env.get_episode_progress()
        assert progress["completed"] is False
        assert progress["level_end_x"] is None

    def test_completion_via_info_key_signal(self):
        """An optional info-dict key can also signal completion (e.g. act change)."""
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [
            {"x": 0, "level_end_bonus": 0},
            {"x": 10, "level_end_bonus": 0},
            {"x": 20, "level_end_bonus": 0},
            {"x": 30, "level_end_bonus": 1},
        ]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos),
            scale=0.0,
            game="SonicTheHedgehog-Genesis",
            level_end_info_key="level_end_bonus",
            completion_bonus=25.0,
        )
        env.reset()

        _, _, _, _, info1 = env.step(0)
        assert info1["level_complete"] is False

        _, _, _, _, info2 = env.step(0)
        assert info2["level_complete"] is False

        _, _, _, _, info3 = env.step(0)
        assert info3["shaped_completion_bonus"] == pytest.approx(25.0)
        assert info3["level_complete"] is True

        assert env.get_episode_progress()["completed"] is True


# ---------------------------------------------------------------------------
# PlatformerRewardWrapper: get_episode_progress (ADR-004 / R5)
# ---------------------------------------------------------------------------


class TestPlatformerRewardWrapperEpisodeProgress:
    def test_progress_shape_before_any_step(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 7}]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos),
            game="SonicTheHedgehog-Genesis",
            level_end_x=1000.0,
        )
        env.reset()
        progress = env.get_episode_progress()
        assert progress == {"completed": False, "max_x": pytest.approx(7.0), "level_end_x": pytest.approx(1000.0)}

    def test_progress_reflects_completion_rate_across_episodes(self):
        """Simulates what an eval loop would do: query progress per episode."""
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        # Episode 1: reaches the end. Episode 2: falls short.
        infos = [{"x": 0}, {"x": 100}]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos),
            game="SonicTheHedgehog-Genesis",
            level_end_x=100.0,
        )

        env.reset()
        env.step(0)
        episode1 = env.get_episode_progress()
        assert episode1["completed"] is True
        assert episode1["max_x"] == pytest.approx(100.0)

        # New episode starts short of the threshold.
        env2_infos = [{"x": 0}, {"x": 40}]
        env.env = _FakeRetroEnv(env2_infos)
        env.reset()
        env.step(0)
        episode2 = env.get_episode_progress()
        assert episode2["completed"] is False
        assert episode2["max_x"] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# TimeLimitWrapper tests
# ---------------------------------------------------------------------------


class TestTimeLimitWrapper:
    def test_truncates_at_max_steps(self):
        from golds.environments.retro.wrappers import TimeLimitWrapper

        env = TimeLimitWrapper(_FakeRetroEnv(), max_steps=5)
        env.reset()

        for i in range(4):
            _, _, _, truncated, _ = env.step(0)
            assert not truncated, f"Should not truncate at step {i+1}"

        _, _, _, truncated, _ = env.step(0)
        assert truncated, "Should truncate at step 5"

    def test_reset_clears_step_count(self):
        from golds.environments.retro.wrappers import TimeLimitWrapper

        env = TimeLimitWrapper(_FakeRetroEnv(), max_steps=3)
        env.reset()

        for _ in range(3):
            env.step(0)

        env.reset()
        for i in range(2):
            _, _, _, truncated, _ = env.step(0)
            assert not truncated

        _, _, _, truncated, _ = env.step(0)
        assert truncated

    def test_preserves_terminated_flag(self):
        from golds.environments.retro.wrappers import TimeLimitWrapper

        class _TerminatingEnv(_FakeRetroEnv):
            def step(self, action):
                obs, reward, _, truncated, info = super().step(action)
                return obs, reward, True, truncated, info

        env = TimeLimitWrapper(_TerminatingEnv(), max_steps=100)
        env.reset()
        _, _, terminated, _, _ = env.step(0)
        assert terminated
