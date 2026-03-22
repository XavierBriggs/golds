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
