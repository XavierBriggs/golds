# Training Upgrade v3 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add action space reduction, sticky actions, multi-level rotation, RND exploration, and extended reward shaping to make retro agents complete levels instead of getting stuck.

**Architecture:** New gym wrappers (`DiscreteActionWrapper`, `StickyActionWrapper`, `MultiLevelWrapper`) go in the existing `wrappers.py`. RND is a `VecEnvWrapper` in a new `rnd.py` module. Extended reward signals are added to the existing `PlatformerRewardWrapper`. All wrappers are wired through the existing config → trainer → factory → maker pipeline.

**Tech Stack:** Python 3.12, gymnasium, stable-retro, stable-baselines3, PyTorch, Pydantic, pytest

**Spec:** `docs/superpowers/specs/2026-03-22-training-upgrade-v3-design.md`

---

## Chunk 1: Core Wrappers (Action Space + Sticky Actions)

### Task 1: DiscreteActionWrapper

**Files:**
- Modify: `src/golds/environments/retro/wrappers.py`
- Modify: `tests/test_wrappers.py`

- [ ] **Step 1: Write failing tests for DiscreteActionWrapper**

Add to `tests/test_wrappers.py`:

```python
class _FakeRetroEnvWithButtons(gym.Env):
    """Fake retro env with configurable buttons attr for action space testing."""

    def __init__(self, buttons):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8)
        self.action_space = gym.spaces.MultiBinary(len(buttons))
        self.buttons = buttons  # mimics env.unwrapped.buttons
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

        genesis_buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        env = DiscreteActionWrapper(
            _FakeRetroEnvWithButtons(genesis_buttons), action_set="platformer"
        )
        assert env.action_space == gym.spaces.Discrete(9)

    def test_fighter_has_12_actions(self):
        from golds.environments.retro.wrappers import DiscreteActionWrapper

        genesis_buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
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

    def test_action_maps_to_correct_buttons(self):
        from golds.environments.retro.wrappers import DiscreteActionWrapper

        nes_buttons = ["B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT"]
        env = DiscreteActionWrapper(
            _FakeRetroEnvWithButtons(nes_buttons), action_set="platformer"
        )
        env.reset()
        # Action 0 = NOOP → all zeros
        env.step(0)
        assert list(env.env._last_action) == [0, 0, 0, 0, 0, 0, 0, 0]
        # Action 6 = A (jump) → only A button
        env.step(6)
        a_idx = nes_buttons.index("A")
        assert env.env._last_action[a_idx] == 1
        assert sum(env.env._last_action) == 1

    def test_step_returns_correct_shape(self):
        from golds.environments.retro.wrappers import DiscreteActionWrapper

        nes_buttons = ["B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT"]
        env = DiscreteActionWrapper(
            _FakeRetroEnvWithButtons(nes_buttons), action_set="platformer"
        )
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (84, 84, 3)
        assert reward == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_wrappers.py::TestDiscreteActionWrapper -v`
Expected: FAIL with `ImportError` or `AttributeError`

- [ ] **Step 3: Implement DiscreteActionWrapper**

Add to `src/golds/environments/retro/wrappers.py`:

```python
import numpy as np

# Action set definitions: list of (name, button_names_to_press) tuples.
# Button names match env.unwrapped.buttons (e.g. "A", "B", "RIGHT").
ACTION_SETS: dict[str, list[tuple[str, list[str]]]] = {
    "platformer": [
        ("NOOP", []),
        ("Left", ["LEFT"]),
        ("Right", ["RIGHT"]),
        ("Down", ["DOWN"]),
        ("Right+B", ["RIGHT", "B"]),
        ("Down+B", ["DOWN", "B"]),
        ("A", ["A"]),
        ("Right+A", ["RIGHT", "A"]),
        ("Right+A+B", ["RIGHT", "A", "B"]),
    ],
    "fighter": [
        ("NOOP", []),
        ("Left", ["LEFT"]),
        ("Right", ["RIGHT"]),
        ("Up", ["UP"]),
        ("Down", ["DOWN"]),
        ("A", ["A"]),
        ("B", ["B"]),
        ("Down+A", ["DOWN", "A"]),
        ("Down+B", ["DOWN", "B"]),
        ("Left+A", ["LEFT", "A"]),
        ("Right+A", ["RIGHT", "A"]),
        ("Right+B", ["RIGHT", "B"]),
    ],
    "puzzle": [
        ("NOOP", []),
        ("Left", ["LEFT"]),
        ("Right", ["RIGHT"]),
        ("Down", ["DOWN"]),
        ("A", ["A"]),
    ],
}


class DiscreteActionWrapper(gym.Wrapper):
    """Maps a small Discrete action space to MultiBinary button presses."""

    def __init__(self, env: gym.Env, action_set: str = "platformer") -> None:
        super().__init__(env)
        if action_set not in ACTION_SETS:
            raise ValueError(f"Unknown action_set: {action_set}. Choose from {list(ACTION_SETS)}")

        buttons = env.unwrapped.buttons
        n_buttons = len(buttons)
        combos = ACTION_SETS[action_set]

        # Build lookup table: action_index → MultiBinary array
        self._lookup = np.zeros((len(combos), n_buttons), dtype=np.int8)
        for i, (_name, btn_names) in enumerate(combos):
            for btn in btn_names:
                if btn in buttons:
                    self._lookup[i, buttons.index(btn)] = 1

        self.action_space = gym.spaces.Discrete(len(combos))

    def step(self, action: int) -> tuple:
        return self.env.step(self._lookup[action])

    def reset(self, **kwargs) -> tuple:
        return self.env.reset(**kwargs)
```

Also add `import numpy as np` at the top of `wrappers.py` (it was removed earlier for lint).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_wrappers.py::TestDiscreteActionWrapper -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/golds/environments/retro/wrappers.py tests/test_wrappers.py
git commit -m "feat: add DiscreteActionWrapper for action space reduction"
```

---

### Task 2: StickyActionWrapper

**Files:**
- Modify: `src/golds/environments/retro/wrappers.py`
- Modify: `tests/test_wrappers.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_wrappers.py`:

```python
class TestStickyActionWrapper:
    def test_no_stick_on_first_step(self):
        from golds.environments.retro.wrappers import StickyActionWrapper

        env = StickyActionWrapper(_FakeRetroEnv(), stickprob=1.0)
        env.reset()
        # First step should never stick (no previous action)
        env.step(0)
        # No assertion needed — just verifying it doesn't error

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
        env.step(5)   # first step: action 5 (no stick)
        env.step(3)   # should stick to 5
        env.step(1)   # should stick to 5
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
        env.step(3)  # after reset, no last_action, so action 3 passes through
        assert inner.actions[-1] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_wrappers.py::TestStickyActionWrapper -v`
Expected: FAIL

- [ ] **Step 3: Implement StickyActionWrapper**

Add to `src/golds/environments/retro/wrappers.py`:

```python
class StickyActionWrapper(gym.Wrapper):
    """Repeat the previous action with probability stickprob.

    Breaks determinism in retro games so the agent cannot memorize
    frame-perfect action sequences. Standard practice: stickprob=0.25.
    """

    def __init__(self, env: gym.Env, stickprob: float = 0.25) -> None:
        super().__init__(env)
        self.stickprob = stickprob
        self._last_action = None
        self._rng = np.random.default_rng()

    def step(self, action: Any) -> tuple:
        if self._last_action is not None and self._rng.random() < self.stickprob:
            action = self._last_action
        self._last_action = action
        return self.env.step(action)

    def reset(self, **kwargs) -> tuple:
        self._last_action = None
        seed = kwargs.get("seed")
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return self.env.reset(**kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_wrappers.py::TestStickyActionWrapper -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/golds/environments/retro/wrappers.py tests/test_wrappers.py
git commit -m "feat: add StickyActionWrapper for breaking determinism"
```

---

### Task 3: MultiLevelWrapper

**Files:**
- Modify: `src/golds/environments/retro/wrappers.py`
- Modify: `tests/test_wrappers.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_wrappers.py`:

```python
class _FakeRetroEnvWithState(gym.Env):
    """Fake retro env that simulates load_state/initial_state for level rotation."""

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(7)
        self.initial_state = b"default_state"
        self._loaded_states = []

        # Fake emulator object
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


class TestMultiLevelWrapper:
    def test_picks_from_level_list(self):
        from golds.environments.retro.wrappers import MultiLevelWrapper

        inner = _FakeRetroEnvWithState()
        levels = ["GreenHillZone.Act1", "GreenHillZone.Act2", "MarbleZone.Act1"]
        env = MultiLevelWrapper(inner, levels=levels)

        for _ in range(10):
            env.reset()

        # All loaded states should be from the level list
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

        # With 100 resets and 3 levels, all should be covered
        assert set(inner._loaded_states) == set(levels)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_wrappers.py::TestMultiLevelWrapper -v`
Expected: FAIL

- [ ] **Step 3: Implement MultiLevelWrapper**

Add to `src/golds/environments/retro/wrappers.py`:

```python
class MultiLevelWrapper(gym.Wrapper):
    """Randomly select a level on each episode reset.

    On ``reset()``, picks a random level from the configured list and sets
    the retro emulator's ``initial_state`` so the subsequent reset loads
    the correct level.
    """

    def __init__(self, env: gym.Env, levels: list[str]) -> None:
        super().__init__(env)
        if not levels:
            raise ValueError("levels must be a non-empty list")
        self.levels = levels
        self._rng = np.random.default_rng()

    def reset(self, **kwargs) -> tuple:
        level = self._rng.choice(self.levels)
        inner = self.env.unwrapped
        inner.load_state(level)
        inner.initial_state = inner.em.get_state()
        return self.env.reset(**kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_wrappers.py::TestMultiLevelWrapper -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/golds/environments/retro/wrappers.py tests/test_wrappers.py
git commit -m "feat: add MultiLevelWrapper for level rotation"
```

---

### Task 4: Schema + Config Pipeline

**Files:**
- Modify: `src/golds/config/schema.py`
- Modify: `src/golds/environments/retro/maker.py`
- Modify: `src/golds/environments/atari/maker.py`
- Modify: `src/golds/training/trainer.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Add new fields to EnvironmentConfig**

In `src/golds/config/schema.py`, add after `max_episode_steps`:

```python
    action_set: Literal["full", "platformer", "fighter", "puzzle"] = Field(
        default="full",
        description="Action space reduction set. 'full' uses all filtered actions.",
    )
    sticky_action_prob: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Probability of repeating previous action (0 = disabled, 0.25 = standard).",
    )
    levels: list[str] = Field(
        default_factory=list,
        description="Level rotation list for multi-level training. Empty = single default level.",
    )
    death_penalty: float = Field(
        default=0.0,
        le=0,
        description="Penalty applied on episode termination (should be negative or 0).",
    )
    collectible_reward_scale: float = Field(
        default=0.0,
        ge=0,
        description="Scale for collectible bonuses (rings/coins). 0 = disabled.",
    )
    time_penalty: float = Field(
        default=0.0,
        le=0,
        description="Per-step penalty to encourage speed (should be negative or 0).",
    )
```

- [ ] **Step 2: Add RND fields to TrainingConfig**

In `src/golds/config/schema.py`, add after `self_play_sampling`:

```python
    rnd_enabled: bool = Field(
        default=False,
        description="Enable RND (Random Network Distillation) intrinsic reward.",
    )
    rnd_reward_scale: float = Field(
        default=0.01,
        ge=0,
        description="Scale factor for RND intrinsic reward.",
    )
    rnd_learning_rate: float = Field(
        default=1e-4,
        ge=0,
        description="Learning rate for RND predictor network.",
    )
```

- [ ] **Step 3: Update maker.py allowed set and make_retro_env signature**

In `src/golds/environments/retro/maker.py`, update `make_retro_env()` to accept new params:

```python
def make_retro_env(
    game: str,
    state: str | None = None,
    screen_size: int = 84,
    grayscale: bool = True,
    clip_reward: bool = True,
    frame_skip: int = 4,
    players: int = 1,
    opponent_mode: str = "none",
    opponent_model_path: str | None = None,
    opponent_snapshot_dir: str | None = None,
    opponent_reload_interval_steps: int = 500,
    x_pos_reward_scale: float = 0.0,
    max_episode_steps: int = 0,
    action_set: str = "full",
    sticky_action_prob: float = 0.0,
    levels: list | None = None,
    death_penalty: float = 0.0,
    collectible_reward_scale: float = 0.0,
    time_penalty: float = 0.0,
) -> gym.Env:
```

Rewrite the wrapper application section to match the spec ordering:

```python
    # Create base environment
    env = retro.make(...)

    # 1. Action space reduction (closest to raw env)
    if action_set != "full":
        from golds.environments.retro.wrappers import DiscreteActionWrapper
        env = DiscreteActionWrapper(env, action_set=action_set)

    # 2. Sticky actions
    if sticky_action_prob > 0:
        from golds.environments.retro.wrappers import StickyActionWrapper
        env = StickyActionWrapper(env, stickprob=sticky_action_prob)

    # 3. Multi-level rotation
    if levels:
        from golds.environments.retro.wrappers import MultiLevelWrapper
        env = MultiLevelWrapper(env, levels=levels)

    # 4. Reward shaping (before Monitor so episode stats include shaped rewards)
    has_shaping = x_pos_reward_scale > 0 or death_penalty != 0 or collectible_reward_scale > 0 or time_penalty != 0
    if has_shaping:
        from golds.environments.retro.wrappers import PlatformerRewardWrapper
        env = PlatformerRewardWrapper(
            env, scale=x_pos_reward_scale, game=game,
            death_penalty=death_penalty,
            collectible_reward_scale=collectible_reward_scale,
            time_penalty=time_penalty,
        )

    # 5. Time limit for fighting games
    if max_episode_steps > 0:
        from golds.environments.retro.wrappers import TimeLimitWrapper
        env = TimeLimitWrapper(env, max_steps=max_episode_steps)

    # 6. Monitor AFTER reward shaping so info["episode"] captures shaped rewards
    env = Monitor(env)

    # 7. Frame skipping
    if frame_skip > 1:
        env = FrameSkip(env, skip=frame_skip)

    # 8. Preprocessing
    env = RetroPreprocessing(env, screen_size=screen_size, grayscale=grayscale, clip_reward=clip_reward)
```

Update `allowed` set in `make_retro_vec_env()`:

```python
        allowed = {
            "screen_size", "grayscale", "clip_reward", "frame_skip",
            "x_pos_reward_scale", "max_episode_steps",
            "action_set", "sticky_action_prob", "levels",
            "death_penalty", "collectible_reward_scale", "time_penalty",
        }
```

Update `default_kwargs` similarly.

- [ ] **Step 4: Update Atari maker allowed set**

In `src/golds/environments/atari/maker.py`, verify `_atari_allowed` does NOT include any of the new retro fields. No changes needed if the filter is already allowlist-based (it is).

- [ ] **Step 5: Update trainer wrapper_kwargs**

In `src/golds/training/trainer.py`, update `_create_train_env()`:

```python
            wrapper_kwargs={
                "terminal_on_life_loss": env_config.terminal_on_life_loss,
                "clip_reward": env_config.clip_reward,
                "x_pos_reward_scale": env_config.x_pos_reward_scale,
                "max_episode_steps": env_config.max_episode_steps,
                "action_set": env_config.action_set,
                "sticky_action_prob": env_config.sticky_action_prob,
                "levels": env_config.levels if env_config.levels else None,
                "death_penalty": env_config.death_penalty,
                "collectible_reward_scale": env_config.collectible_reward_scale,
                "time_penalty": env_config.time_penalty,
            },
```

Update `_create_eval_env()` similarly but with `sticky_action_prob` overridden to `0`:

```python
            wrapper_kwargs={
                "terminal_on_life_loss": False,
                "clip_reward": env_config.clip_reward,
                "x_pos_reward_scale": env_config.x_pos_reward_scale,
                "max_episode_steps": env_config.max_episode_steps,
                "action_set": env_config.action_set,
                "sticky_action_prob": 0.0,  # No sticky actions during eval
                "levels": env_config.levels if env_config.levels else None,
                "death_penalty": env_config.death_penalty,
                "collectible_reward_scale": env_config.collectible_reward_scale,
                "time_penalty": env_config.time_penalty,
            },
```

- [ ] **Step 6: Write a config validation test**

Add to `tests/test_config.py`:

```python
def test_environment_config_new_v3_fields():
    from golds.config.schema import EnvironmentConfig
    cfg = EnvironmentConfig(
        platform="retro",
        game_id="sonic_the_hedgehog",
        action_set="platformer",
        sticky_action_prob=0.25,
        levels=["GreenHillZone.Act1", "GreenHillZone.Act2"],
        death_penalty=-1.0,
        collectible_reward_scale=0.01,
        time_penalty=-0.001,
    )
    assert cfg.action_set == "platformer"
    assert cfg.sticky_action_prob == 0.25
    assert len(cfg.levels) == 2
    assert cfg.death_penalty == -1.0

def test_training_config_rnd_fields():
    from golds.config.schema import TrainingConfig
    cfg = TrainingConfig(rnd_enabled=True, rnd_reward_scale=0.05, rnd_learning_rate=5e-5)
    assert cfg.rnd_enabled is True
    assert cfg.rnd_reward_scale == 0.05
```

- [ ] **Step 7: Run all tests**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 8: Run linter**

Run: `uv run ruff check src/ tests/`
Expected: Clean

- [ ] **Step 9: Commit**

```bash
git add src/golds/config/schema.py src/golds/environments/retro/maker.py src/golds/environments/atari/maker.py src/golds/training/trainer.py tests/test_config.py
git commit -m "feat: wire action_set, sticky_action_prob, levels, reward shaping through config pipeline"
```

---

## Chunk 2: Extended Reward Shaping + Game Configs

### Task 5: Extend PlatformerRewardWrapper

**Files:**
- Modify: `src/golds/environments/retro/wrappers.py`
- Modify: `tests/test_wrappers.py`

- [ ] **Step 1: Write failing tests for extended shaping**

Add to `tests/test_wrappers.py`:

```python
class TestPlatformerRewardWrapperExtended:
    def test_death_penalty_on_terminated(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        class _TerminatingEnv(_FakeRetroEnv):
            def __init__(self):
                super().__init__(info_sequence=[{"x": 0}, {"x": 0}])
            def step(self, action):
                obs, reward, _, truncated, info = super().step(action)
                return obs, reward, True, truncated, info  # terminated=True

        env = PlatformerRewardWrapper(
            _TerminatingEnv(), scale=0.0, game="SonicTheHedgehog-Genesis",
            death_penalty=-1.0,
        )
        env.reset()
        _, reward, _, _, info = env.step(0)
        assert info["shaped_death"] == -1.0
        assert reward == pytest.approx(1.0 + (-1.0))  # raw + death

    def test_no_death_penalty_on_truncated(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        class _TruncatingEnv(_FakeRetroEnv):
            def __init__(self):
                super().__init__(info_sequence=[{"x": 0}, {"x": 0}])
            def step(self, action):
                obs, reward, terminated, _, info = super().step(action)
                return obs, reward, terminated, True, info  # truncated=True

        env = PlatformerRewardWrapper(
            _TruncatingEnv(), scale=0.0, game="SonicTheHedgehog-Genesis",
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
            _FakeRetroEnv(infos), scale=0.0, game="SonicTheHedgehog-Genesis",
            collectible_reward_scale=0.1,
        )
        env.reset()
        _, reward, _, _, info = env.step(0)
        # 5 new rings * 0.1 = 0.5
        assert info["shaped_collectible"] == pytest.approx(0.5)
        assert reward == pytest.approx(1.0 + 0.5)

    def test_time_penalty(self):
        from golds.environments.retro.wrappers import PlatformerRewardWrapper

        infos = [{"x": 0}, {"x": 0}]
        env = PlatformerRewardWrapper(
            _FakeRetroEnv(infos), scale=0.0, game="SonicTheHedgehog-Genesis",
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
            _FakeRetroEnv(infos), scale=0.1, game="SonicTheHedgehog-Genesis",
            death_penalty=-1.0,
            collectible_reward_scale=0.1,
            time_penalty=-0.001,
        )
        env.reset()
        _, reward, _, _, info = env.step(0)
        expected = 1.0 + (0.1 * 10) + (0.1 * 3) + (-0.001) + 0.0  # raw + x + collect + time + no death
        assert reward == pytest.approx(expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_wrappers.py::TestPlatformerRewardWrapperExtended -v`
Expected: FAIL (PlatformerRewardWrapper doesn't accept new params yet)

- [ ] **Step 3: Extend PlatformerRewardWrapper implementation**

Update `PlatformerRewardWrapper.__init__` and `step` in `src/golds/environments/retro/wrappers.py`:

```python
    _COLLECTIBLE_EXTRACTORS: dict[str, Any] = {
        "Sonic": lambda info: info.get("rings", 0),
        "SuperMarioBros": lambda info: info.get("coins", 0),
    }

    def __init__(
        self,
        env: gym.Env,
        scale: float = 0.1,
        game: str = "",
        death_penalty: float = 0.0,
        collectible_reward_scale: float = 0.0,
        time_penalty: float = 0.0,
    ) -> None:
        super().__init__(env)
        self.scale = scale
        self.death_penalty = death_penalty
        self.collectible_reward_scale = collectible_reward_scale
        self.time_penalty = time_penalty
        self._x_old: float = 0.0
        self._collectible_old: float = 0.0
        self._extractor = self._resolve_extractor(game)
        self._collectible_extractor = self._resolve_collectible_extractor(game)

    @classmethod
    def _resolve_collectible_extractor(cls, game: str):
        for prefix, fn in cls._COLLECTIBLE_EXTRACTORS.items():
            if game.startswith(prefix):
                return fn
        return lambda info: 0  # no collectibles by default

    def step(self, action: Any) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # X-position shaping
        x_new = float(self._extractor(info))
        x_shaped = self.scale * (x_new - self._x_old)
        self._x_old = x_new

        # Death penalty (only on true termination, not truncation)
        death = self.death_penalty if terminated and not truncated else 0.0

        # Collectible bonus (delta)
        collect_new = float(self._collectible_extractor(info))
        collect_delta = max(0.0, collect_new - self._collectible_old)
        collect_shaped = self.collectible_reward_scale * collect_delta
        self._collectible_old = collect_new

        # Time penalty
        time_shaped = self.time_penalty

        total_shaped = x_shaped + death + collect_shaped + time_shaped

        info["raw_reward"] = reward
        info["shaped_x_progress"] = x_shaped
        info["shaped_death"] = death
        info["shaped_collectible"] = collect_shaped
        info["shaped_time"] = time_shaped
        # Keep backward-compatible key
        info["shaped_reward"] = total_shaped

        return obs, reward + total_shaped, terminated, truncated, info

    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)
        self._x_old = float(self._extractor(info))
        self._collectible_old = float(self._collectible_extractor(info))
        return obs, info
```

- [ ] **Step 4: Run all wrapper tests**

Run: `uv run pytest tests/test_wrappers.py -v`
Expected: All pass (including old tests that use the existing API)

- [ ] **Step 5: Commit**

```bash
git add src/golds/environments/retro/wrappers.py tests/test_wrappers.py
git commit -m "feat: extend PlatformerRewardWrapper with death/collectible/time signals"
```

---

### Task 6: Update Game Configs

**Files:**
- Modify: All `configs/games/*.yaml` files

- [ ] **Step 1: Update platformer configs**

For `sonic_the_hedgehog.yaml`, `super_mario_bros.yaml`, `super_mario_bros_2_japan.yaml`, `mega_man_2.yaml`:

```yaml
environment:
  clip_reward: false
  reward_regime: raw
  action_set: platformer
  sticky_action_prob: 0.25
  x_pos_reward_scale: 0.1
  death_penalty: -1.0
  collectible_reward_scale: 0.01
  time_penalty: -0.001

ppo:
  learning_rate: 3e-4
  n_steps: 512
  clip_range: 0.2
  lr_schedule: linear
  clip_schedule: linear
```

Set `total_timesteps: 100000000` for Sonic, Mario 2 Japan, Mega Man 2.
Set `total_timesteps: 50000000` for Mario 1.

- [ ] **Step 2: Update fighting game configs**

For `mortal_kombat_ii.yaml`, `street_fighter_ii.yaml`:

```yaml
environment:
  action_set: fighter
  sticky_action_prob: 0.25

ppo:
  learning_rate: 3e-4
  n_steps: 512
  clip_range: 0.2
```

- [ ] **Step 3: Update puzzle config**

For `tetris.yaml`:

```yaml
environment:
  action_set: puzzle
  sticky_action_prob: 0.25

ppo:
  learning_rate: 3e-4
  n_steps: 512
  clip_range: 0.2
```

- [ ] **Step 4: Update Atari configs**

For `pong.yaml`, `breakout.yaml`, `space_invaders.yaml`, `ms_pacman.yaml`, `enduro.yaml`, `frostbite.yaml`, `montezuma_revenge.yaml`:

```yaml
ppo:
  learning_rate: 3e-4
  n_steps: 512
  clip_range: 0.2
  # Keep lr_schedule and clip_schedule as linear (already set)
```

- [ ] **Step 5: Run config validation**

Run: `uv run pytest tests/test_config.py tests/test_config_loader.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add configs/games/
git commit -m "config: v3 hyperparameters — action sets, sticky actions, extended reward shaping"
```

---

## Chunk 3: RND Exploration

### Task 7: RND Module

**Files:**
- Create: `src/golds/training/rnd.py`
- Create: `tests/test_rnd.py`

- [ ] **Step 1: Write failing tests for RND networks**

Create `tests/test_rnd.py`:

```python
"""Tests for RND (Random Network Distillation) module."""
import numpy as np
import pytest
import torch


class TestRNDNetwork:
    def test_forward_shape(self):
        from golds.training.rnd import RNDNetwork

        net = RNDNetwork()
        x = torch.randn(4, 1, 84, 84)  # batch=4, single grayscale frame
        out = net(x)
        assert out.shape == (4, 512)

    def test_target_and_predictor_differ(self):
        from golds.training.rnd import RNDNetwork

        target = RNDNetwork()
        predictor = RNDNetwork()
        x = torch.randn(2, 1, 84, 84)
        # Different random inits should give different outputs
        t_out = target(x)
        p_out = predictor(x)
        assert not torch.allclose(t_out, p_out)


class TestRunningMeanStd:
    def test_tracks_mean_and_var(self):
        from golds.training.rnd import RunningMeanStd

        rms = RunningMeanStd()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rms.update(data)
        assert rms.mean == pytest.approx(3.0, abs=0.01)
        assert rms.var == pytest.approx(2.0, abs=0.1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rnd.py -v`
Expected: FAIL

- [ ] **Step 3: Implement RND networks and RunningMeanStd**

Create `src/golds/training/rnd.py`:

```python
"""Random Network Distillation (RND) for intrinsic exploration reward."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import VecEnvWrapper


class RNDNetwork(nn.Module):
    """Small CNN that maps (1, 84, 84) grayscale frames to 512-dim embeddings."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RunningMeanStd:
    """Welford online mean/variance tracker."""

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = len(x)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.count = total


class RNDRewardWrapper(VecEnvWrapper):
    """VecEnv wrapper that adds RND intrinsic reward inline during step().

    Must be applied after VecFrameStack and VecTransposeImage so observations
    are in (C, H, W) format. Extracts a single frame from the stack for RND.
    """

    def __init__(
        self,
        venv,
        scale: float = 0.01,
        learning_rate: float = 1e-4,
        device: str = "auto",
    ) -> None:
        super().__init__(venv)
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self.target = RNDNetwork().to(self._device)
        self.predictor = RNDNetwork().to(self._device)
        self.scale = scale

        # Freeze target
        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        self.reward_rms = RunningMeanStd()
        self.obs_rms = RunningMeanStd()

    def _preprocess(self, obs: np.ndarray) -> torch.Tensor:
        """Extract last frame from stack, normalize to [0,1], send to device."""
        # obs shape: (n_envs, C, H, W) where C = frame_stack (e.g. 4)
        # Take last channel only for RND
        single = obs[:, -1:, :, :].astype(np.float32) / 255.0
        # Normalize with running stats
        self.obs_rms.update(single.reshape(-1))
        single = (single - self.obs_rms.mean) / (np.sqrt(self.obs_rms.var) + 1e-8)
        return torch.from_numpy(single).to(self._device)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        obs_t = self._preprocess(obs)

        # Compute intrinsic reward
        with torch.no_grad():
            target_feat = self.target(obs_t)
            pred_feat = self.predictor(obs_t)
            intrinsic = ((target_feat - pred_feat) ** 2).mean(dim=1).cpu().numpy()

        # Normalize
        self.reward_rms.update(intrinsic)
        normalized = intrinsic / (np.sqrt(self.reward_rms.var) + 1e-8)
        shaped = self.scale * normalized

        # Train predictor (small update each step)
        pred_feat_grad = self.predictor(obs_t)
        with torch.no_grad():
            target_feat_stable = self.target(obs_t)
        loss = ((pred_feat_grad - target_feat_stable) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Add to rewards and log
        for i, info in enumerate(infos):
            info["rnd_intrinsic_reward"] = float(shaped[i])

        return obs, rewards + shaped, dones, infos

    def reset(self):
        return self.venv.reset()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rnd.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/golds/training/rnd.py tests/test_rnd.py
git commit -m "feat: add RND module for intrinsic exploration reward"
```

---

### Task 8: Wire RND into Factory/Trainer

**Files:**
- Modify: `src/golds/environments/factory.py`
- Modify: `src/golds/training/trainer.py`

- [ ] **Step 1: Add RND wrapper to EnvironmentFactory**

In `src/golds/environments/factory.py`, add RND application after the reward normalization block and before the self-play wrapper:

```python
        # Optional RND exploration reward
        if kwargs.get("rnd_enabled", False):
            from golds.training.rnd import RNDRewardWrapper
            rnd_scale = kwargs.get("rnd_reward_scale", 0.01)
            rnd_lr = kwargs.get("rnd_learning_rate", 1e-4)
            rnd_device = kwargs.get("device", "auto")
            vec_env = RNDRewardWrapper(vec_env, scale=rnd_scale, learning_rate=rnd_lr, device=rnd_device)
```

- [ ] **Step 2: Pass RND config from Trainer to Factory**

In `src/golds/training/trainer.py`, update the `EnvironmentFactory.create()` call in `_create_train_env()` to pass RND config:

```python
        train_env = EnvironmentFactory.create(
            ...,
            rnd_enabled=self.config.training.rnd_enabled,
            rnd_reward_scale=self.config.training.rnd_reward_scale,
            rnd_learning_rate=self.config.training.rnd_learning_rate,
            device=self.config.training.device,
        )
```

Do NOT pass `rnd_enabled` to eval env (no exploration bonus during evaluation).

- [ ] **Step 3: Run all tests**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 4: Run linter**

Run: `uv run ruff check src/ tests/`
Expected: Clean

- [ ] **Step 5: Commit**

```bash
git add src/golds/environments/factory.py src/golds/training/trainer.py
git commit -m "feat: wire RND reward wrapper into factory and trainer"
```

---

## Chunk 4: Documentation + Notebooks

### Task 9: Update QUICKSTART.md

**Files:**
- Modify: `docs/QUICKSTART.md`

- [ ] **Step 1: Add Action Sets section after "Override defaults"**

```markdown
### Action Sets

Retro games use reduced action sets for faster learning:

    action_set: platformer   # 9 actions: move, jump, run, duck
    action_set: fighter       # 12 actions: move, punch, kick, combos
    action_set: puzzle        # 5 actions: move, rotate
    action_set: full          # All filtered actions (default)

Set in your game config under `environment:`.
```

- [ ] **Step 2: Add Advanced sections before "System Check"**

```markdown
### Advanced: Multi-Level Training

Train across multiple levels simultaneously for better generalization:

```yaml
environment:
  levels:
    - GreenHillZone.Act1
    - GreenHillZone.Act2
    - MarbleZone.Act1
```

Each parallel environment randomly selects a level on reset.

### Advanced: Exploration Bonus (RND)

Enable Random Network Distillation for exploration in sparse-reward games:

```yaml
training:
  rnd_enabled: true
  rnd_reward_scale: 0.01
```
```

- [ ] **Step 3: Update Common Issues**

Add: `**"Agent gets stuck"** -- Try enabling sticky actions (sticky_action_prob: 0.25) and reducing the action space (action_set: platformer). Also consider enabling RND exploration.`

Update model path pattern: remove the nested `/<game_name>/` since output dirs are now unified.

- [ ] **Step 4: Commit**

```bash
git add docs/QUICKSTART.md
git commit -m "docs: update QUICKSTART for v3 config fields"
```

---

### Task 10: Update Notebook 02 (Environment Pipeline)

**Files:**
- Modify: `notebooks/02_environment_pipeline.ipynb`

- [ ] **Step 1: Add DiscreteActionWrapper section after cell 22 (Button-to-MultiBinary)**

New markdown cell:

```markdown
### 3.4 Action Space Reduction — DiscreteActionWrapper

The MultiBinary action space gives the agent all possible button combinations — including nonsensical ones like pressing Up+Down simultaneously. For a Genesis game that's $2^{12} = 4096$ possible actions. Even with `Actions.FILTERED`, we get ~126.

GOLDS's `DiscreteActionWrapper` reduces this to just the meaningful combinations...
```

New code cell demonstrating the action mapping table and showing before/after action space sizes.

- [ ] **Step 2: Add StickyActionWrapper section**

New markdown cell explaining stochasticity + new code cell showing probability demo.

- [ ] **Step 3: Add Multi-Level Rotation section**

New markdown cell + code cell simulating level distribution across N envs.

- [ ] **Step 4: Update the pipeline diagram in the Summary section**

Update the ASCII diagram to show the full v3 stack with corrected Monitor placement.

- [ ] **Step 5: Commit**

```bash
git add notebooks/02_environment_pipeline.ipynb
git commit -m "docs: add v3 wrapper sections to environment pipeline notebook"
```

---

### Task 11: Update Notebook 04 (Reward Engineering)

**Files:**
- Modify: `notebooks/04_reward_engineering.ipynb`

- [ ] **Step 1: Extend Custom Reward Wrappers section**

Add cells showing death penalty, collectible bonus, and time penalty examples. Show the full `PlatformerRewardWrapper` with all 4 signals active, including the debug info dict output.

- [ ] **Step 2: Add new "Intrinsic Curiosity and RND" section**

New markdown cell explaining RND architecture (target vs predictor), when to use it, VecEnvWrapper integration. New code cell with a diagram of the two networks and a synthetic demo of intrinsic reward decaying as the predictor learns.

- [ ] **Step 3: Update GOLDS Reward Regimes table**

Add new config fields to the table. Add note about `clip_reward: false` being required for reward shaping.

- [ ] **Step 4: Commit**

```bash
git add notebooks/04_reward_engineering.ipynb
git commit -m "docs: add extended reward shaping and RND sections to reward engineering notebook"
```

---

## Chunk 5: Final Verification

### Task 12: Integration Test + Lint

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All tests pass

- [ ] **Step 2: Run linter**

Run: `uv run ruff check src/ tests/`
Expected: Clean

- [ ] **Step 3: Verify wrapper import**

Run: `uv run python -c "from golds.environments.retro.wrappers import DiscreteActionWrapper, StickyActionWrapper, MultiLevelWrapper; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Verify RND import**

Run: `uv run python -c "from golds.training.rnd import RNDRewardWrapper, RNDNetwork; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Verify config loads with new fields**

Run: `uv run python -c "from golds.config.loader import ConfigLoader; c = ConfigLoader(); cfg = c.load_game('sonic_the_hedgehog'); print(f'action_set={cfg.environment.action_set}, sticky={cfg.environment.sticky_action_prob}')"`
Expected: `action_set=platformer, sticky=0.25`

- [ ] **Step 6: Final commit if any fixups needed**

```bash
git add -A && git commit -m "fix: final v3 integration fixups"
```
