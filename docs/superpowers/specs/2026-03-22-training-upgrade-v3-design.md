# GOLDS Training Upgrade v3 — Design Spec

**Date:** 2026-03-22
**Status:** Approved
**Author:** Xavier Briggs + Claude

## Problem Statement

Round 2 training (50M steps) produced agents for Mario and Sonic that move rightward but get stuck at obstacles and never complete levels. Root causes identified through research:

1. **Action space too large** — `retro.Actions.FILTERED` exposes ~126 button combinations. The agent wastes sample budget on useless combos (e.g., Up+Down+B+A simultaneously).
2. **No stochasticity** — Retro games are fully deterministic. Without noise, agents memorize frame-perfect action sequences instead of learning reactive policies from observations.
3. **Reward shaping is x-position only** — No penalty for death, no bonus for collectibles, no time pressure. Agent learns "move right" but not "survive while moving right."
4. **Single-level training** — Agent overfits to one level's layout instead of learning generalizable platformer skills.
5. **No exploration bonus** — When the agent reaches a hard section (gap, enemy), the reward signal goes flat and the agent has no incentive to try new strategies.

## Research Basis

Findings from the OpenAI Retro Contest (2018), ICM/RND exploration papers, and SB3 community best practices:

- **Retro Contest winners** (Dharmaraja, Aborg) used Joint PPO across multiple levels, reduced action spaces, and novelty-based reward bonuses.
- **Action space reduction** from 126 to 7-12 actions is the single highest-impact change for platformers.
- **Sticky actions** (stickprob=0.25) are standard practice for breaking determinism in retro games.
- **RND** (Random Network Distillation) achieved superhuman on Montezuma's Revenge and works for Sonic, with caveats around dynamic backgrounds.
- **CnnPolicy + frame stacking** outperforms CnnLstmPolicy in practice for platformers.
- **100M+ timesteps** typically needed for reliable level completion in Sonic/Mario.

## Design

### 1. Action Space Reduction

**New wrapper:** `DiscreteActionWrapper` in `src/golds/environments/retro/wrappers.py`

Maps a small `Discrete(N)` action space to the `MultiBinary` button presses that retro expects. Each action index corresponds to a meaningful button combination.

**Action sets:**

| Set | N | Actions |
|-----|---|---------|
| `platformer` | 9 | NOOP, Left, Right, Down, Right+B (run/spin), Down+B (spin dash), A (jump), Right+A (jump forward), Right+A+B (running jump) |
| `fighter` | 12 | NOOP, Left, Right, Up, Down, A, B, Down+A, Down+B, Left+A, Right+A, Right+B |
| `puzzle` | 5 | NOOP, Left, Right, Down, A (rotate) |
| `full` | N/A | No wrapper applied; uses all filtered actions (current behavior) |

**Note:** The `platformer` set includes Down and Down+B because ducking/rolling is essential for Sonic (rolling through tunnels, spin dash) and crouching is used in Mario for entering pipes.

**Button index mapping:** The wrapper reads `env.unwrapped.buttons` to get the button names for the current console, then builds the multi-binary vectors at init time. This handles NES (8 buttons) vs Genesis (12 buttons) automatically. Tests must cover both button layouts using a fake env with a configurable `buttons` attribute.

**Config:** `action_set: str = "full"` in `EnvironmentConfig`.

### 2. Sticky Actions

**New wrapper:** `StickyActionWrapper` in `src/golds/environments/retro/wrappers.py`

With probability `p`, repeats the previous action instead of executing the new one. Breaks determinism so the agent cannot memorize frame-perfect sequences.

```python
class StickyActionWrapper(gym.Wrapper):
    def __init__(self, env, stickprob=0.25):
        super().__init__(env)
        self.stickprob = stickprob
        self._last_action = None
        self._rng = np.random.default_rng()  # independent RNG, seeded on reset

    def step(self, action):
        if self._last_action is not None and self._rng.random() < self.stickprob:
            action = self._last_action
        self._last_action = action
        return self.env.step(action)

    def reset(self, **kwargs):
        self._last_action = None
        seed = kwargs.get("seed")
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return self.env.reset(**kwargs)
```

**Note:** Uses an independent `np.random.default_rng()` initialized in `__init__` instead of `self.np_random` to avoid `AttributeError` if `step()` is called before `reset(seed=...)`. The RNG is re-seeded on `reset(seed=...)` for reproducibility.

**Config:** `sticky_action_prob: float = 0.0` in `EnvironmentConfig`. Set to `0.25` for all retro games. Atari games already have stochasticity via NoopReset, so leave at `0.0`.

### 3. Multi-Level Rotation

**New wrapper:** `MultiLevelWrapper` in `src/golds/environments/retro/wrappers.py`

On each `reset()`, picks a random level from a configured list. The wrapper sets the retro env's `initial_state` property so the subsequent `reset()` loads the correct level instead of the default.

```python
class MultiLevelWrapper(gym.Wrapper):
    def __init__(self, env, levels):
        super().__init__(env)
        self.levels = levels
        self._rng = np.random.default_rng()

    def reset(self, **kwargs):
        level = self._rng.choice(self.levels)
        # Set the initial state so reset() loads the chosen level.
        # stable-retro's RetroEnv.reset() calls em.set_state(self.initial_state).
        # By loading the level into the emulator and snapshotting,
        # we ensure reset() restores to the right level.
        inner = self.env.unwrapped
        inner.load_state(level)
        inner.initial_state = inner.em.get_state()
        return self.env.reset(**kwargs)
```

**Note on load_state + reset interaction:** `retro.RetroEnv.reset()` restores from `self.initial_state`. If we only called `load_state()` and then `reset()`, the reset would overwrite the loaded state with the original default. The fix is to snapshot the emulator state after `load_state()` and assign it to `initial_state`, so the subsequent `reset()` restores to the correct level. This was identified as a critical correctness issue in review.

When `levels` is set, each env in the vectorized pool picks independently, so 24 parallel envs naturally cover different levels.

**Config:** `levels: list[str] = []` in `EnvironmentConfig`. Empty list means single default level (current behavior).

**Example Sonic config:**
```yaml
environment:
  levels:
    - GreenHillZone.Act1
    - GreenHillZone.Act2
    - GreenHillZone.Act3
    - MarbleZone.Act1
    - MarbleZone.Act2
    - MarbleZone.Act3
```

**Example Mario config:**
```yaml
environment:
  levels:
    - Level1-1
    - Level1-2
    - Level1-3
    - Level2-1
```

### 4. Intrinsic Curiosity (RND)

**New module:** `src/golds/training/rnd.py`

RND adds an exploration bonus by measuring how "surprising" each observation is. It uses two small CNNs:
- **Target network** — fixed random weights, never trained
- **Predictor network** — trained to match target's output

Intrinsic reward = MSE between predictor and target outputs. Novel states produce high error (high reward); familiar states produce low error.

**Architecture:**
```
Input: (84, 84, 1) grayscale frame (single frame, not stacked)
  → Conv2d(1, 32, 8, stride=4) → ReLU
  → Conv2d(32, 64, 4, stride=2) → ReLU
  → Conv2d(64, 64, 3, stride=1) → ReLU
  → Flatten → Linear(3136, 512)
Output: 512-dim embedding
```

Both networks share the same architecture. Only the predictor is trained.

**Integration via `RNDRewardWrapper` (VecEnv wrapper):**

The review identified that SB3's `PPO.collect_rollouts()` computes advantages immediately after storing rewards, before any `_on_rollout_end` callback fires. Therefore, a callback-based approach cannot modify rewards in time. Instead, RND is implemented as a `VecEnvWrapper` that adds intrinsic rewards inline during `step()`:

```python
class RNDRewardWrapper(VecEnvWrapper):
    """Adds RND intrinsic reward to extrinsic reward inline during step()."""

    def __init__(self, venv, target_net, predictor_net, scale=0.01, lr=1e-4):
        super().__init__(venv)
        self.target = target_net      # fixed random CNN
        self.predictor = predictor_net  # trained CNN
        self.scale = scale
        self.optimizer = torch.optim.Adam(predictor_net.parameters(), lr=lr)
        self.reward_rms = RunningMeanStd()  # for intrinsic reward normalization

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        # Compute intrinsic reward
        with torch.no_grad():
            obs_t = self._preprocess(obs)  # extract single frame, normalize
            target_feat = self.target(obs_t)
            pred_feat = self.predictor(obs_t)
            intrinsic = ((target_feat - pred_feat) ** 2).mean(dim=1).cpu().numpy()
        # Normalize and scale
        self.reward_rms.update(intrinsic)
        normalized = intrinsic / (np.sqrt(self.reward_rms.var) + 1e-8)
        shaped = self.scale * normalized
        # Train predictor
        self._train_predictor(obs_t, target_feat)
        # Add to rewards
        for i, info in enumerate(infos):
            info["rnd_intrinsic_reward"] = shaped[i]
        return obs, rewards + shaped, dones, infos
```

This wrapper is applied in `EnvironmentFactory.create()` after all other wrappers, similar to how `VecNormalize` and `VecTwoPlayerOpponentWrapper` are applied.

**Config fields in `TrainingConfig`:**
```yaml
rnd_enabled: bool = false
rnd_reward_scale: float = 0.01
rnd_learning_rate: float = 1e-4
```

**Sonic caveat:** Dynamic backgrounds (timer, clouds) produce spurious intrinsic reward. The small scale factor (0.01) and reward normalization mitigate this. If needed, can be lowered further.

### 5. Extended Reward Shaping

Extend `PlatformerRewardWrapper` with additional signals:

| Signal | Config field | Default | Description |
|--------|-------------|---------|-------------|
| X-position progress | `x_pos_reward_scale` | 0.0 | `scale * (x_new - x_old)` — existing |
| Death penalty | `death_penalty` | 0.0 | Applied on `terminated=True` (not truncation) |
| Collectible bonus | `collectible_reward_scale` | 0.0 | Scale for rings (Sonic) or coins (Mario) from info dict |
| Time penalty | `time_penalty` | 0.0 | Per-step negative reward to encourage speed |

**Collectible info dict keys** (from stable-retro's `data.json` files):
- Sonic: `info["rings"]` — ring count
- Mario: `info["coins"]` — coin count
- Mega Man: not applicable (no collectible counter in data.json)

The collectible extractor follows the same `_EXTRACTORS` pattern as x-position:
```python
_COLLECTIBLE_EXTRACTORS = {
    "Sonic": lambda info: info.get("rings", 0),
    "SuperMarioBros": lambda info: info.get("coins", 0),
}
```

All signals tracked separately in `info` dict for debugging:
```python
info["raw_reward"] = original_reward
info["shaped_x_progress"] = x_delta
info["shaped_death"] = death_penalty_applied
info["shaped_collectible"] = collectible_bonus
info["shaped_time"] = time_penalty
```

**Config fields in `EnvironmentConfig`:**
```yaml
death_penalty: float = 0.0
collectible_reward_scale: float = 0.0
time_penalty: float = 0.0
```

**Important:** `clip_reward: false` is required for any game using reward shaping. If `clip_reward: true`, `RetroPreprocessing` (which runs after the reward wrapper) will clip all shaped rewards to {-1, 0, +1}, nullifying the shaping. All retro game configs with reward shaping must explicitly set `clip_reward: false`.

### 6. Hyperparameter Updates

**All retro games (changes from v2 defaults noted):**
```yaml
environment:
  sticky_action_prob: 0.25    # NEW: breaks determinism
  clip_reward: false           # REQUIRED: preserves reward shaping magnitudes
  reward_regime: raw           # Already set in v2
  action_set: <genre-appropriate>  # NEW
ppo:
  n_steps: 512                # WAS: 256. Longer rollouts capture more level progress.
  clip_range: 0.2             # WAS: 0.1. Wider clip for longer training runs (standard PPO default).
  learning_rate: 3e-4         # WAS: 2.5e-4. OpenAI Retro Contest default.
```

**Platformers (Mario, Sonic, Mega Man):**
```yaml
environment:
  action_set: platformer
  x_pos_reward_scale: 0.1
  death_penalty: -1.0
  time_penalty: -0.001
  collectible_reward_scale: 0.01  # rings/coins bonus
training:
  total_timesteps: 100000000  # 100M (was 50M for Sonic/Mario2, 20M for Mario1)
  rnd_enabled: true
  rnd_reward_scale: 0.01
```

**Fighting games (MK2, SF2):**
```yaml
environment:
  action_set: fighter
  max_episode_steps: 10000
training:
  rnd_enabled: false          # health delta is dense enough
```

**Puzzle (Tetris):**
```yaml
environment:
  action_set: puzzle
training:
  rnd_enabled: false
```

**Atari games (Pong, Breakout, Space Invaders, Ms Pac-Man):**
```yaml
# Atari games keep clip_reward: true and reward_regime: clipped (standard DeepMind).
# Only PPO hyperparameters change:
ppo:
  n_steps: 512                # WAS: 128. Longer rollouts.
  clip_range: 0.2             # WAS: 0.1.
  learning_rate: 3e-4         # WAS: 2.5e-4.
  # lr_schedule and clip_schedule remain linear (already set in v2).
```

### 7. Wrapper Ordering

Full wrapper stack in `make_retro_env()`:

```
retro.make()
  → DiscreteActionWrapper    (if action_set != "full")
  → StickyActionWrapper      (if sticky_action_prob > 0)
  → MultiLevelWrapper        (if levels is non-empty)
  → PlatformerRewardWrapper  (if any reward shaping enabled)
  → TimeLimitWrapper         (if max_episode_steps > 0)
  → Monitor                  (always — captures shaped episode stats)
  → FrameSkip
  → RetroPreprocessing
```

**Ordering rationale:**
- **Action space wrappers first** (closest to raw env) — they transform the action space all subsequent wrappers see.
- **Reward wrappers before Monitor** — so `info["episode"]["r"]` reflects the fully shaped reward. The previous ordering (Monitor before reward wrappers) was a bug that caused Monitor to record unshaped rewards.
- **Monitor after reward shaping** — captures episode length and shaped cumulative reward for SB3's eval/logging.
- **FrameSkip and RetroPreprocessing last** — observation transforms don't affect reward or action spaces.

**Eval environments:** The eval env must also get `DiscreteActionWrapper` (otherwise the model outputs 9 actions but the env expects 126, causing a shape mismatch). `StickyActionWrapper` should be disabled during eval (`sticky_action_prob=0`) for deterministic evaluation. Reward wrappers are optional for eval but harmless. `MultiLevelWrapper` can be enabled or disabled for eval depending on whether you want per-level or cross-level evaluation.

The `Trainer._create_eval_env()` must pass the same `wrapper_kwargs` as `_create_train_env()`, with `sticky_action_prob` overridden to `0`.

### 8. CnnLstmPolicy

Already supported in schema (`policy: CnnLstmPolicy`). No code changes needed. Research indicates CnnPolicy + frame stacking is better for platformers, so CnnPolicy remains the default. CnnLstmPolicy is available for experimentation via config toggle.

## File Changes

| File | Action | What |
|------|--------|------|
| `src/golds/environments/retro/wrappers.py` | Modify | Add `DiscreteActionWrapper`, `StickyActionWrapper`, `MultiLevelWrapper`. Extend `PlatformerRewardWrapper` with death/collectible/time signals and `_COLLECTIBLE_EXTRACTORS`. |
| `src/golds/environments/retro/maker.py` | Modify | Wire new wrappers in correct order. Move Monitor after reward wrappers. Add new params to `make_retro_env()` signature. Update `allowed` set in `make_retro_vec_env()` to include: `action_set`, `sticky_action_prob`, `levels`, `death_penalty`, `collectible_reward_scale`, `time_penalty`. |
| `src/golds/config/schema.py` | Modify | Add to `EnvironmentConfig`: `action_set`, `sticky_action_prob`, `levels`, `death_penalty`, `collectible_reward_scale`, `time_penalty`. Add to `TrainingConfig`: `rnd_enabled`, `rnd_reward_scale`, `rnd_learning_rate`. |
| `src/golds/training/rnd.py` | Create | `RNDModule` (target + predictor CNNs), `RNDRewardWrapper` (VecEnvWrapper that adds intrinsic rewards inline). |
| `src/golds/training/trainer.py` | Modify | Wire `RNDRewardWrapper` in `_create_train_env()` when `rnd_enabled=true`. Update `wrapper_kwargs` in both `_create_train_env()` and `_create_eval_env()` to include new config fields. Override `sticky_action_prob=0` in eval wrapper_kwargs. |
| `src/golds/environments/factory.py` | Modify | Apply `RNDRewardWrapper` after frame stacking (similar to VecNormalize placement). |
| `src/golds/environments/atari/maker.py` | Verify | Confirm `_atari_allowed` set filters out all new retro-specific fields (`action_set`, `sticky_action_prob`, `levels`, `death_penalty`, `collectible_reward_scale`, `time_penalty`). |
| `configs/games/sonic_the_hedgehog.yaml` | Modify | Full v3 config with action_set, sticky actions, levels, RND, 100M steps. |
| `configs/games/super_mario_bros.yaml` | Modify | Full v3 config. |
| `configs/games/super_mario_bros_2_japan.yaml` | Modify | Full v3 config. |
| `configs/games/mortal_kombat_ii.yaml` | Modify | Fighter action set, sticky actions, updated PPO hyperparams. |
| `configs/games/tetris.yaml` | Modify | Puzzle action set, sticky actions, updated PPO hyperparams. |
| `configs/games/mega_man_2.yaml` | Modify | Platformer action set, sticky actions, updated PPO hyperparams. |
| `configs/games/street_fighter_ii.yaml` | Modify | Fighter action set, sticky actions, updated PPO hyperparams. |
| `configs/games/pong.yaml` | Modify | PPO: n_steps=512, clip_range=0.2, learning_rate=3e-4. |
| `configs/games/breakout.yaml` | Modify | PPO: n_steps=512, clip_range=0.2, learning_rate=3e-4. |
| `configs/games/space_invaders.yaml` | Modify | PPO: n_steps=512, clip_range=0.2, learning_rate=3e-4. |
| `configs/games/ms_pacman.yaml` | Modify | PPO: n_steps=512, clip_range=0.2, learning_rate=3e-4. |
| `tests/test_wrappers.py` | Modify | Tests for `DiscreteActionWrapper` (both NES 8-button and Genesis 12-button layouts), `StickyActionWrapper`, `MultiLevelWrapper`, extended `PlatformerRewardWrapper`. |
| `tests/test_rnd.py` | Create | Tests for `RNDModule` forward pass and `RNDRewardWrapper` reward augmentation. |
| `docs/QUICKSTART.md` | Modify | Update for new config fields, action sets, v3 training recommendations. |
| `notebooks/02_environment_pipeline.ipynb` | Modify | Add sections on action space reduction, sticky actions, multi-level rotation. Update wrapper stack diagram. |
| `notebooks/04_reward_engineering.ipynb` | Modify | Add section on extended reward shaping (death/collectible/time). Update GOLDS reward regime table. Add RND exploration bonus section. |

## Documentation Updates

### QUICKSTART.md Changes

- Add "Action Sets" section explaining `action_set: platformer|fighter|puzzle|full`
- Update "Common Issues" with sticky actions explanation
- Add "Advanced: Multi-Level Training" section
- Add "Advanced: Exploration Bonus (RND)" section
- Update config field reference table with all new fields

### Notebook 02 (Environment Pipeline) Changes

- **After "Button-to-MultiBinary" section**: Add subsection on `DiscreteActionWrapper` — why 126 actions is bad, how we reduce to 9/12/5, code walkthrough with NES vs Genesis examples
- **After "GOLDS Retro Preprocessing" section**: Update wrapper stack diagram to include new wrappers. Add `StickyActionWrapper` explanation with probability demo showing how it breaks determinism.
- **New section "Multi-Level Rotation"**: Why single-level overfits, how `MultiLevelWrapper` works, the `load_state`/`initial_state` pattern, visualization of level distribution across parallel envs
- **"Complete Pipeline" section**: Update the ASCII pipeline diagram to show the full v3 stack with correct Monitor placement

### Notebook 04 (Reward Engineering) Changes

- **"Custom Reward Wrappers" section**: Extend with death penalty, collectible bonus, and time penalty examples. Show the full `PlatformerRewardWrapper` with all signals. Add note about `clip_reward: false` requirement.
- **New section "Intrinsic Curiosity and RND"**: What it is, how it works (target vs predictor network diagram), when to use it, the Sonic dynamic-background caveat, VecEnvWrapper integration pattern
- **"GOLDS Reward Regimes" section**: Update table to include new config fields. Add example showing all reward signals active simultaneously with debug info dict output.

## Rollout Plan

**Phase 1 — Core fixes (immediate):**
Action space reduction + sticky actions + updated hyperparameters. Train Mario and Sonic for 20M steps to validate improvement over v2.

**Phase 2 — Extended shaping:**
Enable death penalty + collectible bonus + time penalty. Compare against Phase 1 results at 20M steps.

**Phase 3 — Exploration:**
Enable RND for platformers. Train to 100M steps.

**Phase 4 — Multi-level:**
Add level rotation configs for Sonic (6 acts) and Mario (4+ levels). Train to 100M steps. This is the biggest change and benefits from Phases 1-3 being validated first.

## Success Criteria

- Mario completes World 1-1 within 100M steps
- Sonic reaches end of GreenHillZone Act 1 within 100M steps
- MK2 self-play agent wins >60% against random opponent
- No regression on Atari games (Pong, Breakout, Space Invaders)
