# Gymnasium & Stable-Retro Ecosystem State (2025)

> **Note:** This document was compiled from knowledge available as of early-mid 2025.
> Verify versions against PyPI and GitHub before pinning in production.
> Some details may have shifted — always cross-check changelogs.

---

## 1. Gymnasium (Farama Foundation)

**Latest version (as of early 2025):** `1.0.0` (stable release reached in late 2024), with patch releases `1.0.x` following.

### Key API changes since 0.29

- **`gymnasium.make()` now returns `gymnasium.Env` (v1 API):** The `v1` API, introduced experimentally in 0.29, became the default in 1.0. The signature of `step()` changed from returning `(obs, reward, done, info)` to `(obs, reward, terminated, truncated, info)` — the `done` boolean was split into `terminated` and `truncated`. This was already available behind `gymnasium.make(..., apply_api_compatibility=False)` since 0.26 but is now the only API.
- **`reset()` returns `(obs, info)`** — the old single-return `obs = env.reset()` pattern no longer works.
- **`render_mode` is now required at construction time:** Pass `render_mode="human"` or `render_mode="rgb_array"` to `gymnasium.make()`. The old `env.render(mode=...)` per-call API is removed.
- **Namespace changes:** All Atari environments use the `ALE/` prefix: `ALE/Pong-v5`, `ALE/Breakout-v5`, etc. The legacy `Pong-v4` IDs are removed.
- **Wrapper API overhaul:** Wrappers now subclass `gymnasium.Wrapper` and must implement the new v1 protocol. Many wrappers were renamed or reorganized under `gymnasium.wrappers`.
- **Autoreset behavior:** `gymnasium.make()` wraps envs in `AutoResetWrapper` by default in vector environments.

### Migration notes

- Replace `import gym` with `import gymnasium as gym` (already widely adopted).
- Update all `step()` call sites to unpack 5 values.
- Pin `gymnasium>=1.0.0,<2.0.0` for stability.

**Links:**
- Changelog: https://github.com/Farama-Foundation/Gymnasium/releases
- Migration guide: https://gymnasium.farama.org/introduction/migration_guide/

---

## 2. stable-retro

**Latest version (as of early 2025):** `0.9.2`

### Status

- `stable-retro` is the **community-maintained fork** of OpenAI's `gym-retro` (which was abandoned around 2020). Maintained primarily by Farama Foundation contributors.
- Actively maintained but with **infrequent releases** — development is volunteer-driven.
- Supports Gymnasium's new v1 API via `retro.make()` which returns a Gymnasium-compatible env when `gymnasium` is installed.
- Supports SNES, NES, Genesis/Mega Drive, Game Boy, and other console ROMs via libretro cores.

### Known issues

- **ROM integration is manual:** You must supply your own ROMs and import them with `python -m retro.import /path/to/roms/`.
- **Observation space can be large** (raw pixel frames) — always wrap with frame-stacking and resizing wrappers.
- **Action space:** Uses `MultiBinary` by default (one bit per button). Use `retro.Actions.DISCRETE` or `retro.Actions.MULTI_DISCRETE` for compatibility with most SB3 algorithms.
- **Python 3.12 support:** Was added in 0.9.2 but may still have edge cases with some libretro cores. Test thoroughly.
- **Gymnasium compatibility:** Works with Gymnasium but does **not** register environments in `gymnasium.envs.registry` — you must use `retro.make()` directly.

**Links:**
- Repository: https://github.com/Farama-Foundation/stable-retro
- PyPI: https://pypi.org/project/stable-retro/

---

## 3. ale-py (Arcade Learning Environment)

**Latest version (as of early 2025):** `0.10.x` (ALE v5 series)

### Key details

- `ale-py` is the Python interface to the Arcade Learning Environment, used by `gymnasium[atari]`.
- **ALE v5 env ID format:** All environments use the `ALE/GameName-v5` format (e.g., `ALE/Pong-v5`). The `-v5` suffix indicates the ALE v5 protocol with `terminated`/`truncated` semantics.
- **`gymnasium[atari]` installs `ale-py` and `shimmy[atari]` automatically** as of Gymnasium 1.0.
- The old `-ram-v4` and `-v4` IDs are **deprecated and removed** in Gymnasium 1.0+.
- `ale-py>=0.9.0` is required for Gymnasium 1.0 compatibility. Earlier versions only support the old `gym` API.
- **ROM auto-download:** `ale-py` ships with Atari ROMs (via `AutoROM` acceptance or bundled ROMs since 0.9), so manual ROM installation is no longer needed for most Atari games.

**Links:**
- Repository: https://github.com/Farama-Foundation/Arcade-Learning-Environment
- PyPI: https://pypi.org/project/ale-py/

---

## 4. Stable Baselines3 (SB3)

**Latest version (as of early 2025):** `2.4.x`

### Key updates since 2.3.0

- **Full Gymnasium 1.0 support** — the v1 API (`terminated`/`truncated`) is fully supported. SB3 2.x dropped support for the legacy `gym` package entirely.
- **CrossQ:** Added as a new algorithm in SB3 core — a sample-efficient off-policy method that avoids target networks.
- **Improved type hints** and better mypy compatibility.
- **Dictionary observation support** improved across all algorithms.
- **Python 3.12 support** confirmed.

### RecurrentPPO status

RecurrentPPO (LSTM-based PPO) is **not in SB3 core** — it lives in `sb3-contrib` (see below). There were discussions about merging it into mainline SB3, but as of 2025 it remains in contrib.

**Links:**
- Repository: https://github.com/DLR-RM/stable-baselines3
- Docs: https://stable-baselines3.readthedocs.io/

---

## 5. sb3-contrib

**Latest version (as of early 2025):** `2.4.x` (tracks SB3 versioning)

### Available algorithms

| Algorithm | Description | Observation |
|-----------|-------------|-------------|
| **RecurrentPPO** | LSTM-based PPO for partial observability | Most popular contrib algo |
| **TQC** | Truncated Quantile Critics (distributional off-policy) | Good for continuous control |
| **TRPO** | Trust Region Policy Optimization | Classic policy gradient |
| **ARS** | Augmented Random Search | Evolution-based, simple |
| **QR-DQN** | Quantile Regression DQN | Distributional RL |
| **CrossQ** | Batch normalization-based off-policy | Moved to SB3 core in 2.4+ |

### Notes on RecurrentPPO

- Requires `sb3-contrib` — install with `pip install sb3-contrib`.
- Uses sequence-based training; set `n_steps` carefully (longer rollouts help LSTM learning).
- **Does not support `Dict` observation spaces** cleanly — flatten or use a custom feature extractor.
- Compatible with Gymnasium 1.0 as of sb3-contrib 2.3+.

**Links:**
- Repository: https://github.com/Farama-Foundation/stable-baselines3-contrib (originally DLR-RM, transferred to Farama)
- Docs: https://sb3-contrib.readthedocs.io/

---

## 6. Shimmy (Compatibility Layer)

**Latest version (as of early 2025):** `2.0.x`

### Purpose

Shimmy provides **compatibility wrappers** to convert environments from other APIs into Gymnasium-compatible environments. Key adapters:

- **`shimmy[atari]`** — wraps ALE environments for Gymnasium (included automatically with `gymnasium[atari]`).
- **`shimmy[dm-control]`** — wraps DeepMind Control Suite environments.
- **`shimmy[openspiel]`** — wraps OpenSpiel game environments.
- **`shimmy[gym-v21]` / `shimmy[gym-v26]`** — wraps old `gym` 0.21 or 0.26 environments to work with Gymnasium 1.0.

### Key notes

- If you have legacy `gym`-based environments, Shimmy's `GymV26CompatibilityV0` wrapper is the official migration path.
- For Atari, Shimmy is installed transitively — you typically don't need to install it manually.

**Links:**
- Repository: https://github.com/Farama-Foundation/Shimmy
- Docs: https://shimmy.farama.org/

---

## 7. Version Compatibility Matrix

### Recommended pinning (tested compatible set)

```toml
# pyproject.toml / requirements.txt
gymnasium = ">=1.0.0,<2.0.0"
ale-py = ">=0.9.0,<0.11.0"
stable-baselines3 = ">=2.3.0,<3.0.0"
sb3-contrib = ">=2.3.0,<3.0.0"      # if using RecurrentPPO
stable-retro = ">=0.9.2"
shimmy = ">=2.0.0,<3.0.0"           # usually transitive
torch = ">=2.0.0,<2.6.0"
```

### Compatibility summary

| Library | Version | Gymnasium 1.0 | Python 3.12 | Notes |
|---------|---------|---------------|-------------|-------|
| **gymnasium** | 1.0.x | -- | Yes | v1 API is default |
| **ale-py** | 0.9.x-0.10.x | Yes | Yes | ALE/Game-v5 IDs |
| **stable-baselines3** | 2.3.x-2.4.x | Yes | Yes | Dropped legacy gym |
| **sb3-contrib** | 2.3.x-2.4.x | Yes | Yes | Must match SB3 version |
| **stable-retro** | 0.9.2 | Yes | Partial | Use retro.make() |
| **shimmy** | 2.0.x | Yes | Yes | Compatibility bridge |
| **torch** | 2.1-2.5 | N/A | Yes (2.2+) | CUDA 12.x recommended |

### Version conflict warnings

- **SB3 and sb3-contrib versions must match** (e.g., both 2.4.0). Mismatched versions cause import errors.
- **Gymnasium <1.0 + SB3 >=2.3** works but triggers deprecation warnings. Pin Gymnasium >=1.0 for clean operation.
- **ale-py <0.9 is incompatible** with Gymnasium 1.0 (missing v5 environment registrations).

---

## 8. Known Issues & Gotchas

### Environment creation

- **Atari env IDs changed:** Use `ALE/Pong-v5`, not `PongNoFrameskip-v4`. The old IDs will raise `NameNotFound`.
- **stable-retro envs are not in the Gymnasium registry:** You cannot use `gymnasium.make("RetroGame-v0")`. Always use `retro.make(game="GameName", ...)`.
- **Action space mismatch:** stable-retro defaults to `MultiBinary` actions. SB3's PPO/DQN expect `Discrete`. Convert with `retro.Actions.DISCRETE` in `retro.make()` or wrap with `gymnasium.wrappers.TransformAction`.

### Training

- **Frame stacking order:** Gymnasium 1.0 changed some wrapper defaults. Explicitly set `gymnasium.wrappers.FrameStackObservation(env, stack_size=4)` rather than relying on defaults.
- **RecurrentPPO memory usage:** LSTM hidden states are stored per environment step. With large `n_steps` and many parallel envs, VRAM usage can spike. Start with `n_envs=4, n_steps=128`.
- **AutoReset in VecEnv:** SB3's `DummyVecEnv`/`SubprocVecEnv` handle auto-resetting internally. Do **not** additionally wrap with Gymnasium's `AutoResetWrapper` or you will get double resets.
- **Observation dtype:** ALE returns `uint8` observations by default. SB3 handles normalization internally via `VecTransposeImage`, but custom wrappers should preserve `uint8` to avoid 4x memory overhead from `float32` conversion.

### Installation

- **`stable-retro` build from source** may be needed on some platforms (especially Apple Silicon). The PyPI wheel coverage has improved but is not universal.
- **ROM legality:** stable-retro requires you to legally own ROMs. Use `python -m retro.import` to register them.
- **Conflicting `gym` and `gymnasium` installations:** Having both `gym` and `gymnasium` installed can cause subtle import conflicts. Prefer a clean venv with only `gymnasium`.

---

## Recommendations for This Project (GOLDS)

Based on the current `pyproject.toml` pins (`gymnasium>=0.29.0`, `ale-py>=0.9.0`, `stable-baselines3[extra]>=2.3.0`, `stable-retro>=0.9.2`):

1. **Tighten the gymnasium pin** to `>=1.0.0,<2.0.0` — this ensures you are on the stable v1 API and avoids accidentally installing 0.29.x which has a different default API.
2. **Add `sb3-contrib>=2.3.0`** if RecurrentPPO is needed for partially observable games (NES games with hidden state).
3. **Pin torch upper bound** to `<2.6.0` to avoid untested breaking changes.
4. **Ensure SB3 and sb3-contrib versions match** — consider pinning both to the same minor version.
5. **Test Python 3.12 compatibility** end-to-end, especially stable-retro's libretro cores.

---

*Last updated: 2025. Verify all versions against PyPI before deploying.*
