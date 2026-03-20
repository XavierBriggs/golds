# GOLDS Codebase Audit

**Date:** 2026-03-19
**Scope:** Full source tree at `/Users/xavierbriggs/development/golds`
**Commit:** `2a4dfac` (main)

---

## 1. Module Dependency Graph

### Core Package (`src/golds/`)

```
golds/__init__.py
  (no internal imports)

golds/__main__.py
  -> golds.cli.main

golds/cli/__init__.py
  (no internal imports)

golds/cli/main.py
  -> golds.__init__          (__version__)
  -> golds.cli.train         (train_app)
  -> golds.cli.evaluate      (eval_app)
  -> golds.cli.roms          (rom_app)
  -> golds.environments.registry  (lazy, inside list_games())
  -> golds.utils.device           (lazy, inside info())

golds/cli/train.py
  -> golds.config.loader        (lazy, inside train_run/train_game)
  -> golds.training.trainer     (lazy, inside train_run/train_game)
  -> golds.environments.factory (lazy, inside train_preflight)
  -> golds.environments.registry(lazy, inside train_game)

golds/cli/evaluate.py
  -> golds.evaluation.evaluator (lazy, inside eval_model/eval_compare)

golds/cli/roms.py
  -> golds.roms.manager         (lazy, inside rom_import/rom_verify)

golds/config/__init__.py
  -> golds.config.schema    (ExperimentConfig, PPOConfig, EnvironmentConfig, TrainingConfig)
  -> golds.config.loader    (ConfigLoader)

golds/config/loader.py
  -> golds.config.schema    (ExperimentConfig)

golds/config/schema.py
  (no internal imports -- leaf node)

golds/environments/__init__.py
  -> golds.environments.factory  (EnvironmentFactory)
  -> golds.environments.registry (GameRegistry, GameRegistration)

golds/environments/factory.py
  -> golds.environments.registry        (GameRegistry)
  -> golds.environments.atari.maker     (lazy, via _lazy_register_platforms)
  -> golds.environments.retro.maker     (lazy, via _lazy_register_platforms)
  -> golds.environments.retro.self_play (lazy, inside create())

golds/environments/registry.py
  (no internal imports -- leaf node; registers all games at module level)

golds/environments/atari/__init__.py
  -> golds.environments.atari.maker (AtariEnvironmentMaker)

golds/environments/atari/env_id.py
  (no internal imports -- leaf node)

golds/environments/atari/maker.py
  -> golds.environments.atari.env_id (resolve_atari_env_id)

golds/environments/common/__init__.py
  (empty -- no imports, no code)

golds/environments/retro/__init__.py
  -> golds.environments.retro.maker (RetroEnvironmentMaker)

golds/environments/retro/maker.py
  (no internal imports -- leaf node)

golds/environments/retro/self_play.py
  (no internal imports -- leaf node)

golds/evaluation/__init__.py
  -> golds.evaluation.evaluator (Evaluator)

golds/evaluation/evaluator.py
  -> golds.environments.factory (EnvironmentFactory)

golds/training/__init__.py
  -> golds.training.trainer (Trainer)

golds/training/callbacks.py
  (no internal imports -- leaf node)

golds/training/trainer.py
  -> golds.config.schema         (ExperimentConfig)
  -> golds.environments.factory  (EnvironmentFactory)
  -> golds.training.callbacks    (ProgressCallback, SaveOnBestTrainingRewardCallback,
                                   SelfPlaySnapshotCallback, SafeCheckpointCallback,
                                   create_eval_callback)
  -> golds.utils.device          (get_device)

golds/roms/__init__.py
  -> golds.roms.manager (ROMManager)

golds/roms/manager.py
  (no internal imports -- leaf node)

golds/utils/__init__.py
  -> golds.utils.device (get_device)

golds/utils/device.py
  (no internal imports -- leaf node)
```

### External Dependencies (Third-Party)

| Module | Used By |
|--------|---------|
| `typer` | cli/main.py, cli/train.py, cli/evaluate.py, cli/roms.py |
| `rich` | cli/main.py, cli/evaluate.py, cli/roms.py, cli/train.py, evaluation/evaluator.py, training/trainer.py |
| `pydantic` | config/schema.py |
| `yaml` (PyYAML) | config/loader.py |
| `stable_baselines3` | environments/factory.py, environments/atari/maker.py, environments/retro/maker.py, environments/retro/self_play.py, training/callbacks.py, training/trainer.py, evaluation/evaluator.py |
| `gymnasium` | environments/atari/maker.py, environments/retro/maker.py, environments/retro/self_play.py |
| `torch` | utils/device.py |
| `numpy` | environments/retro/maker.py, environments/retro/self_play.py, training/callbacks.py, evaluation/evaluator.py, cli/train.py |
| `cv2` (OpenCV) | environments/retro/maker.py |
| `retro` (stable-retro) | environments/retro/maker.py (optional import) |
| `ale_py` | environments/atari/maker.py (optional import) |

### Tracking Sub-project (`golds-tracking/`)

```
golds-tracking/scripts/run_queue.py
  -> slack_logger.slack_notify  (post_message_with_backoff)
  -> yaml (PyYAML)

golds-tracking/scripts/slack_test.py
  -> slack_logger.slack_notify  (post_message_with_backoff)

golds-tracking/scripts/gpu_count.py
  -> torch

golds-tracking/slack_logger/__init__.py
  (empty)

golds-tracking/slack_logger/slack_log_tee.py
  -> slack_logger.slack_notify  (post_message_with_backoff)

golds-tracking/slack_logger/slack_notify.py
  (no internal imports -- leaf node; uses only stdlib)

scripts/run_queue.py
  (thin wrapper that delegates to golds-tracking/scripts/run_queue.py via runpy)
```

---

## 2. Data Flow Diagram

```
                          YAML Config Files
                                |
                   +------------+------------+
                   |                         |
           configs/defaults.yaml    configs/games/<game>.yaml
                   |                         |
                   v                         v
            +------+------+          +-------+-------+
            | DEFAULT_CONFIG|         | user YAML dict |
            | (Python dict) |         |                |
            +------+-------+         +-------+--------+
                   |                          |
                   +--- deep_merge() ---------+
                   |
                   v
          +--------+---------+
          | merged Python dict|
          +--------+---------+
                   |
                   v
          +--------+---------+
          | ExperimentConfig  |   <-- Pydantic validation + defaults
          | .environment      |       (PPOConfig, EnvironmentConfig,
          | .ppo              |        TrainingConfig)
          | .training         |
          +--------+---------+
                   |
          +--------+---------+
          |                  |
          v                  v
  .to_ppo_kwargs()     EnvironmentFactory.create()
          |                  |
          v                  v
  +-------+--------+   +----+----+
  | PPO(**kwargs)   |   | VecEnv  |
  | policy=CnnPolicy|   | (train) |
  | tensorboard_log |   +---------+
  | device=...      |       |
  +-------+--------+       |
          |                 |
          +----+------------+
               |
               v
        model.learn(
          total_timesteps=...,
          callback=CallbackList[
            VerboseEvalCallback,    <-- eval_env (single env, DummyVecEnv)
            SafeCheckpointCallback,
            SaveOnBestTrainingRewardCallback,
            ProgressCallback,
            SelfPlaySnapshotCallback (conditional)
          ],
          tb_log_name=config.name,
        )
               |
               v
        +------+------+
        |   Outputs    |
        +--------------+
        | outputs/<name>/
        |   config.json
        |   models/
        |     final_model.zip
        |     best_training/best_training_model.zip
        |     checkpoints/<name>_<steps>_steps.zip
        |   best/best_model.zip    (from EvalCallback)
        |   eval/evaluations.npz   (from EvalCallback)
        |   logs/<name>_0/         (TensorBoard)
        |   videos/                (empty -- not implemented)
        |   self_play/opponents/   (if self_play mode)
        +--------------+
```

---

## 3. File Inventory

### Core Package (`src/golds/`)

| File | Lines | Responsibility |
|------|------:|----------------|
| `__init__.py` | 7 | Package root; exports `__version__` |
| `__main__.py` | 6 | Module entry point (`python -m golds`) |
| `cli/__init__.py` | 1 | CLI sub-package marker |
| `cli/main.py` | 135 | Root Typer app; `list-games`, `info`, `tensorboard`, `version` commands |
| `cli/train.py` | 253 | `train run`, `train preflight`, `train game`, `train list-configs` commands |
| `cli/evaluate.py` | 108 | `eval model`, `eval compare` commands |
| `cli/roms.py` | 177 | `rom import`, `rom list`, `rom verify`, `rom info` commands |
| `config/__init__.py` | 12 | Re-exports schema classes + ConfigLoader |
| `config/loader.py` | 168 | YAML loading, deep_merge, DEFAULT_CONFIG dict, ConfigLoader class |
| `config/schema.py` | 139 | Pydantic models: PPOConfig, EnvironmentConfig, TrainingConfig, ExperimentConfig |
| `environments/__init__.py` | 6 | Re-exports EnvironmentFactory, GameRegistry |
| `environments/factory.py` | 177 | Unified EnvironmentFactory; lazy platform registration; VecFrameStack/VecTransposeImage |
| `environments/registry.py` | 195 | GameRegistry class + all 12 game registrations (7 Atari, 5 Retro) |
| `environments/atari/__init__.py` | 5 | Re-exports AtariEnvironmentMaker |
| `environments/atari/env_id.py` | 42 | Atari env-id normalization (legacy v4 <-> ALE v5) |
| `environments/atari/maker.py` | 227 | `make_atari_vec_env()` + dead `AtariEnvironmentMaker` class |
| `environments/common/__init__.py` | 1 | Empty (docstring only) |
| `environments/retro/__init__.py` | 4 | Re-exports RetroEnvironmentMaker |
| `environments/retro/maker.py` | 344 | RetroPreprocessing, FrameSkip wrappers; `make_retro_vec_env()` + dead `RetroEnvironmentMaker` class |
| `environments/retro/self_play.py` | 270 | OpponentSpec, TwoPlayerOpponentWrapper, VecTwoPlayerOpponentWrapper |
| `evaluation/__init__.py` | 5 | Re-exports Evaluator |
| `evaluation/evaluator.py` | 170 | Evaluator class + `quick_evaluate()` convenience function |
| `training/__init__.py` | 5 | Re-exports Trainer |
| `training/callbacks.py` | 363 | SaveOnBestTrainingRewardCallback, TensorBoardVideoCallback (stub), ProgressCallback, VerboseEvalCallback, SafeCheckpointCallback, SelfPlaySnapshotCallback, create_eval_callback() |
| `training/trainer.py` | 382 | Trainer class: env creation, model creation, callback wiring, disk space check, training loop |
| `roms/__init__.py` | 5 | Re-exports ROMManager |
| `roms/manager.py` | 252 | ROMManager: scan, import, verify, list ROMs; SHA1 hashing |
| `utils/__init__.py` | 5 | Re-exports get_device |
| `utils/device.py` | 38 | `get_device()` and `get_device_info()` |

### Tests (`tests/`)

| File | Lines | Responsibility |
|------|------:|----------------|
| `__init__.py` | 1 | Package marker |
| `conftest.py` | 41 | Pytest fixtures: project_root, configs_dir, sample_config |
| `test_atari_env_id.py` | 16 | 3 tests for env-id resolution |
| `test_config.py` | 116 | Tests for PPOConfig, EnvironmentConfig, ExperimentConfig, deep_merge, ConfigLoader |

### Tracking Sub-project (`golds-tracking/`)

| File | Lines | Responsibility |
|------|------:|----------------|
| `scripts/run_queue.py` | 467 | Job queue runner: YAML queue parsing, preflight, resume strategy, Slack notifications, watchdog |
| `scripts/gpu_count.py` | 26 | Print CUDA GPU count and names |
| `scripts/slack_test.py` | 42 | Send test Slack notification |
| `slack_logger/__init__.py` | 0 | Empty package marker |
| `slack_logger/slack_log_tee.py` | 171 | Tee stdin to log file + periodic Slack notifications with checkpoint detection |
| `slack_logger/slack_notify.py` | 115 | Slack webhook posting with exponential backoff |

### Top-level Script

| File | Lines | Responsibility |
|------|------:|----------------|
| `scripts/run_queue.py` | 25 | Thin wrapper that delegates to `golds-tracking/scripts/run_queue.py` via `runpy` |

**Total:** ~3,999 lines of Python across 37 files (excluding blank lines in empty `__init__.py` files).

---

## 4. Dead Code Inventory

### 4.1 `src/golds/environments/common/__init__.py` -- Empty Module

The file contains only a docstring (`"""Common environment utilities and wrappers."""`). No classes, functions, or imports. The `common/` sub-package is never imported by any other module. It appears to be a placeholder for future shared wrappers that were never implemented.

### 4.2 `TensorBoardVideoCallback` in `training/callbacks.py` (lines 79-109)

This callback class is a stub. Its `_on_step` method prints a message when triggered but performs no actual video recording. The comment explicitly says `"This is a placeholder for the recording logic"`. It is never instantiated anywhere in the codebase -- the Trainer class does not use it, and no CLI command references it.

### 4.3 `AtariEnvironmentMaker` class in `environments/atari/maker.py` (lines 185-227)

This class wraps the free function `make_atari_vec_env()` with a class-based API and maintains its own `GAMES` dictionary (duplicating what `GameRegistry` provides). It is:
- Never instantiated by `EnvironmentFactory`
- Never imported by the training pipeline
- Only re-exported by `environments/atari/__init__.py`, but nothing consumes that export

The class also has 3 extra games (`enduro`, `beam_rider`, `freeway`) not present in `GameRegistry`, making the two registries inconsistent.

### 4.4 `RetroEnvironmentMaker` class in `environments/retro/maker.py` (lines 284-344)

Same pattern as `AtariEnvironmentMaker`. This class wraps `make_retro_vec_env()` with a class-based API and maintains its own `GAMES` dictionary (only 2 games vs. 5 in `GameRegistry`). It is:
- Never instantiated by `EnvironmentFactory`
- Only re-exported by `environments/retro/__init__.py`, but nothing consumes that export
- Contains `list_available_games()` and `is_game_available()` static methods that duplicate `ROMManager.verify_game_available()` and `ROMManager.list_available_games()`

### 4.5 `TwoPlayerOpponentWrapper` (gym-level) in `environments/retro/self_play.py` (lines 66-164)

The `TwoPlayerOpponentWrapper` is a `gym.Wrapper` for a single (non-vectorized) environment. The codebase exclusively uses the `VecTwoPlayerOpponentWrapper` (applied at the VecEnv level in `factory.py`). The gym-level wrapper is never instantiated. It appears to be an earlier implementation that was superseded by the VecEnv variant but never removed.

### 4.6 `quick_evaluate()` function in `evaluation/evaluator.py` (lines 149-169)

A convenience function that creates an `Evaluator`, runs evaluation, and prints results. It is never called by any CLI command or other module. The CLI commands directly instantiate `Evaluator`.

### 4.7 `videos/` directory creation in `training/trainer.py`

The Trainer creates `self.video_dir = self.output_dir / "videos"` and ensures it exists, but no code ever writes to it (since `TensorBoardVideoCallback` is a stub and is never used).

### 4.8 `ROMManager.get_instructions()` and `ROMManager.ensure_directory()` in `roms/manager.py`

`get_instructions()` returns a multi-line string but is never called. The CLI `rom info` command hardcodes its own instructions inline. `ensure_directory()` is similarly never called externally.

### 4.9 Duplicate `_parse_dotenv()` implementation

`_parse_dotenv()` is implemented identically in both `golds-tracking/scripts/run_queue.py` (lines 40-59) and `golds-tracking/slack_logger/slack_notify.py` (lines 17-37).

---

## 5. Known Issues

### HIGH Severity

| # | Issue | Details |
|---|-------|---------|
| H1 | **Space Invaders Round 2 crashed** | `04_space_invaders_ppo.json`: `exit_code_train=1`, ran only ~51 seconds (00:46:07 to 00:46:58). Resumed from checkpoint at 5,400,000 steps toward a 30M target. No diagnostic info recorded in the metadata. Likely a crash during resume or environment creation. |
| H2 | **Sonic Round 2 incomplete** | `01_sonic_the_hedgehog_ppo.json`: has `start_time` (2025-12-19T06:31:51) but no `end_time`, `exit_code_train`, or `exit_code_tee`. The process either was killed externally, is still running, or the metadata was never updated. Resumed from checkpoint at 1,499,976 steps toward a 40M target. |
| H3 | **No structured results storage** | Training runs produce TensorBoard logs and scattered model files, but there is no structured results database, CSV, or JSON summary of final metrics per run. Comparing runs requires manual TensorBoard inspection or ad-hoc evaluation. |
| H4 | **Queue system does not track outcomes** | The queue runner records `exit_code_train` in JSON metadata but does not record evaluation metrics, final reward, or training progress. A run that exits 0 after reaching `total_timesteps` by resuming (i.e., `remaining <= 0`, skipping training) is indistinguishable from a run that completed full training. |

### MEDIUM Severity

| # | Issue | Details |
|---|-------|---------|
| M1 | **`total_timesteps <= 0` early-exit saves no model** | In `trainer.py` lines 341-344, when resuming and the target is already reached, the code prints a warning and skips the training loop. However, `model.save(final_model_path)` still executes afterward, so a model IS saved. The real issue is that no evaluation happens in this path, so the "final model" quality is unknown. |
| M2 | **Eval defaults to 10 episodes (too few)** | Both `eval_episodes` in `TrainingConfig` and the CLI `--episodes` flag default to 10. For high-variance games (Space Invaders, platformers), 10 episodes produces noisy estimates. Round 2 configs reduce this further to 3 for retro games (to avoid stalls), compounding the problem. |
| M3 | **No multi-seed evaluation** | All configs use `seed: 42`. There is no mechanism to run the same config with multiple seeds and aggregate results. Single-seed results are not reproducibility-robust. |
| M4 | **Reward regimes not labeled** | Atari configs use `clip_reward: true` (rewards in {-1, 0, 1}) while retro configs inherit the default (`clip_reward: true`). Evaluation results mix clipped and unclipped reward scales with no labeling. The evaluator does not distinguish between clipped training rewards and true game scores. |
| M5 | **TensorBoardVideoCallback is a stub** | The callback exists but does nothing. The `videos/` output directory is created but always empty. Users expecting video recording get nothing. |
| M6 | **`environments/common/` is empty** | A sub-package exists with only a docstring. It adds unnecessary directory structure and may confuse contributors expecting shared utilities there. |
| M7 | **DRY violation: DEFAULT_CONFIG duplicates defaults.yaml** | `config/loader.py` contains a `DEFAULT_CONFIG` Python dict (lines 13-46) that is a near-exact copy of `configs/defaults.yaml`. When `defaults.yaml` is found on disk, `deep_merge(DEFAULT_CONFIG, loaded)` is used, meaning the Python dict acts as a fallback. If someone updates one but not the other, silent divergence occurs. |

### LOW Severity

| # | Issue | Details |
|---|-------|---------|
| L1 | **AtariEnvironmentMaker.GAMES out of sync with GameRegistry** | The dead class lists 10 Atari games; GameRegistry lists 7. Three games (`enduro`, `beam_rider`, `freeway`) exist only in the dead code. |
| L2 | **RetroEnvironmentMaker.GAMES out of sync with GameRegistry** | The dead class lists 2 retro games; GameRegistry lists 5. |
| L3 | **Duplicate `_parse_dotenv()` in tracking scripts** | Identical ~20-line function in `run_queue.py` and `slack_notify.py`. |
| L4 | **Duplicate `_is_valid_sb3_zip()` in tracking scripts and self_play.py** | Same validation logic implemented in `golds-tracking/scripts/run_queue.py` and `src/golds/environments/retro/self_play.py`. |
| L5 | **`resume_strategy` field not in runs_meta for Sonic** | The `01_sonic_the_hedgehog_ppo.json` metadata has `"resume_strategy": null` but the queue.yaml does not specify one. This is correct behavior (defaults to `latest_checkpoint` via `resume_latest: true`) but may confuse readers of the metadata. |
| L6 | **Test coverage is minimal** | Only 2 test files with ~174 lines total. No tests for Trainer, Evaluator, callbacks, environment factory, ROM manager, or the queue system. |
| L7 | **`multiprocessing.set_start_method("spawn", force=True)` called in factory.py** | On non-Linux platforms, this is called every time `create()` runs (guarded by a try/except). Using `force=True` can affect other multiprocessing users in the same process. |

---

## 6. Config System Analysis

### Loading Pipeline

```
User runs:  uv run golds train run --config configs/games/tetris.yaml

1. ConfigLoader.__init__(config_dir=Path("configs"))
   -> self._defaults = self._load_defaults()
      -> Checks for configs/defaults.yaml
      -> If exists: deep_merge(DEFAULT_CONFIG, yaml.safe_load("defaults.yaml"))
      -> If not: DEFAULT_CONFIG.copy()

2. ConfigLoader.load(Path("configs/games/tetris.yaml"))
   -> user_config = yaml.safe_load(tetris.yaml)
   -> merged = deep_merge(self._defaults, user_config)
   -> ExperimentConfig(**merged)  # Pydantic validation

3. CLI overrides (--seed, --device) applied directly:
   -> exp_config.training.seed = seed
   -> exp_config.training.device = device

4. Trainer.__init__(config, output_dir)
   -> self._save_config()  # Dumps config.model_dump() as JSON

5. Trainer._create_model(train_env)
   -> ppo_kwargs = self.config.to_ppo_kwargs()
   -> PPO(policy="CnnPolicy", env=train_env, **ppo_kwargs, ...)
```

### Merge Precedence (highest wins)

1. CLI flags (`--seed`, `--device`)
2. Game-specific YAML (`configs/games/<game>.yaml`)
3. `configs/defaults.yaml` (on-disk defaults)
4. `DEFAULT_CONFIG` dict in `loader.py` (hardcoded fallback)
5. Pydantic field defaults in `schema.py`

### The DRY Violation in Detail

**`config/loader.py` lines 13-46** define `DEFAULT_CONFIG`:

```python
DEFAULT_CONFIG: dict[str, Any] = {
    "ppo": {
        "learning_rate": 2.5e-4,
        "n_steps": 128,
        "batch_size": 256,
        ...
    },
    "environment": {
        "n_envs": 8,
        "frame_stack": 4,
        ...
    },
    "training": {
        "total_timesteps": 10_000_000,
        "eval_freq": 50_000,
        ...
    },
}
```

**`configs/defaults.yaml`** contains the same values in YAML format. The Python dict exists as a fallback when `defaults.yaml` is not on disk (e.g., running from a different working directory). When both exist, the YAML is merged ON TOP of the Python dict, so the YAML wins for any key present in both.

**Risk:** If a developer changes a default in `defaults.yaml` but not in `DEFAULT_CONFIG` (or vice versa), the effective default depends on whether the YAML file is reachable at runtime. This is especially subtle because `ConfigLoader` uses a relative path (`Path("configs")`) that depends on the current working directory.

Additionally, `schema.py` Pydantic models define their own field defaults (e.g., `learning_rate: float = Field(default=2.5e-4)`), creating a *third* source of truth. The Pydantic defaults only apply when keys are absent from the merged dict, but they create a maintenance burden when changing defaults.

### Fields Present in Schema but Absent from DEFAULT_CONFIG / defaults.yaml

| Field | Defined In | Missing From |
|-------|-----------|-------------|
| `environment.state` | schema.py | DEFAULT_CONFIG, defaults.yaml |
| `environment.players` | schema.py | DEFAULT_CONFIG, defaults.yaml |
| `environment.opponent` | schema.py | DEFAULT_CONFIG, defaults.yaml |
| `environment.opponent_model_path` | schema.py | DEFAULT_CONFIG, defaults.yaml |
| `training.eval_deterministic` | schema.py | DEFAULT_CONFIG, defaults.yaml |
| `training.self_play_snapshot_freq` | schema.py | DEFAULT_CONFIG, defaults.yaml |
| `training.self_play_max_snapshots` | schema.py | DEFAULT_CONFIG, defaults.yaml |

These fields rely solely on Pydantic defaults and are not overridable through `defaults.yaml`.

---

## 7. Complexity Hotspots

### By Line Count (top 5 source files)

| Rank | File | Lines | Notes |
|------|------|------:|-------|
| 1 | `golds-tracking/scripts/run_queue.py` | 467 | Highest in the repo. Mixes job parsing, resume strategy, preflight, subprocess management, watchdog, and Slack notifications in a single file. |
| 2 | `training/trainer.py` | 382 | Moderately complex. Clear separation of concerns within the class. Disk-space check and WSL error handling add bulk. |
| 3 | `training/callbacks.py` | 363 | 6 callback classes + 1 factory function. `VerboseEvalCallback` has heartbeat/timing logic that adds complexity. |
| 4 | `environments/retro/maker.py` | 344 | Contains 2 wrapper classes (RetroPreprocessing, FrameSkip), 2 factory functions, and 1 dead compat class (RetroEnvironmentMaker). |
| 5 | `environments/retro/self_play.py` | 270 | 2 parallel implementations (gym.Wrapper and VecEnvWrapper) of the same opponent logic. One is dead code. |

### By Cyclomatic Complexity (qualitative)

1. **`golds-tracking/scripts/run_queue.py:run_job()`** -- Most complex single function. Handles config resolution, resume strategy with 6 branches, preflight subprocess, training subprocess with tee piping, watchdog loop with stall detection, and metadata updates. ~130 lines.

2. **`environments/factory.py:EnvironmentFactory.create()`** -- Moderately complex. Extracts kwargs from a catch-all `**kwargs`, delegates to platform-specific makers, applies 2 common wrappers, and conditionally applies self-play wrapper. The monkey-patching at module level (`_create_with_lazy_register`) adds indirection.

3. **`training/callbacks.py:VerboseEvalCallback._on_step()`** -- Wraps `super()._on_step()` in try/except, manages timing state, and delegates to `_log_success_callback()` which itself tracks episode completions.

4. **`environments/atari/maker.py:make_atari_vec_env()`** -- Triple-nested try/except for PermissionError fallback across different multiprocessing start methods.

### Structural Complexity

- **Lazy registration pattern in `factory.py`**: The module-level monkey-patching of `EnvironmentFactory.create` with `_create_with_lazy_register` is non-obvious. A reader must understand that `create()` is replaced at import time. This avoids circular imports but makes the call chain harder to trace.

- **`**kwargs` pass-through in `EnvironmentFactory.create()`**: Retro-specific parameters (`players`, `opponent_mode`, etc.) are extracted from `**kwargs` inside `create()` AND forwarded via `**kwargs` to the platform maker. This means the same parameters may be processed twice (once extracted, once passed through).

---

## Appendix: Run Metadata Summary

| Job | Config | Resume From | Start | End | Exit |
|-----|--------|-------------|-------|-----|------|
| 01_super_mario_bros_2_ppo | super_mario_bros_2_japan.yaml | checkpoint 7.8M steps | 2025-12-18T13:07 | 2025-12-18T19:05 | 0 |
| 02_mortal_kombat_ii_ppo | mortal_kombat_ii.yaml | checkpoint 1.3M steps | 2025-12-18T19:05 | 2025-12-18T22:13 | 0 |
| 03_tetris_ppo | tetris.yaml | (fresh) | 2025-12-18T22:13 | 2025-12-19T00:45 | 0 |
| 04_space_invaders_ppo | space_invaders.yaml | checkpoint 5.4M steps | 2025-12-19T00:46 | 2025-12-19T00:46 | **1** |
| 01_sonic_the_hedgehog_ppo | sonic_the_hedgehog.yaml | checkpoint 1.5M steps | 2025-12-19T06:31 | **missing** | **missing** |

All runs executed on a WSL2 environment at `/mnt/c/Users/ironb_68qna1s/Development/golds/`.
