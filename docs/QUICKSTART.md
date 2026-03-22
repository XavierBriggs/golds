# GOLDS Quickstart Guide

## First Time Setup

Run the setup wizard to check your system:

    golds setup

This checks your GPU, dependencies, ROMs, and runs a pipeline test.

## Train Your First Model

The fastest way to start training:

    golds go pong

That's it. This will:
- Load the Pong config from `configs/games/pong.yaml`
- Create a timestamped output directory in `outputs/`
- Start training on the best available device (GPU if found)
- Save checkpoints, eval results, and the final model automatically

### Override defaults

    golds go pong --timesteps 1000000          # Shorter run
    golds go breakout --device cpu             # Force CPU
    golds go pong --resume                     # Resume previous run

## Action Sets

Retro games use reduced action sets for faster learning. Instead of ~126
raw button combinations, the agent sees only the meaningful ones:

| Action set | Actions | Games |
|------------|---------|-------|
| `platformer` | 9 (move, jump, run, duck) | Mario, Sonic, Mega Man |
| `fighter` | 12 (move, punch, kick, combos) | MK2, Street Fighter |
| `puzzle` | 5 (move, rotate) | Tetris |
| `full` | All filtered (~126) | Default / fallback |

Set in your game config under `environment:`:

```yaml
environment:
  action_set: platformer
```

## Evaluate a Trained Model

After training completes, the output path is printed. Evaluate with:

    golds eval model outputs/pong_YYYYMMDD-HHMM/best/best_model.zip --game pong --episodes 10

Or run a full benchmark (100 episodes x 3 seeds):

    golds eval benchmark outputs/pong_YYYYMMDD-HHMM/best/best_model.zip --game pong

## Train Multiple Games

Train all configured games in sequence:

    golds train-all

Or pick specific games:

    golds train-all --games pong,breakout,space_invaders

Games that already hit their timestep target are automatically skipped.
If training was interrupted, it auto-resumes from the latest checkpoint.

## Check Progress

See training status across all games:

    golds status

This shows which games are trained, progress percentage, best rewards,
human-normalized scores, and training time.

## View Results

    golds results show                    # All results
    golds results show --game pong        # Filter by game
    golds results leaderboard             # Cross-game ranking

## Watch Training Live

Open TensorBoard to see reward curves in real time:

    golds tensorboard --logdir outputs/

Then open http://localhost:6006 in your browser.

## Record Gameplay Video

    bash scripts/record_best_mp4.sh pong outputs/pong_YYYYMMDD-HHMM

## Available Games

See all registered games:

    golds list-games

### Atari (work out of the box)
Pong, Breakout, Space Invaders, Q*bert, Seaquest, Asteroids,
Ms. Pac-Man, Montezuma's Revenge, Enduro, Frostbite

### Retro (require ROM import)
Super Mario Bros, Super Mario Bros 2 Japan, Tetris,
Mortal Kombat II, Sonic the Hedgehog, Street Fighter II, Mega Man 2

To set up retro games:

    uv pip install stable-retro
    golds rom import ./roms
    golds rom list

## Advanced: Multi-Level Training

Train across multiple levels simultaneously for better generalization.
Each parallel environment randomly selects a level on reset:

```yaml
environment:
  levels:
    - GreenHillZone.Act1
    - GreenHillZone.Act2
    - MarbleZone.Act1
```

This prevents the agent from memorizing a single level layout.

## Advanced: Exploration Bonus (RND)

Enable Random Network Distillation for exploration in sparse-reward
or hard-exploration games. RND gives the agent a bonus for visiting
novel states:

```yaml
training:
  rnd_enabled: true
  rnd_reward_scale: 0.01
```

Useful for platformers where the agent gets stuck at obstacles.
Not needed for fighting games (health delta is already dense).

## System Check

Run diagnostics anytime:

    golds doctor

## Common Issues

**"Model not found"** -- Check the output path. Models are saved at:
`outputs/<game>_YYYYMMDD-HHMM/best/best_model.zip`

**Training is slow** -- Make sure your GPU is detected: `golds doctor`.
On Mac, MPS should be auto-detected. On Linux/WSL, CUDA should be available.

**"stable-retro not installed"** -- Retro games need an extra package:
`uv pip install stable-retro`. Then import ROMs with `golds rom import ./roms`.

**"Agent gets stuck"** -- Enable sticky actions (`sticky_action_prob: 0.25`)
and reduce the action space (`action_set: platformer`). Consider enabling
RND exploration (`rnd_enabled: true`) and extended reward shaping
(`death_penalty: -1.0`, `time_penalty: -0.001`).

## Config Reference

### Environment fields

| Field | Default | Description |
|-------|---------|-------------|
| `platform` | required | `atari` or `retro` |
| `game_id` | required | Game identifier |
| `n_envs` | 8 | Parallel environments |
| `frame_stack` | 4 | Frames to stack |
| `frame_skip` | 4 | Frames to skip |
| `clip_reward` | true | Clip rewards to {-1,0,+1} |
| `reward_regime` | clipped | `clipped`, `raw`, or `normalized` |
| `action_set` | full | `full`, `platformer`, `fighter`, `puzzle` |
| `sticky_action_prob` | 0.0 | Probability of repeating previous action |
| `levels` | [] | Level rotation list (empty = single level) |
| `x_pos_reward_scale` | 0.0 | X-position progress shaping scale |
| `death_penalty` | 0.0 | Penalty on termination (negative) |
| `collectible_reward_scale` | 0.0 | Bonus for rings/coins |
| `time_penalty` | 0.0 | Per-step penalty (negative) |
| `max_episode_steps` | 0 | Episode truncation limit |

### Training fields

| Field | Default | Description |
|-------|---------|-------------|
| `total_timesteps` | 10M | Training budget |
| `eval_freq` | 50K | Evaluation interval |
| `save_freq` | 100K | Checkpoint interval |
| `device` | auto | `auto`, `cuda`, `mps`, `cpu` |
| `rnd_enabled` | false | Enable RND exploration |
| `rnd_reward_scale` | 0.01 | RND intrinsic reward scale |

## Command Reference

| Command | Description |
|---------|-------------|
| `golds go <game>` | Train a game with one command |
| `golds train-all` | Train all configured games |
| `golds setup` | Check system and dependencies |
| `golds status` | Show training progress |
| `golds eval model <path> --game <game>` | Evaluate a model |
| `golds eval benchmark <path> --game <game>` | Full benchmark (multi-seed) |
| `golds results show` | Show training results |
| `golds results leaderboard` | Cross-game ranking |
| `golds list-games` | List available games |
| `golds doctor` | System diagnostics |
| `golds tensorboard` | Launch TensorBoard |
