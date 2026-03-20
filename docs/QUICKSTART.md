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

## Evaluate a Trained Model

After training completes, the output path is printed. Evaluate with:

    golds eval model outputs/pong_YYYYMMDD-HHMM/pong/best/best_model.zip --game pong --episodes 10

Or run a full benchmark (100 episodes x 3 seeds):

    golds eval benchmark outputs/pong_YYYYMMDD-HHMM/pong/best/best_model.zip --game pong

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

    bash scripts/record_best_mp4.sh pong outputs/pong_YYYYMMDD-HHMM/pong

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

## System Check

Run diagnostics anytime:

    golds doctor

## Common Issues

**"Model not found"** -- Check the output path. The trainer creates a
nested directory: `outputs/<run>/<game_name>/best/best_model.zip`

**Training is slow** -- Make sure your GPU is detected: `golds doctor`.
On Mac, MPS should be auto-detected. On Linux/WSL, CUDA should be available.

**"stable-retro not installed"** -- Retro games need an extra package:
`uv pip install stable-retro`. Then import ROMs with `golds rom import ./roms`.

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
