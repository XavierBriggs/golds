# GOLDS - Multi-Environment RL Training System

Train reinforcement learning agents on Atari and NES games using Gymnasium, Stable-Retro, and Stable-Baselines3 PPO.

## Features

- **Multi-Platform Support**: Train on Atari 2600 games (via ALE) and NES/SNES games (via stable-retro)
- **PPO Training**: State-of-the-art Proximal Policy Optimization with DeepMind hyperparameters
- **GPU Acceleration**: CUDA support for fast CNN-based policy training
- **Easy CLI**: Simple commands for training, evaluation, and ROM management
- **Modular Design**: Factory pattern for easy extension to new games/platforms

## Quick Start

### Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
cd golds
uv sync

# Verify installation
uv run golds --help
```

### Train Space Invaders (No ROM Setup Needed)

```bash
# Quick training run
uv run golds train game space_invaders --timesteps 1000000 --envs 8

# Or with full config
uv run golds train run --config configs/games/space_invaders.yaml
```

### Train Super Mario Bros (Requires ROM Import)

```bash
# 1. Place your ROM file in the roms/ directory
# 2. Import ROMs
uv run golds rom import ./roms

# 3. Verify import
uv run golds rom verify SuperMarioBros-Nes

# 4. Train
uv run golds train game super_mario_bros --timesteps 10000000
```

## Commands

### Training

```bash
# Train a specific game with defaults
golds train game space_invaders

# Train with custom config
golds train run --config configs/games/space_invaders.yaml

# Resume training from checkpoint
golds train run --config config.yaml --resume outputs/models/checkpoints/model.zip
```

### Evaluation

```bash
# Evaluate trained model
golds eval model outputs/models/best/best_model.zip --game space_invaders --episodes 20

# Compare multiple models
golds eval compare model1.zip model2.zip --game space_invaders
```

### ROM Management

```bash
# Import ROMs from directory
golds rom import ./roms

# List available games
golds rom list

# Verify specific game
golds rom verify SuperMarioBros-Nes

# Show setup instructions
golds rom info
```

### Utilities

```bash
# List registered games
golds list-games

# Show system info
golds info

# Launch TensorBoard
golds tensorboard
```

## Configuration

Configuration files use YAML format. See `configs/` for examples.

```yaml
# configs/games/space_invaders.yaml
name: space_invaders
environment:
  platform: atari
  game_id: space_invaders
  n_envs: 8

ppo:
  learning_rate: 2.5e-4
  n_steps: 128

training:
  total_timesteps: 10000000
  device: auto
```

## Project Structure

```
golds/
├── src/golds/           # Main package
│   ├── cli/             # CLI commands
│   ├── config/          # Configuration system
│   ├── environments/    # Environment factory
│   ├── training/        # Training pipeline
│   ├── evaluation/      # Model evaluation
│   └── roms/            # ROM management
├── configs/             # Configuration files
├── roms/                # ROM directory (gitignored)
├── outputs/             # Training outputs (gitignored)
└── scripts/             # Setup scripts
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended for training)
- WSL2 on Windows (for CUDA support)

## ROM Setup

### Atari Games
Atari ROMs are included automatically. No setup needed.

### NES/SNES Games
1. Obtain ROMs legally (dump from cartridges you own)
2. Place in `roms/` directory
3. Run `golds rom import ./roms`

See `roms/README.md` for detailed instructions.

## License

MIT License
