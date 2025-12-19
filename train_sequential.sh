#!/usr/bin/env bash

set -euo pipefail

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

run_job () {
    local name="$1"
    shift
    echo "==== $(date -Is) START $name ===="
    "$@" 2>&1 | python slack_logger/slack_log_tee.py "$name" "$LOG_DIR/$name.log"
    echo "==== $(date -Is) DONE $name ===="
    echo
}
run_job "01_super_mario_bros_2_ppo" uv run golds train run --config configs/games/super_mario_bros_2_japan.yaml
run_job "02_mortal_kombat_ii_ppo" uv run golds train run --config configs/games/mortal_kombat_ii.yaml
run_job "03_tetris_ppo" uv run golds train run --config configs/games/tetris.yaml
run_job "04_space_invaders_ppo"  uv run golds train run --config configs/games/space_invaders.yaml

echo "All jobs finished at $(date -Is)"