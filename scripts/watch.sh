#!/usr/bin/env bash
# Record the best model playing and open the video.
# Usage: bash scripts/watch.sh <game_id>
# Example: bash scripts/watch.sh pong

set -euo pipefail

GAME="${1:?Usage: bash scripts/watch.sh <game_id>}"

# Find the best model across all output directories for this game.
# Supports unified (outputs/{game}_{ts}/best/) and legacy nested
# (outputs/{game}_{ts}/{game}/best/) directory layouts.
MODEL=$(find outputs -path "*${GAME}*/best/best_model.zip" -o \
                     -path "*${GAME}*/best_training/best_training_model.zip" -o \
                     -path "*${GAME}*/models/final_model.zip" 2>/dev/null | \
        head -1)

if [[ -z "${MODEL}" ]]; then
    echo "No trained model found for ${GAME}."
    echo "Train first with: golds go ${GAME}"
    exit 1
fi

# EXP_DIR is the parent that contains best/, models/, etc.
# Unified:  outputs/pong_20260321-0540/best/best_model.zip  -> outputs/pong_20260321-0540
# Legacy:   outputs/pong_20260321-0540/pong/best/best_model.zip -> outputs/pong_20260321-0540/pong
EXP_DIR=$(dirname "$(dirname "${MODEL}")")

echo "Game: ${GAME}"
echo "Model: ${MODEL}"
echo "Recording gameplay..."

bash scripts/record_best_mp4.sh "${EXP_DIR}" --steps 5000

# Find the generated MP4
MP4=$(ls -t "${EXP_DIR}/videos/"*.mp4 2>/dev/null | head -1)

if [[ -z "${MP4}" ]]; then
    echo "No MP4 generated. Check the output above for errors."
    exit 1
fi

echo "Video saved: ${MP4}"

# Open the video (macOS, Linux, WSL2)
if command -v open >/dev/null 2>&1; then
    open "${MP4}"
elif command -v explorer.exe >/dev/null 2>&1; then
    WIN_PATH=$(wslpath -w "${MP4}")
    explorer.exe "${WIN_PATH}"
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${MP4}"
else
    echo "Open manually: ${MP4}"
fi
