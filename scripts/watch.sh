#!/usr/bin/env bash
# Record the best model playing and open the video.
# Usage: bash scripts/watch.sh <game_id>
# Example: bash scripts/watch.sh pong

set -euo pipefail

GAME="${1:?Usage: bash scripts/watch.sh <game_id>}"

# Find the latest output directory for this game
OUTPUT_DIR=$(ls -dt outputs/${GAME}_* outputs/${GAME} 2>/dev/null | head -1)

if [[ -z "${OUTPUT_DIR}" ]]; then
    echo "No output directory found for ${GAME}. Train first with: golds go ${GAME}"
    exit 1
fi

# The trainer nests under the game name
EXP_DIR="${OUTPUT_DIR}/${GAME}"
if [[ ! -d "${EXP_DIR}" ]]; then
    EXP_DIR="${OUTPUT_DIR}"
fi

echo "Recording ${GAME} from ${EXP_DIR}..."
bash scripts/record_best_mp4.sh "${EXP_DIR}" --steps 5000

# Find the generated MP4
MP4=$(ls -t "${EXP_DIR}/videos/"*.mp4 2>/dev/null | head -1)

if [[ -z "${MP4}" ]]; then
    echo "No MP4 generated. Check if a model exists in ${EXP_DIR}"
    exit 1
fi

echo "Video saved: ${MP4}"

# Open the video (works on macOS, Linux with xdg-open, WSL2 with explorer.exe)
if command -v open >/dev/null 2>&1; then
    open "${MP4}"
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${MP4}"
elif command -v explorer.exe >/dev/null 2>&1; then
    # WSL2: convert Linux path to Windows path
    WIN_PATH=$(wslpath -w "${MP4}")
    explorer.exe "${WIN_PATH}"
else
    echo "Open manually: ${MP4}"
fi
