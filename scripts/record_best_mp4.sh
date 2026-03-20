#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

exp_dir="outputs/space_invaders"
out_dir=""
steps="10000"
seed="0"
game_id=""

usage() {
  cat <<'EOF'
Record an MP4 of the current best model playing.

Usage:
  scripts/record_best_mp4.sh [exp_dir] [--out DIR] [--steps N] [--seed N]
  scripts/record_best_mp4.sh [game_id] [--out DIR] [--steps N] [--seed N]
  scripts/record_best_mp4.sh --game GAME_ID [--exp EXP_DIR] [--out DIR] [--steps N] [--seed N]

Defaults:
  exp_dir: outputs/<game_id> (or outputs/space_invaders)
  --out:   <exp_dir>/videos
  --steps: 10000
  --seed:  0

Model selection order:
  <exp_dir>/best/best_model.zip
  <exp_dir>/models/best_training/best_training_model.zip
  <exp_dir>/models/final_model(.zip)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" != "" && "${1:-}" != --* ]]; then
  # Backwards compatible positional argument:
  # - if it's an existing directory => exp_dir
  # - otherwise => game_id (mapped to outputs/<game_id>)
  if [[ -d "$1" ]]; then
    exp_dir="$1"
  else
    game_id="$1"
    exp_dir="outputs/$game_id"
  fi
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --game)
      game_id="${2:-}"; shift 2 ;;
    --exp|--exp-dir)
      exp_dir="${2:-}"; shift 2 ;;
    --out)
      out_dir="${2:-}"; shift 2 ;;
    --steps)
      steps="${2:-}"; shift 2 ;;
    --seed)
      seed="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "${game_id}" && ( "${exp_dir}" == "outputs/space_invaders" || -z "${exp_dir}" ) ]]; then
  exp_dir="outputs/$game_id"
fi

if [[ -z "${out_dir}" ]]; then
  out_dir="${exp_dir}/videos"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found in PATH. Install uv or run the Python snippet directly." >&2
  exit 1
fi

# Avoid uv defaulting to a non-writable cache location (common in sandboxed/CI environments).
: "${UV_CACHE_DIR:=.uv-cache}"

UV_CACHE_DIR="$UV_CACHE_DIR" uv run python - "$exp_dir" "$out_dir" "$steps" "$seed" "${game_id}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from golds.environments.factory import EnvironmentFactory

exp_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
steps = int(sys.argv[3])
seed = int(sys.argv[4])
game_id_cli = sys.argv[5] if len(sys.argv) > 5 else ""

if not exp_dir.exists():
    raise SystemExit(f"Experiment directory not found: {exp_dir}")

out_dir.mkdir(parents=True, exist_ok=True)

config_path = exp_dir / "config.json"
if config_path.exists():
    cfg = json.loads(config_path.read_text())
    game_id = cfg.get("environment", {}).get("game_id", "space_invaders")
    frame_stack = int(cfg.get("environment", {}).get("frame_stack", 4))
else:
    game_id = game_id_cli or "space_invaders"
    frame_stack = 4

candidates = [
    exp_dir / "best" / "best_model.zip",
    exp_dir / "models" / "best_training" / "best_training_model.zip",
    exp_dir / "models" / "final_model.zip",
    exp_dir / "models" / "final_model",
]
model_path = next((p for p in candidates if p.exists()), None)
if model_path is None:
    raise SystemExit(
        "No model found. Looked for:\\n  " + "\\n  ".join(str(p) for p in candidates)
    )

env: VecEnv = EnvironmentFactory.create_eval_env(game_id=game_id, frame_stack=frame_stack, seed=seed)
name_prefix = model_path.stem

def try_wrap_vec_video_recorder(env: VecEnv) -> VecEnv | None:
    try:
        import moviepy  # noqa: F401
    except Exception:
        return None
    from stable_baselines3.common.vec_env import VecVideoRecorder

    return VecVideoRecorder(
        env,
        video_folder=str(out_dir),
        record_video_trigger=lambda step: step == 0,
        video_length=steps,
        name_prefix=name_prefix,
    )

env_with_recorder = try_wrap_vec_video_recorder(env)

model = PPO.load(model_path, env=env_with_recorder or env)

if env_with_recorder is not None:
    obs = env_with_recorder.reset()
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env_with_recorder.step(action)
        if dones[0]:
            obs = env_with_recorder.reset()
    env_with_recorder.close()
else:
    import cv2

    mp4_path = out_dir / f"{name_prefix}.mp4"

    obs = env.reset()
    frames_written = 0
    writer = None
    try:
        for _ in range(steps):
            images = env.get_images()
            if not images:
                raise RuntimeError(
                    "Environment did not return frames. "
                    "Ensure envs are created with `render_mode='rgb_array'`."
                )
            frame_rgb = images[0]
            if frame_rgb is None:
                raise RuntimeError("Received empty frame from env.get_images().")
            frame_rgb = np.asarray(frame_rgb)

            if writer is None:
                height, width = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(mp4_path), fourcc, 30.0, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(
                        f"Failed to open video writer for {mp4_path}. "
                        "Check that your OpenCV build supports MP4 encoding."
                    )

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            frames_written += 1

            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env.step(action)
            if dones[0]:
                obs = env.reset()
    finally:
        env.close()
        if writer is not None:
            writer.release()

    if frames_written == 0:
        raise RuntimeError("No frames written; video was not created.")

print(f"Model: {model_path}")
print(f"Saved MP4(s) to: {out_dir}")
PY
