#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Interactive MP4 recorder for any trained model in outputs/<game_id>.
#
# Supports:
# - best eval model:          outputs/<game>/best/best_model.zip
# - best training model:      outputs/<game>/models/best_training/best_training_model.zip
# - final model:              outputs/<game>/models/final_model(.zip)
# - checkpoints (conventional): outputs/<game>/models/checkpoints/<game>_<steps>_steps.zip

usage() {
  cat <<'EOF'
Record an MP4 of a selected model (interactive).

Usage:
  scripts/record_model_mp4.sh

Notes:
  - Default output folder: outputs/<game>/videos
  - Duration is in seconds; default FPS=30 so 60s => 1800 env steps recorded.
  - Uses `uv run` and sets `UV_CACHE_DIR=.uv-cache` by default.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found in PATH." >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
outputs_dir="${repo_root}/outputs"

if [[ ! -d "${outputs_dir}" ]]; then
  echo "No outputs directory found at: ${outputs_dir}" >&2
  exit 1
fi

echo "Select a game (from ${outputs_dir}):"
mapfile -t games < <(
  for d in "${outputs_dir}"/*; do
    [[ -d "${d}" ]] || continue
    base="$(basename "${d}")"
    # Filter out accidental top-level artifact folders.
    case "${base}" in
      logs|models|videos|checkpoints|tensorboard) continue ;;
    esac
    # Heuristic: experiment dirs usually have config.json or models/.
    if [[ -f "${d}/config.json" || -d "${d}/models" || -d "${d}/best" ]]; then
      echo "${base}"
    fi
  done | sort
)
if [[ ${#games[@]} -eq 0 ]]; then
  echo "No trained games found under ${outputs_dir}." >&2
  exit 1
fi

PS3="Game #: "
select game_id in "${games[@]}" "Custom..."; do
  if [[ "${REPLY}" =~ ^[0-9]+$ ]] && (( REPLY >= 1 && REPLY <= ${#games[@]} )); then
    break
  fi
  if [[ "${game_id}" == "Custom..." ]]; then
    read -r -p "Enter game id (directory under outputs/): " game_id
    break
  fi
  echo "Invalid selection." >&2
done

exp_dir="${outputs_dir}/${game_id}"
if [[ ! -d "${exp_dir}" ]]; then
  echo "Experiment directory not found: ${exp_dir}" >&2
  exit 1
fi

echo
echo "Select model type:"
PS3="Model #: "
select model_kind in "best_eval" "best_training" "final" "checkpoint" "custom_path"; do
  case "${model_kind}" in
    best_eval|best_training|final|checkpoint|custom_path) break ;;
    *) echo "Invalid selection." >&2 ;;
  esac
done

model_path=""
name_hint=""

case "${model_kind}" in
  best_eval)
    model_path="${exp_dir}/best/best_model.zip"
    name_hint="best_model"
    ;;
  best_training)
    model_path="${exp_dir}/models/best_training/best_training_model.zip"
    name_hint="best_training_model"
    ;;
  final)
    if [[ -f "${exp_dir}/models/final_model.zip" ]]; then
      model_path="${exp_dir}/models/final_model.zip"
    else
      model_path="${exp_dir}/models/final_model"
    fi
    name_hint="final_model"
    ;;
  checkpoint)
    ckpt_dir="${exp_dir}/models/checkpoints"
    if [[ ! -d "${ckpt_dir}" ]]; then
      echo "No checkpoint directory found: ${ckpt_dir}" >&2
      exit 1
    fi
    mapfile -t ckpts < <(ls -1 "${ckpt_dir}"/*.zip 2>/dev/null | sort -V || true)
    if [[ ${#ckpts[@]} -eq 0 ]]; then
      echo "No checkpoint .zip files found in: ${ckpt_dir}" >&2
      exit 1
    fi
    echo
    echo "Available checkpoints (newest last):"
    # Print a compact list of step numbers if they match the conventional pattern.
    for p in "${ckpts[@]}"; do
      b="$(basename "${p}")"
      if [[ "${b}" =~ _([0-9]+)_steps\.zip$ ]]; then
        echo "  ${BASH_REMATCH[1]}"
      else
        echo "  ${b}"
      fi
    done
    echo
    read -r -p "Enter checkpoint step number (e.g. 1000000), or paste a filename: " ckpt_sel
    if [[ "${ckpt_sel}" =~ ^[0-9]+$ ]]; then
      candidate="${ckpt_dir}/${game_id}_${ckpt_sel}_steps.zip"
      if [[ ! -f "${candidate}" ]]; then
        echo "Checkpoint not found: ${candidate}" >&2
        exit 1
      fi
      model_path="${candidate}"
      name_hint="ckpt_${ckpt_sel}"
    else
      # Allow selecting by filename or path
      if [[ -f "${ckpt_sel}" ]]; then
        model_path="${ckpt_sel}"
      elif [[ -f "${ckpt_dir}/${ckpt_sel}" ]]; then
        model_path="${ckpt_dir}/${ckpt_sel}"
      else
        echo "Checkpoint selection not found: ${ckpt_sel}" >&2
        exit 1
      fi
      name_hint="checkpoint"
    fi
    ;;
  custom_path)
    read -r -p "Enter full path to a SB3 .zip model: " model_path
    name_hint="custom"
    ;;
esac

if [[ -z "${model_path}" || ! -e "${model_path}" ]]; then
  echo "Model not found: ${model_path}" >&2
  exit 1
fi

echo
read -r -p "Video length seconds [60]: " seconds
seconds="${seconds:-60}"
if ! [[ "${seconds}" =~ ^[0-9]+$ ]]; then
  echo "Seconds must be an integer." >&2
  exit 1
fi

read -r -p "FPS [30]: " fps
fps="${fps:-30}"
if ! [[ "${fps}" =~ ^[0-9]+$ ]]; then
  echo "FPS must be an integer." >&2
  exit 1
fi

read -r -p "Seed [0]: " seed
seed="${seed:-0}"
if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
  echo "Seed must be an integer." >&2
  exit 1
fi

read -r -p "Output directory [${exp_dir}/videos]: " out_dir
out_dir="${out_dir:-${exp_dir}/videos}"

steps=$(( seconds * fps ))

echo
echo "Recording:"
echo "  game:  ${game_id}"
echo "  model: ${model_path}"
echo "  out:   ${out_dir}"
echo "  secs:  ${seconds}  fps: ${fps}  steps: ${steps}"
echo

: "${UV_CACHE_DIR:=${repo_root}/.uv-cache}"

UV_CACHE_DIR="${UV_CACHE_DIR}" uv run python - "${exp_dir}" "${out_dir}" "${model_path}" "${steps}" "${fps}" "${seed}" "${name_hint}" <<'PY'
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from golds.environments.factory import EnvironmentFactory

exp_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
model_path = Path(sys.argv[3])
steps = int(sys.argv[4])
fps = int(sys.argv[5])
seed = int(sys.argv[6])
name_hint = sys.argv[7] if len(sys.argv) > 7 else ""

if not exp_dir.exists():
    raise SystemExit(f"Experiment directory not found: {exp_dir}")
if not model_path.exists():
    raise SystemExit(f"Model not found: {model_path}")

out_dir.mkdir(parents=True, exist_ok=True)

config_path = exp_dir / "config.json"
game_id = None
frame_stack = 4
if config_path.exists():
    cfg = json.loads(config_path.read_text())
    game_id = cfg.get("environment", {}).get("game_id")
    frame_stack = int(cfg.get("environment", {}).get("frame_stack", 4))
if not game_id:
    raise SystemExit(f"Could not determine game_id (missing {config_path}).")

env: VecEnv = EnvironmentFactory.create_eval_env(game_id=game_id, frame_stack=frame_stack, seed=seed)

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
        name_prefix="tmp",  # will be replaced once we know which model loaded
    )

@dataclass(frozen=True)
class Candidate:
    label: str
    path: Path


def parse_checkpoint_steps(p: Path) -> int | None:
    name = p.name
    if not name.endswith("_steps.zip"):
        return None
    try:
        steps_part = name.rsplit("_", 2)[-2]
        return int(steps_part)
    except Exception:
        return None


def checkpoint_candidates(selected: Path) -> list[Candidate]:
    ckpt_dir = exp_dir / "models" / "checkpoints"
    if not (ckpt_dir.exists() and ckpt_dir.is_dir()):
        return [Candidate("selected", selected)]
    if selected.parent != ckpt_dir:
        return [Candidate("selected", selected)]

    selected_steps = parse_checkpoint_steps(selected)
    all_ckpts: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob("*.zip"):
        s = parse_checkpoint_steps(p)
        if s is None:
            continue
        all_ckpts.append((s, p))
    all_ckpts.sort(key=lambda t: t[0], reverse=True)

    if not all_ckpts:
        return [Candidate("selected", selected)]

    start_index = 0
    if selected_steps is not None:
        for i, (s, p) in enumerate(all_ckpts):
            if s == selected_steps and p == selected:
                start_index = i
                break

    # Try selected first, then older checkpoints.
    return [
        Candidate(f"checkpoint_{s}", p)
        for (s, p) in all_ckpts[start_index:]
    ]


def select_loadable_model(env: VecEnv, selected: Path) -> tuple[PPO, Path]:
    last_err: Exception | None = None
    for cand in checkpoint_candidates(selected):
        try:
            model = PPO.load(cand.path, env=env)
            return model, cand.path
        except Exception as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Could not load model: {selected}")


env_with_recorder = try_wrap_vec_video_recorder(env)
model, chosen_model_path = select_loadable_model(env_with_recorder or env, model_path)

prefix = chosen_model_path.stem
if name_hint and name_hint not in prefix:
    prefix = f"{prefix}_{name_hint}"

if env_with_recorder is not None:
    # Patch the recorder name_prefix now that we know the actual model loaded.
    # VecVideoRecorder only uses name_prefix when closing, so setting the attribute works.
    setattr(env_with_recorder, "name_prefix", prefix)

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

    mp4_path = out_dir / f"{prefix}.mp4"

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
                writer = cv2.VideoWriter(str(mp4_path), fourcc, float(fps), (width, height))
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

print(f"Game: {game_id}")
if chosen_model_path != model_path:
    print(f"Selected model failed to load: {model_path}")
print(f"Model: {chosen_model_path}")
print(f"Saved MP4(s) to: {out_dir}")
PY
