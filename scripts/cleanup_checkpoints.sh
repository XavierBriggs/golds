#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'EOF'
Cleanup old Stable-Baselines3 checkpoint zips under outputs/*/models/checkpoints.

Dry-run by default (prints what would be deleted). Use --apply to actually delete.

Usage:
  scripts/cleanup_checkpoints.sh [--outputs DIR] [--game GAME_ID] [--keep-last N] [--keep-every STEPS] [--prune-corrupt] [--apply]

Options:
  --outputs DIR        Output root (default: outputs)
  --game GAME_ID       Only clean outputs/<GAME_ID>
  --keep-last N        Keep the most recent N checkpoints by step number (default: 10)
  --keep-every STEPS   Also keep checkpoints where steps % STEPS == 0 (default: 0 = disabled)
  --prune-corrupt      Also delete corrupted checkpoint zips (BadZipFile, missing 'data', etc.)
  --apply              Actually delete files (otherwise dry-run)

Notes:
  - Checkpoint naming convention assumed: <exp_name>_<steps>_steps.zip
  - Keeps are computed per experiment directory.
EOF
}

outputs_dir="outputs"
game_id=""
keep_last="10"
keep_every="0"
apply="0"
prune_corrupt="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outputs)
      outputs_dir="${2:-}"; shift 2 ;;
    --game)
      game_id="${2:-}"; shift 2 ;;
    --keep-last)
      keep_last="${2:-}"; shift 2 ;;
    --keep-every)
      keep_every="${2:-}"; shift 2 ;;
    --prune-corrupt)
      prune_corrupt="1"; shift ;;
    --apply)
      apply="1"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "${keep_last}" =~ ^[0-9]+$ ]]; then
  echo "--keep-last must be an integer" >&2
  exit 2
fi
if ! [[ "${keep_every}" =~ ^[0-9]+$ ]]; then
  echo "--keep-every must be an integer" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
outputs_dir="$(cd "${repo_root}" && printf "%s/%s" "${repo_root}" "${outputs_dir}")"

if [[ ! -d "${outputs_dir}" ]]; then
  echo "outputs dir not found: ${outputs_dir}" >&2
  exit 1
fi

targets=()
if [[ -n "${game_id}" ]]; then
  targets+=("${outputs_dir}/${game_id}")
else
  while IFS= read -r -d '' d; do
    targets+=("$d")
  done < <(find "${outputs_dir}" -mindepth 1 -maxdepth 1 -type d -print0)
fi

python_bin=""
if command -v python >/dev/null 2>&1; then
  python_bin="python"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="python3"
elif [[ -x "${repo_root}/.venv/bin/python" ]]; then
  python_bin="${repo_root}/.venv/bin/python"
fi

total_delete=0
total_keep=0
total_corrupt=0

for exp_dir in "${targets[@]}"; do
  ckpt_dir="${exp_dir}/models/checkpoints"
  [[ -d "${ckpt_dir}" ]] || continue

  mapfile -t files < <(ls -1 "${ckpt_dir}"/*.zip 2>/dev/null || true)
  [[ ${#files[@]} -gt 0 ]] || continue

  declare -A by_step=()
  steps_list=()
  other_files=()

  for f in "${files[@]}"; do
    base="$(basename "${f}")"
    if [[ "${base}" =~ _([0-9]+)_steps\.zip$ ]]; then
      step="${BASH_REMATCH[1]}"
      by_step["${step}"]="${f}"
      steps_list+=("${step}")
    else
      other_files+=("${f}")
    fi
  done

  IFS=$'\n' steps_sorted=($(printf "%s\n" "${steps_list[@]}" | sort -n))
  unset IFS

  keep_set=()
  if (( keep_last > 0 )); then
    # Keep last N by step number
    start=$(( ${#steps_sorted[@]} - keep_last ))
    if (( start < 0 )); then start=0; fi
    for ((i=start; i<${#steps_sorted[@]}; i++)); do
      keep_set+=("${steps_sorted[i]}")
    done
  fi

  if (( keep_every > 0 )); then
    for step in "${steps_sorted[@]}"; do
      if (( step % keep_every == 0 )); then
        keep_set+=("${step}")
      fi
    done
  fi

  # Unique keep_set
  declare -A keep_steps=()
  for s in "${keep_set[@]}"; do keep_steps["$s"]=1; done

  # Optional corrupt detection
  declare -A corrupt=()
  if (( prune_corrupt == 1 )) && [[ -n "${python_bin}" ]]; then
    "${python_bin}" - <<'PY' "${files[@]}" 2>/dev/null || true
import sys
from pathlib import Path
from zipfile import ZipFile, BadZipFile

def is_ok(p: Path) -> bool:
    try:
        with ZipFile(p, "r") as zf:
            zf.read("data")
        return True
    except (BadZipFile, KeyError, OSError):
        return False

bad = [str(Path(x)) for x in sys.argv[1:] if not is_ok(Path(x))]
for b in bad:
    print(b)
PY
    while IFS= read -r bad; do
      [[ -n "${bad}" ]] || continue
      corrupt["${bad}"]=1
    done < <("${python_bin}" - <<'PY' "${files[@]}" 2>/dev/null || true
import sys
from pathlib import Path
from zipfile import ZipFile, BadZipFile

def is_ok(p: Path) -> bool:
    try:
        with ZipFile(p, "r") as zf:
            zf.read("data")
        return True
    except (BadZipFile, KeyError, OSError):
        return False

for x in sys.argv[1:]:
    p = Path(x)
    if not is_ok(p):
        print(str(p))
PY
)
  fi

  echo "== $(basename "${exp_dir}") checkpoints =="
  echo "dir: ${ckpt_dir}"
  echo "total: ${#files[@]} (named: ${#steps_sorted[@]}, other: ${#other_files[@]})"

  # Keep/delete decisions
  for step in "${steps_sorted[@]}"; do
    f="${by_step[${step}]}"
    if [[ -n "${corrupt[${f}]:-}" ]]; then
      echo "DELETE (corrupt): ${f}"
      ((total_corrupt+=1))
      ((total_delete+=1))
      if (( apply == 1 )); then rm -f -- "${f}"; fi
      continue
    fi
    if [[ -n "${keep_steps[${step}]:-}" ]]; then
      ((total_keep+=1))
      continue
    fi
    echo "DELETE: ${f}"
    ((total_delete+=1))
    if (( apply == 1 )); then rm -f -- "${f}"; fi
  done

  # Non-conforming names: only delete if corrupt pruning is enabled and they are corrupt.
  if (( prune_corrupt == 1 )); then
    for f in "${other_files[@]}"; do
      if [[ -n "${corrupt[${f}]:-}" ]]; then
        echo "DELETE (corrupt): ${f}"
        ((total_corrupt+=1))
        ((total_delete+=1))
        if (( apply == 1 )); then rm -f -- "${f}"; fi
      else
        ((total_keep+=1))
      fi
    done
  else
    total_keep=$((total_keep + ${#other_files[@]}))
  fi

  echo
done

if (( apply == 1 )); then
  echo "Applied cleanup."
else
  echo "Dry-run only (no files deleted). Use --apply to delete."
fi
echo "Kept: ${total_keep}  Deleted: ${total_delete}  (corrupt deleted: ${total_corrupt})"

