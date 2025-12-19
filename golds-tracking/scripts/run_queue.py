import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

# Add parent directory to path so we can import slack_logger
_script_dir = Path(__file__).resolve().parent
tracking_root = _script_dir.parent  # golds-tracking/
golds_root = tracking_root.parent   # main repo root (pyproject.toml, configs/games, etc.)
if str(tracking_root) not in sys.path:
    sys.path.insert(0, str(tracking_root))

try:
    import yaml
except Exception as e:
    print("Missing dependency: pyyaml. Install it and try again.")
    raise

from slack_logger.slack_notify import post_message_with_backoff


@dataclass
class Job:
    name: str
    config: str
    retries: int = 0
    resume_latest: bool = False


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _resolve_path(candidate: str | Path, *, base_dirs: list[Path]) -> Path:
    p = Path(candidate)
    if p.is_absolute():
        return p
    for base in base_dirs:
        resolved = base / p
        if resolved.exists():
            return resolved
    return p


def _safe_load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _experiment_name_from_config(config_path: Path) -> str | None:
    cfg = _safe_load_yaml(config_path)
    name = cfg.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _checkpoint_steps(path: Path) -> int | None:
    # Conventional pattern: <exp_name>_<steps>_steps.zip
    name = path.name
    if not name.endswith("_steps.zip"):
        return None
    try:
        return int(name.rsplit("_", 2)[-2])
    except Exception:
        return None


def _is_valid_sb3_zip(path: Path) -> bool:
    try:
        with ZipFile(path, "r") as zf:
            _ = zf.read("data")
        return True
    except (BadZipFile, KeyError, OSError):
        return False


def _find_latest_checkpoint(exp_dir: Path) -> Path | None:
    ckpt_dir = exp_dir / "models" / "checkpoints"
    if not (ckpt_dir.exists() and ckpt_dir.is_dir()):
        return None

    candidates: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob("*.zip"):
        steps = _checkpoint_steps(p)
        if steps is None:
            continue
        candidates.append((steps, p))

    # Prefer the newest checkpoint by step count.
    candidates.sort(key=lambda t: t[0], reverse=True)

    # Fall back to older ones if the newest is corrupted/truncated.
    for _, p in candidates:
        if _is_valid_sb3_zip(p):
            return p
    return None


def load_jobs(queue_path: Path) -> tuple[list[Job], dict[str, Any]]:
    data = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
    defaults = data.get("default", {}) if isinstance(data, dict) else {}
    jobs_raw = data.get("jobs", []) if isinstance(data, dict) else []
    force_resume = os.environ.get("GOLDS_QUEUE_RESUME_LATEST", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    jobs: list[Job] = []
    for item in jobs_raw:
        name = str(item["name"])
        config = str(item["config"])
        retries = int(item.get("retries", defaults.get("retries", 0)))
        resume_latest = bool(
            item.get("resume_latest", defaults.get("resume_latest", False))
        )
        if force_resume:
            resume_latest = True
        jobs.append(
            Job(name=name, config=config, retries=retries, resume_latest=resume_latest)
        )
    return jobs, data


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_job(job: Job, log_dir: Path, meta_dir: Path) -> int:
    ensure_dir(log_dir)
    ensure_dir(meta_dir)

    log_path = log_dir / f"{job.name}.log"
    meta_path = meta_dir / f"{job.name}.json"

    config_path = _resolve_path(
        job.config,
        base_dirs=[
            Path.cwd(),
            tracking_root,
            golds_root,
        ],
    )

    cmd = [
        "uv",
        "run",
        "golds",
        "train",
        "run",
        "--config",
        str(config_path),
    ]

    output_root_env = os.environ.get("GOLDS_TRAIN_OUTPUT_DIR", "").strip()
    if output_root_env:
        output_root = Path(output_root_env)
        if not output_root.is_absolute():
            output_root = (golds_root / output_root).resolve()
        cmd += ["--output", str(output_root)]
    else:
        # Default to main repo outputs/, regardless of where the queue runner is executed from.
        output_root = golds_root / "outputs"
        cmd += ["--output", str(output_root)]

    proc_env = os.environ.copy()
    # Avoid uv defaulting to a non-writable cache location (common in sandboxed/CI environments).
    proc_env.setdefault("UV_CACHE_DIR", str(golds_root / ".uv-cache"))

    exp_name = _experiment_name_from_config(config_path) if config_path.exists() else None
    exp_dir = (output_root / exp_name) if exp_name else None
    resume_path: Path | None = None
    if job.resume_latest and exp_dir is not None:
        resume_path = _find_latest_checkpoint(exp_dir)
        if resume_path is not None:
            cmd += ["--resume", str(resume_path)]

    meta: dict[str, Any] = {
        "job_name": job.name,
        "config": job.config,
        "config_resolved": str(config_path),
        "resume_latest": job.resume_latest,
        "resume_resolved": str(resume_path) if resume_path is not None else None,
        "start_time": now_iso(),
        "command": cmd,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if not config_path.exists():
        msg = (
            f"queue failed {job.name}: config not found '{job.config}' "
            f"(resolved to '{config_path}') {now_iso()}"
        )
        post_message_with_backoff(msg, username="golds")
        meta["end_time"] = now_iso()
        meta["exit_code_train"] = 2
        meta["exit_code_tee"] = None
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return 2

    start_msg = f"queue start {job.name} {now_iso()}"
    post_message_with_backoff(start_msg, username="golds")

    tee_cmd = [
        sys.executable,
        "-m",
        "slack_logger.slack_log_tee",
        job.name,
        str(log_path),
    ]

    proc_train = subprocess.Popen(
        cmd,
        cwd=str(golds_root),
        env=proc_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc_train.stdout is not None
    proc_tee = subprocess.Popen(
        tee_cmd,
        cwd=str(tracking_root),
        stdin=proc_train.stdout,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
    )

    code_train = proc_train.wait()
    code_tee = proc_tee.wait()

    meta["end_time"] = now_iso()
    meta["exit_code_train"] = code_train
    meta["exit_code_tee"] = code_tee
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if code_train == 0:
        post_message_with_backoff(f"queue done {job.name} {now_iso()}", username="golds")
    else:
        post_message_with_backoff(f"queue failed {job.name} code {code_train} {now_iso()}", username="golds")

    return code_train


def main() -> int:
    queue_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs") / "queue.yaml"
    queue_path = _resolve_path(queue_arg, base_dirs=[Path.cwd(), tracking_root])
    jobs, _ = load_jobs(queue_path)

    log_dir_raw = Path(os.environ.get("LOG_DIR", "logs"))
    meta_dir_raw = Path(os.environ.get("RUN_META_DIR", "runs_meta"))
    log_dir = log_dir_raw if log_dir_raw.is_absolute() else (tracking_root / log_dir_raw)
    meta_dir = meta_dir_raw if meta_dir_raw.is_absolute() else (tracking_root / meta_dir_raw)

    overall_ok = True
    for job in jobs:
        attempts = 0
        while True:
            attempts += 1
            code = run_job(job, log_dir=log_dir, meta_dir=meta_dir)
            if code == 0:
                break
            if attempts > job.retries + 1:
                overall_ok = False
                break
            post_message_with_backoff(f"retry {job.name} attempt {attempts} {now_iso()}", username="golds")
            time.sleep(5)

        if not overall_ok:
            break

    if overall_ok:
        post_message_with_backoff(f"queue all done {now_iso()}", username="golds")
        return 0
    post_message_with_backoff(f"queue stopped early {now_iso()}", username="golds")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
