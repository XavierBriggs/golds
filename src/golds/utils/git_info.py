"""Git provenance capture for reproducibility tracking (R12)."""

from __future__ import annotations

import subprocess
from pathlib import Path

_UNKNOWN_SHA = "unknown"


def get_git_provenance(cwd: str | Path | None = None) -> tuple[str, bool]:
    """Capture the current git commit SHA and working-tree dirty state.

    Args:
        cwd: Directory to run git commands in. Defaults to the current
            working directory.

    Returns:
        Tuple of (git_sha, git_dirty). If ``cwd`` is not inside a git
        repository, git is unavailable, or a command fails/times out,
        returns ("unknown", False) rather than raising.
    """
    str_cwd = str(cwd) if cwd is not None else None

    try:
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str_cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return _UNKNOWN_SHA, False

    if sha_result.returncode != 0:
        return _UNKNOWN_SHA, False

    git_sha = sha_result.stdout.strip()
    if not git_sha:
        return _UNKNOWN_SHA, False

    try:
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str_cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return git_sha, False

    git_dirty = status_result.returncode == 0 and bool(status_result.stdout.strip())

    return git_sha, git_dirty
