"""Tests for the git provenance capture helper (R12)."""

from __future__ import annotations

import subprocess

from golds.utils.git_info import get_git_provenance


def test_get_git_provenance_matches_real_repo_head():
    """Against this real repo, git_sha matches `git rev-parse HEAD` exactly."""
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    git_sha, _ = get_git_provenance()

    assert git_sha == expected
    assert len(git_sha) == 40
    assert all(c in "0123456789abcdef" for c in git_sha)


def test_get_git_provenance_not_a_git_repo(tmp_path):
    """A directory with no .git ancestor falls back to unknown/clean."""
    git_sha, git_dirty = get_git_provenance(cwd=tmp_path)

    assert git_sha == "unknown"
    assert git_dirty is False


def test_get_git_provenance_clean_then_dirty(tmp_path):
    """A freshly committed repo is clean; editing a tracked file marks it dirty."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True
    )

    tracked = tmp_path / "file.txt"
    tracked.write_text("hello\n")
    subprocess.run(["git", "add", "file.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True
    )

    git_sha_clean, git_dirty_clean = get_git_provenance(cwd=tmp_path)
    assert len(git_sha_clean) == 40
    assert git_dirty_clean is False

    tracked.write_text("modified\n")
    git_sha_dirty, git_dirty_dirty = get_git_provenance(cwd=tmp_path)

    assert git_sha_dirty == git_sha_clean  # HEAD unchanged, only working tree
    assert git_dirty_dirty is True
