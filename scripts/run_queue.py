#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "golds-tracking" / "scripts" / "run_queue.py"
    if not target.exists():
        print(f"Expected queue runner not found: {target}", file=sys.stderr)
        return 2

    os.chdir(str(target.parents[1]))
    sys.argv = [str(target), *sys.argv[1:]]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
