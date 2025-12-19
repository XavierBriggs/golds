#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add parent directory (golds-tracking/) to path so we can import slack_logger
_script_dir = Path(__file__).resolve().parent
tracking_root = _script_dir.parent  # golds-tracking/
if str(tracking_root) not in sys.path:
    sys.path.insert(0, str(tracking_root))

from slack_logger.slack_notify import post_message_with_backoff


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a test Slack notification.")
    parser.add_argument(
        "--text",
        default=f"golds slack test {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        help="Message text",
    )
    parser.add_argument("--username", default="golds", help="Slack username")
    args = parser.parse_args()

    ok = post_message_with_backoff(args.text, username=args.username)
    if ok:
        print("Slack: ok")
        return 0
    print(
        "Slack: failed (check `SLACK_WEBHOOK_URL` in your environment or `golds-tracking/.env`; "
        "set `SLACK_NOTIFY_DEBUG=1` for HTTP status output)."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

