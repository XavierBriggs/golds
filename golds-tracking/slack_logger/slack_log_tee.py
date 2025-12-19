import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

# Add parent directory (golds-tracking/) to path so we can import slack_logger
# This works whether run as a script or as a module
_this_file = Path(__file__).resolve()
_this_dir = _this_file.parent
tracking_root = _this_dir.parent  # golds-tracking/
if str(tracking_root) not in sys.path:
    sys.path.insert(0, str(tracking_root))

from slack_logger.slack_notify import post_message_with_backoff


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def is_checkpoint_message(line: str) -> bool:
    """Check if a line contains checkpoint-related information."""
    line_lower = line.lower()
    checkpoint_keywords = [
        "saving new best model",
        "saving best model",
        "checkpoint",
        "saved model",
        "best_training",
        "checkpoints/",
        "final model saved",
    ]
    return any(keyword in line_lower for keyword in checkpoint_keywords)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_name")
    parser.add_argument("log_path")
    args = parser.parse_args()

    job_name: str = args.job_name
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    interval_s = float(os.environ.get("SLACK_NOTIFY_INTERVAL_SECONDS", "20"))
    max_lines = int(os.environ.get("SLACK_NOTIFY_MAX_LINES", "25"))
    max_chars = int(os.environ.get("SLACK_NOTIFY_MAX_CHARS", "3500"))

    buffer: deque[str] = deque()
    last_send = time.monotonic()
    last_progress_send = time.monotonic()
    # Keep recent context lines for checkpoint messages
    recent_context: deque[str] = deque(maxlen=5)

    def flush(force: bool = False) -> None:
        nonlocal last_send
        if not buffer:
            return
        elapsed = time.monotonic() - last_send
        if not force and elapsed < interval_s:
            return

        lines = list(buffer)[-max_lines:]
        buffer.clear()

        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[-max_chars:]

        payload = f"{job_name}\n{now_iso()}\n```\n{text}\n```"
        post_message_with_backoff(payload, username="golds")

        last_send = time.monotonic()

    def send_checkpoint_notification(checkpoint_line: str) -> None:
        """Immediately send checkpoint notification to Slack."""
        # Include recent context lines for better understanding
        # Note: recent_context already includes checkpoint_line, so we use it directly
        context_lines = list(recent_context)
        context_text = "\n".join(context_lines) if context_lines else checkpoint_line

        payload = f"✅ CHECKPOINT: {job_name}\n{now_iso()}\n```\n{context_text}\n```"
        post_message_with_backoff(payload, username="golds")

    with log_path.open("a", encoding="utf-8", errors="replace") as f:
        start_msg = f"{job_name} started {now_iso()}"
        post_message_with_backoff(start_msg, username="golds")

        for raw in sys.stdin:
            line = raw.rstrip("\n")
            f.write(raw)
            f.flush()

            sys.stdout.write(raw)
            sys.stdout.flush()

            # Add to recent context for checkpoint notifications
            recent_context.append(line)

            # Check if this is a checkpoint message
            if is_checkpoint_message(line):
                # Immediately send checkpoint notification
                send_checkpoint_notification(line)
                # Also add to buffer for regular logging
                buffer.append(line)
            else:
                buffer.append(line)

            if time.monotonic() - last_progress_send > 5.0:
                flush(force=False)
                last_progress_send = time.monotonic()

        flush(force=True)
        done_msg = f"{job_name} finished {now_iso()}"
        post_message_with_backoff(done_msg, username="golds")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
