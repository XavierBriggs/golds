import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread

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
    """Return True for "milestone" lines worth notifying immediately."""
    line_lower = line.lower()
    checkpoint_keywords = (
        "saving new best model",
        "saving best model",
        "checkpoints/",
        "final model saved",
    )
    return any(keyword in line_lower for keyword in checkpoint_keywords)


def _slack_worker(
    q: Queue[str],
    stop: Event,
    *,
    attempts: int,
) -> None:
    while True:
        if stop.is_set() and q.empty():
            return
        try:
            msg = q.get(timeout=0.5)
        except Exception:
            continue
        try:
            post_message_with_backoff(msg, username="golds", attempts=attempts)
        finally:
            q.task_done()


def _enqueue(q: Queue[str], payload: str) -> None:
    try:
        q.put_nowait(payload)
    except Full:
        # Do not block training logs if Slack is slow/unreachable.
        # Best-effort: drop the message.
        return


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
    slack_attempts = int(os.environ.get("SLACK_NOTIFY_ATTEMPTS", "3"))
    slack_queue_max = int(os.environ.get("SLACK_NOTIFY_QUEUE_MAX", "100"))
    slack_drain_seconds = float(os.environ.get("SLACK_NOTIFY_DRAIN_SECONDS", "5"))

    buffer: deque[str] = deque()
    last_send = time.monotonic()
    last_progress_send = time.monotonic()
    # Keep recent context lines for checkpoint messages
    recent_context: deque[str] = deque(maxlen=5)

    slack_q: Queue[str] = Queue(maxsize=slack_queue_max)
    stop = Event()
    worker = Thread(
        target=_slack_worker,
        args=(slack_q, stop),
        kwargs={"attempts": slack_attempts},
        daemon=True,
    )
    worker.start()

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
        _enqueue(slack_q, payload)

        last_send = time.monotonic()

    def send_checkpoint_notification(checkpoint_line: str) -> None:
        """Enqueue a checkpoint notification to Slack."""
        # Include recent context lines for better understanding
        # Note: recent_context already includes checkpoint_line, so we use it directly
        context_lines = list(recent_context)
        context_text = "\n".join(context_lines) if context_lines else checkpoint_line

        payload = f"✅ CHECKPOINT: {job_name}\n{now_iso()}\n```\n{context_text}\n```"
        _enqueue(slack_q, payload)

    with log_path.open("a", encoding="utf-8", errors="replace") as f:
        start_msg = f"{job_name} started {now_iso()}"
        _enqueue(slack_q, start_msg)

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
        _enqueue(slack_q, done_msg)

    # Best-effort drain. Never block indefinitely (backpressure here can hang training).
    deadline = time.monotonic() + max(0.0, slack_drain_seconds)
    while (not slack_q.empty()) and time.monotonic() < deadline:
        time.sleep(0.1)
    stop.set()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
