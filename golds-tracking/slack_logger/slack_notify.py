import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path


_WARNED_NO_WEBHOOK = False


def _project_root() -> Path:
    # golds-tracking/slack_logger/slack_notify.py -> golds-tracking/
    return Path(__file__).resolve().parents[1]


def _parse_dotenv(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        env[key] = value
    return env


def _maybe_load_env_from_dotenv() -> None:
    dotenv_path = _project_root() / ".env"
    if not dotenv_path.exists():
        return
    try:
        parsed = _parse_dotenv(dotenv_path.read_text(encoding="utf-8"))
    except Exception:
        return
    for k, v in parsed.items():
        os.environ.setdefault(k, v)


def get_webhook_url() -> str | None:
    _maybe_load_env_from_dotenv()
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if url and url.strip():
        return url.strip()
    return None


def post_message(text: str, username: str | None = None) -> bool:
    global _WARNED_NO_WEBHOOK
    url = get_webhook_url()
    if not url:
        if not _WARNED_NO_WEBHOOK:
            _WARNED_NO_WEBHOOK = True
            print(
                "Slack notifications disabled: `SLACK_WEBHOOK_URL` is not set "
                "(set it in your shell, or put it in `golds-tracking/.env`).",
                file=os.sys.stderr,
            )
        return False

    payload: dict[str, object] = {"text": text}
    if username:
        payload["username"] = username

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            _ = resp.read()
        return True
    except urllib.error.HTTPError as e:
        try:
            _ = e.read()
        except Exception:
            pass
        if os.environ.get("SLACK_NOTIFY_DEBUG", "").strip().lower() in {"1", "true", "yes"}:
            print(f"Slack notification failed: HTTP {e.code}", file=os.sys.stderr)
        return False
    except Exception:
        if os.environ.get("SLACK_NOTIFY_DEBUG", "").strip().lower() in {"1", "true", "yes"}:
            print("Slack notification failed: request error", file=os.sys.stderr)
        return False


def post_message_with_backoff(text: str, username: str | None = None, attempts: int = 5) -> bool:
    # If Slack isn't configured, don't waste time sleeping/backing off.
    if not get_webhook_url():
        return False
    delay = 1.0
    for _ in range(max(1, attempts)):
        ok = post_message(text=text, username=username)
        if ok:
            return True
        time.sleep(delay)
        delay = min(20.0, delay * 2.0)
    return False
