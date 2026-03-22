"""Telegram notification backend for training alerts."""

from __future__ import annotations

import os
import urllib.parse
import urllib.request
from pathlib import Path


class TelegramNotifier:
    """Send training notifications via Telegram Bot API.

    Configure via environment variables:
        GOLDS_TELEGRAM_BOT_TOKEN: Bot token from @BotFather
        GOLDS_TELEGRAM_CHAT_ID: Chat ID to send messages to
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        self.bot_token = bot_token or os.environ.get(
            "GOLDS_TELEGRAM_BOT_TOKEN", "8687875312:AAEj8oBwy00549K1OP7zV8rhOXYZxyqJnk8"
        )
        self.chat_id = chat_id or os.environ.get(
            "GOLDS_TELEGRAM_CHAT_ID", "6518859577"
        )
        self._enabled = bool(self.bot_token and self.chat_id)

    @property
    def enabled(self) -> bool:
        """Whether Telegram notifications are configured."""
        return self._enabled

    def send(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send a message via Telegram.

        Args:
            message: Message text (supports HTML formatting).
            parse_mode: Telegram parse mode ("HTML" or "Markdown").

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self._enabled:
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = urllib.parse.urlencode(
            {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode,
            }
        ).encode()

        try:
            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception:
            return False

    def send_video(self, video_path: str, caption: str = "", parse_mode: str = "HTML") -> bool:
        """Send a video file via Telegram.

        Args:
            video_path: Path to the MP4 file.
            caption: Optional caption (supports HTML).
            parse_mode: Telegram parse mode.

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self._enabled:
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendVideo"
        boundary = "----GoldsBoundary"

        try:
            with open(video_path, "rb") as f:
                video_data = f.read()

            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="chat_id"\r\n\r\n'
                f"{self.chat_id}\r\n"
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="parse_mode"\r\n\r\n'
                f"{parse_mode}\r\n"
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="caption"\r\n\r\n'
                f"{caption}\r\n"
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="video"; '
                f'filename="{Path(video_path).name}"\r\n'
                f"Content-Type: video/mp4\r\n\r\n"
            ).encode() + video_data + f"\r\n--{boundary}--\r\n".encode()

            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.status == 200
        except Exception:
            return False

    def send_training_start(
        self,
        experiment_name: str,
        game_id: str,
        total_timesteps: int,
        device: str = "auto",
    ) -> bool:
        """Send training start notification."""
        msg = (
            f"\U0001f3ae <b>Training Started</b>\n"
            f"Experiment: <code>{experiment_name}</code>\n"
            f"Game: {game_id}\n"
            f"Timesteps: {total_timesteps:,}\n"
            f"Device: {device}"
        )
        return self.send(msg)

    def send_training_complete(
        self,
        experiment_name: str,
        game_id: str,
        wall_time_seconds: float,
        best_reward: float | None = None,
        total_timesteps: int = 0,
    ) -> bool:
        """Send training completion notification."""
        hours = wall_time_seconds / 3600
        msg = (
            f"\u2705 <b>Training Complete</b>\n"
            f"Experiment: <code>{experiment_name}</code>\n"
            f"Game: {game_id}\n"
            f"Duration: {hours:.1f}h\n"
            f"Timesteps: {total_timesteps:,}"
        )
        if best_reward is not None:
            msg += f"\nBest Reward: {best_reward:.1f}"
        return self.send(msg)

    def send_training_failed(
        self,
        experiment_name: str,
        game_id: str,
        error: str,
    ) -> bool:
        """Send training failure notification."""
        # Truncate error to avoid Telegram message limits
        if len(error) > 500:
            error = error[:500] + "..."
        msg = (
            f"\u274c <b>Training Failed</b>\n"
            f"Experiment: <code>{experiment_name}</code>\n"
            f"Game: {game_id}\n"
            f"Error: <pre>{error}</pre>"
        )
        return self.send(msg)
