Golds training system files

Files included
scripts/run_queue.py
scripts/gpu_count.py
configs/queue.yaml
slack_logger/slack_notify.py
slack_logger/slack_log_tee.py

Basic usage
1. Set SLACK_WEBHOOK_URL in your shell
2. Run uv run python scripts/gpu_count.py
3. Run uv run python scripts/run_queue.py
4. (Optional) Test Slack: uv run python scripts/slack_test.py

Environment variables
SLACK_WEBHOOK_URL
SLACK_NOTIFY_DEBUG (set to 1 to print HTTP failure codes)
SLACK_NOTIFY_INTERVAL_SECONDS default 20
SLACK_NOTIFY_MAX_LINES default 25
SLACK_NOTIFY_MAX_CHARS default 3500
LOG_DIR default logs
RUN_META_DIR default runs_meta
GOLDS_TRAIN_OUTPUT_DIR (override output root passed to `golds train run --output`)
GOLDS_QUEUE_RESUME_LATEST (set to 1 to force resume-from-latest for all jobs)

Queue config
- configs/queue.yaml supports:
  - default.resume_latest: true/false (auto `--resume` latest valid checkpoint)
  - jobs[].resume_latest: true/false (per-job override)
