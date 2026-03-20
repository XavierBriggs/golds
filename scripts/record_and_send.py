"""Record best model playing and send video to Telegram."""
import sys
import os
import cv2
import numpy as np
from stable_baselines3 import PPO
from golds.environments.factory import EnvironmentFactory

game = sys.argv[1] if len(sys.argv) > 1 else "ms_pacman"

# Find latest best model
import glob
candidates = sorted(glob.glob(f"outputs/{game}*/best/best_model.zip"))
if not candidates:
    candidates = sorted(glob.glob(f"outputs/{game}*/{game}/best/best_model.zip"))
if not candidates:
    candidates = sorted(glob.glob(f"outputs/{game}*/models/final_model.zip"))
if not candidates:
    print(f"No model found for {game}")
    sys.exit(1)

model_path = candidates[-1]
print(f"Model: {model_path}")

env = EnvironmentFactory.create_eval_env(game_id=game, frame_stack=4, seed=0)
model = PPO.load(model_path, env=env)
obs = env.reset()

os.makedirs("videos", exist_ok=True)
mp4_path = f"videos/{game}.mp4"
writer = None

for _ in range(5000):
    imgs = env.get_images()
    frame = np.asarray(imgs[0])
    if writer is None:
        h, w = frame.shape[0], frame.shape[1]
        writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    action, _ = model.predict(obs, deterministic=True)
    obs, _, dones, _ = env.step(action)
    if dones[0]:
        obs = env.reset()

env.close()
writer.release()
print(f"Saved: {mp4_path}")

# Send to Telegram
try:
    import urllib.request
    import json

    bot_token = "8687875312:AAEj8oBwy00549K1OP7zV8rhOXYZxyqJnk8"
    chat_id = "6518859577"

    # Multipart form upload
    boundary = "----GoldsBoundary"
    with open(mp4_path, "rb") as f:
        video_data = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="chat_id"\r\n\r\n'
        f"{chat_id}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="video"; filename="{game}.mp4"\r\n'
        f"Content-Type: video/mp4\r\n\r\n"
    ).encode() + video_data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"https://api.telegram.org/bot{bot_token}/sendVideo",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    resp = urllib.request.urlopen(req, timeout=30)
    print(f"Sent to Telegram: {resp.status == 200}")
except Exception as e:
    print(f"Telegram send failed: {e}")
    print(f"Video is at: {mp4_path}")
