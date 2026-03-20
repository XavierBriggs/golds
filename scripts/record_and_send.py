"""Record best model playing and send video to Telegram."""
import glob
import os
import sys

import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from golds.environments.factory import EnvironmentFactory
from golds.environments.registry import GameRegistry

game = sys.argv[1] if len(sys.argv) > 1 else "ms_pacman"
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

# Find latest best model
candidates = sorted(glob.glob(f"outputs/{game}*/best/best_model.zip"))
if not candidates:
    candidates = sorted(glob.glob(f"outputs/{game}*/{game}/best/best_model.zip"))
if not candidates:
    candidates = sorted(glob.glob(f"outputs/{game}*/models/final_model.zip"))
if not candidates:
    candidates = sorted(glob.glob(f"outputs/{game}*/{game}/models/final_model.zip"))
if not candidates:
    print(f"No model found for {game}")
    sys.exit(1)

model_path = candidates[-1]
print(f"Model: {model_path}")

# Create the wrapped env for the model to use (84x84 grayscale)
env = EnvironmentFactory.create_eval_env(game_id=game, frame_stack=4, seed=0)
model = PPO.load(model_path, env=env)

# Create a raw render env for recording (full color, full resolution)
game_info = GameRegistry.get(game)
if game_info.platform == "atari":
    raw_env = gym.make(game_info.env_id, render_mode="rgb_array")
else:
    # Retro games
    try:
        import retro
        raw_env = retro.make(game=game_info.env_id, render_mode="rgb_array")
    except Exception:
        raw_env = None

obs = env.reset()
if raw_env is not None:
    raw_env.reset()

os.makedirs("videos", exist_ok=True)
mp4_path = f"videos/{game}.mp4"
writer = None

for i in range(steps):
    # Get raw frame for video
    if raw_env is not None:
        frame = raw_env.render()
        if frame is None:
            frame = np.asarray(env.get_images()[0])
    else:
        frame = np.asarray(env.get_images()[0])

    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if writer is None:
        h, w = frame.shape[0], frame.shape[1]
        writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        print(f"Recording {w}x{h} video...")

    writer.write(frame)

    # Step both envs with same action
    action, _ = model.predict(obs, deterministic=True)
    obs, _, dones, _ = env.step(action)

    if raw_env is not None:
        try:
            # Convert vectorized action to scalar for raw env
            a = action[0] if hasattr(action, '__len__') else action
            raw_env.step(a)
        except Exception:
            pass

    if dones[0]:
        obs = env.reset()
        if raw_env is not None:
            raw_env.reset()

env.close()
if raw_env is not None:
    raw_env.close()
writer.release()
print(f"Saved: {mp4_path}")

# Send to Telegram
try:
    import urllib.request

    bot_token = "8687875312:AAEj8oBwy00549K1OP7zV8rhOXYZxyqJnk8"
    chat_id = "6518859577"

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
    resp = urllib.request.urlopen(req, timeout=60)
    print(f"Sent to Telegram!")
except Exception as e:
    print(f"Telegram send failed: {e}")
    print(f"Video is at: {mp4_path}")
