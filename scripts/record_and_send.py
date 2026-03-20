"""Record best model playing at full resolution and send video to Telegram."""
import glob
import os
import sys

import cv2
import numpy as np
from stable_baselines3 import PPO
from golds.environments.factory import EnvironmentFactory
from golds.environments.registry import GameRegistry

game = sys.argv[1] if len(sys.argv) > 1 else "ms_pacman"
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
SCALE = 4

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

# Create a SINGLE unwrapped env for both playing and recording
game_info = GameRegistry.get(game)

if game_info.platform == "atari":
    import gymnasium as gym
    import ale_py
    if f"{game_info.env_id}" not in gym.envs.registry:
        ale_py.register_v0_v4_envs()
    raw_env = gym.make(game_info.env_id, render_mode="rgb_array")
else:
    import retro
    raw_env = retro.make(
        game=game_info.env_id,
        state=game_info.default_state or retro.State.DEFAULT,
        render_mode="rgb_array",
    )

# Also create the wrapped env for the model
env = EnvironmentFactory.create_eval_env(game_id=game, frame_stack=4, seed=0)
model = PPO.load(model_path, env=env)

# Play in the wrapped env, render from raw env by replaying actions
obs = env.reset()
raw_env.reset()

os.makedirs("videos", exist_ok=True)
mp4_path = f"videos/{game}.mp4"
writer = None

print(f"Recording {steps} frames (upscaled {SCALE}x)...")
for i in range(steps):
    # Predict action from the model
    action, _ = model.predict(obs, deterministic=True)

    # Step the wrapped env (model's env)
    obs, _, dones, infos = env.step(action)

    # Render from raw env AFTER stepping
    # We need to get the raw frame somehow - use the wrapped env's underlying render
    # The VecEnv wraps a DummyVecEnv which wraps the actual gym env
    # Access the underlying env chain to get the raw render
    try:
        # Walk the wrapper chain to get the base ALE/retro env render
        inner = env
        while hasattr(inner, 'venv'):
            inner = inner.venv
        while hasattr(inner, 'envs'):
            inner = inner.envs[0]
        # Now inner should be the gym.Env (possibly wrapped)
        # Get the unwrapped env for rendering
        base = inner.unwrapped if hasattr(inner, 'unwrapped') else inner
        frame = base.render()
    except Exception:
        frame = None

    if frame is not None:
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # Upscale
        frame = cv2.resize(frame, (frame.shape[1] * SCALE, frame.shape[0] * SCALE),
                           interpolation=cv2.INTER_NEAREST)
        if writer is None:
            h, w = frame.shape[0], frame.shape[1]
            writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            print(f"Video: {w}x{h}")
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if dones[0]:
        obs = env.reset()

env.close()
raw_env.close()
if writer:
    writer.release()
    print(f"Saved: {mp4_path}")
else:
    print("ERROR: No frames captured")
    sys.exit(1)

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
    print("Sent to Telegram!")
except Exception as e:
    print(f"Telegram send failed: {e}")
    print(f"Video is at: {mp4_path}")
