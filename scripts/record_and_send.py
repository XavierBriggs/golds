"""Record best model playing at full resolution and send video to Telegram."""
import glob
import os
import sys

import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from golds.environments.registry import GameRegistry

game = sys.argv[1] if len(sys.argv) > 1 else "ms_pacman"
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

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

game_info = GameRegistry.get(game)

if game_info.platform == "atari":
    # Create env with render_mode for recording
    base_env = gym.make(game_info.env_id, render_mode="rgb_array")

    # Wrap for the model (AtariWrapper does preprocessing)
    wrapped = AtariWrapper(base_env)
    vec_env = DummyVecEnv([lambda: wrapped])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecTransposeImage(vec_env)

    model = PPO.load(model_path, env=vec_env)
    obs = vec_env.reset()

    os.makedirs("videos", exist_ok=True)
    mp4_path = f"videos/{game}.mp4"
    writer = None

    print(f"Recording {steps} frames...")
    for i in range(steps):
        # Get the ORIGINAL frame from the base env (before wrappers)
        frame = base_env.render()
        if frame is not None:
            frame = np.asarray(frame)
            if writer is None:
                h, w = frame.shape[0], frame.shape[1]
                writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                print(f"Video: {w}x{h}")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = vec_env.step(action)
        if dones[0]:
            obs = vec_env.reset()

    vec_env.close()
    if writer:
        writer.release()

else:
    # Retro games
    try:
        import retro
    except ImportError:
        print("stable-retro not installed")
        sys.exit(1)

    base_env = retro.make(
        game=game_info.env_id,
        state=game_info.default_state or retro.State.DEFAULT,
        render_mode="rgb_array",
    )

    # Import retro preprocessing
    from golds.environments.retro.maker import RetroPreprocessing, FrameSkip
    from stable_baselines3.common.monitor import Monitor

    proc_env = Monitor(base_env)
    proc_env = FrameSkip(proc_env, skip=4)
    proc_env = RetroPreprocessing(proc_env)
    vec_env = DummyVecEnv([lambda: proc_env])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecTransposeImage(vec_env)

    model = PPO.load(model_path, env=vec_env)
    obs = vec_env.reset()

    os.makedirs("videos", exist_ok=True)
    mp4_path = f"videos/{game}.mp4"
    writer = None

    print(f"Recording {steps} frames...")
    for i in range(steps):
        frame = base_env.render()
        if frame is not None:
            frame = np.asarray(frame)
            if writer is None:
                h, w = frame.shape[0], frame.shape[1]
                writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                print(f"Video: {w}x{h}")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = vec_env.step(action)
        if dones[0]:
            obs = vec_env.reset()

    vec_env.close()
    if writer:
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
    print("Sent to Telegram!")
except Exception as e:
    print(f"Telegram send failed: {e}")
    print(f"Video is at: {mp4_path}")
