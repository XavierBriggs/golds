"""Record best model playing and send video to Telegram using VecVideoRecorder."""
import glob
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder
from golds.environments.factory import EnvironmentFactory
from golds.environments.registry import GameRegistry

game = sys.argv[1] if len(sys.argv) > 1 else "ms_pacman"
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

# Find latest best model (unified then legacy directory layout)
candidates = sorted(glob.glob(f"outputs/{game}*/best/best_model.zip"))
if not candidates:
    candidates = sorted(glob.glob(f"outputs/{game}*/{game}/best/best_model.zip"))
if not candidates:
    candidates = sorted(glob.glob(f"outputs/{game}*/models/final_model.zip"))
if not candidates:
    candidates = sorted(glob.glob(f"outputs/{game}*/{game}/models/final_model.zip"))
if not candidates:
    candidates = sorted(glob.glob(f"outputs/{game}*/models/checkpoints/*.zip"))
if not candidates:
    candidates = sorted(glob.glob(f"outputs/{game}*/{game}/models/checkpoints/*.zip"))
if not candidates:
    print(f"No model found for {game}")
    sys.exit(1)

model_path = candidates[-1]
print(f"Model: {model_path}")

os.makedirs("videos", exist_ok=True)

env = EnvironmentFactory.create_eval_env(game_id=game, frame_stack=4, seed=0)
env = VecVideoRecorder(
    env,
    "videos/",
    record_video_trigger=lambda x: x == 0,
    video_length=steps,
    name_prefix=game,
)

model = PPO.load(model_path, env=env)
obs = env.reset()

print(f"Recording {steps} frames...")
for _ in range(steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, dones, _ = env.step(action)
    if dones[0]:
        obs = env.reset()

env.close()

# Find the generated mp4
mp4_files = sorted(glob.glob(f"videos/{game}*.mp4"), key=os.path.getmtime, reverse=True)
if not mp4_files:
    print("ERROR: No video generated")
    sys.exit(1)

mp4_path = mp4_files[0]
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
