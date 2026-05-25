import pickle
from train import train_loop
from eval import eval
import os
path = 'expert_data_HalfCheetah-v2.pkl'
with open(path, "rb") as f:
    dataset = pickle.load(f)

# Evaluate both

bc_stats  = eval("bc_only.pth", num_episodes=20, env_name="HalfCheetah-v4")
dag_stats = eval("dagger.pth",  num_episodes=20, env_name="HalfCheetah-v4")

print(f"BC     | mean: {bc_stats['mean']:.1f}  std: {bc_stats['std']:.1f}")
print(f"DAgger | mean: {dag_stats['mean']:.1f}  std: {dag_stats['std']:.1f}")

# Optional: save videos of both
eval("bc_only.pth", num_episodes=3, save_video=True, video_tag="bc_HalfCheetah",    env_name="HalfCheetah-v4")
eval("dagger.pth",  num_episodes=3, save_video=True, video_tag="dagger_HalfCheetah", env_name="HalfCheetah-v4")

