import os
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from policy import BCPolicy  # your policy class
import numpy as np
import pickle

os.environ["MUJOCO_GL"] = "glfw"

def load_model(path):
    checkpoint = torch.load(path, weights_only=False)

    # dimensions
    obs_dim = checkpoint["obs_dim"]
    act_dim = checkpoint["act_dim"]

    # create policy and load weights
    policy = BCPolicy(obs_dim, act_dim)
    policy.load_state_dict(checkpoint["model_state_dict"])

    # load normalization stats
    obs_mean = checkpoint["obs_mean"]
    obs_std = checkpoint["obs_std"]
    return policy, obs_mean, obs_std

# ========================
# 2. Create environment with video recording
# ========================

# ========================
# 3. Run evaluation
# ========================
def eval(model_path="bc_policy.pth", num_episodes=10, save_video=False, video_tag="model", env_name="HalfCheetah-v4"):
    policy, obs_mean, obs_std = load_model(model_path)
    
    if save_video:
        video_folder = f"./videos_{video_tag}"
        env = gym.make(env_name, render_mode="rgb_array")
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True)
    else:
        env = gym.make(env_name, render_mode=None)
    
    rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done, truncated, total_reward = False, False, 0
        while not (done or truncated):
            obs_norm = (obs - obs_mean) / obs_std
            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = policy(obs_tensor).squeeze(0).numpy()
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        print(f"Episode {ep+1}: Total Reward = {total_reward:.1f}")
    
    env.close()
    return {"mean": np.mean(rewards), "std": np.std(rewards), "all": rewards}

    

# ========================
# 4. Check saved videos
# ========================
# print("Saved evaluation videos:", os.listdir("./videos_eval"))