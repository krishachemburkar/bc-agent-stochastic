import os
os.environ["MUJOCO_GL"] = "glfw"

import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np

from policy import BCPolicy


def load_model(path):
    checkpoint = torch.load(path, weights_only=False)

    obs_dim = checkpoint["obs_dim"]
    act_dim = checkpoint["act_dim"]

    policy = BCPolicy(obs_dim, act_dim)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    obs_mean = checkpoint["obs_mean"]
    obs_std = checkpoint["obs_std"]

    return policy, obs_mean, obs_std


def eval_expert(
    expert,
    env_name="HalfCheetah-v4",
    num_episodes=20,
    save_video=False,
    video_tag="expert_HalfCheetah"
):
    if save_video:
        video_folder = f"./videos_{video_tag}"
        env = gym.make(env_name, render_mode="rgb_array")
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda episode_id: True
        )
    else:
        env = gym.make(env_name, render_mode=None)

    returns = []

    for ep in range(num_episodes):
        obs, _ = env.reset()

        done = False
        truncated = False
        total_reward = 0.0

        while not (done or truncated):
            with torch.no_grad():
                action = expert(obs)

            if torch.is_tensor(action):
                action = action.cpu().detach().numpy()

            action = np.asarray(action).squeeze()

            obs, reward, done, truncated, _ = env.step(action)
            total_reward += float(reward)

        returns.append(total_reward)
        print(f"[Expert] Episode {ep + 1}: Total Reward = {total_reward:.1f}")

    env.close()

    return {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "all": returns
    }


def eval(
    model_path="bc_policy.pth",
    num_episodes=10,
    save_video=False,
    video_tag="model",
    env_name="HalfCheetah-v4"
):
    policy, obs_mean, obs_std = load_model(model_path)

    if save_video:
        video_folder = f"./videos_{video_tag}"
        env = gym.make(env_name, render_mode="rgb_array")
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda episode_id: True
        )
    else:
        env = gym.make(env_name, render_mode=None)

    rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset()

        done = False
        truncated = False
        total_reward = 0.0

        while not (done or truncated):
            obs_norm = (obs - obs_mean) / obs_std
            obs_tensor = torch.tensor(
                obs_norm,
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                action = policy(obs_tensor).squeeze(0).numpy()

            action = np.asarray(action).squeeze()

            obs, reward, done, truncated, _ = env.step(action)
            total_reward += float(reward)

        rewards.append(total_reward)
        print(f"[{model_path}] Episode {ep + 1}: Total Reward = {total_reward:.1f}")

    env.close()

    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "all": rewards
    }