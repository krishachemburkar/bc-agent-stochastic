import os
os.environ["MUJOCO_GL"] = "glfw"

import gymnasium as gym
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from policy import BCPolicy
from loaded_gaussian_policy import LoadedGaussianPolicy


def train_loop(dataset, dagger=False, env_name="HalfCheetah-v4",
               dagger_iterations=3, rollout_episodes=2, batch_size=256,
               lr=1e-3, num_epoch=100, expert_path="HalfCheetah.pkl",
               save_name="bc_model.pth"):

    model_path = save_name
    
    env = gym.make(env_name, render_mode=None)

    observations = []
    actions = []

    for traj in dataset:
        observations.append(traj['observation'])
        actions.append(traj['action'])

    observations = np.concatenate(observations, axis=0)
    actions = np.concatenate(actions, axis=0)

    obs_mean = np.mean(observations, axis=0)
    obs_std = np.std(observations, axis=0) + 1e-8

    observations_norm = (observations - obs_mean) / obs_std

    obs_tensor = torch.tensor(observations_norm, dtype=torch.float32)
    act_tensor = torch.tensor(actions, dtype=torch.float32)

    train_dataset = TensorDataset(obs_tensor, act_tensor)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    policy = BCPolicy(observations_norm.shape[1], actions.shape[1])
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print("=== Vanilla BC training ===")

    for epoch in range(num_epoch):
        total_loss = 0

        for obs_batch, act_batch in loader:
            pred = policy(obs_batch)
            loss = loss_fn(pred, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

    # Save vanilla BC model
    torch.save({
        "model_state_dict": policy.state_dict(),
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "obs_dim": observations.shape[1],
        "act_dim": actions.shape[1]
    }, "bc_only.pth")

    print("Saved vanilla BC model as bc_only.pth")

    if dagger:
        print("=== DAgger training ===")

        expert = LoadedGaussianPolicy(expert_path)
        expert.eval()

        for iter_idx in range(dagger_iterations):
            print(f"\n=== DAgger Iteration {iter_idx+1} ===")

            policy.eval()
            collected_states = []

            for ep in range(rollout_episodes):
                obs, _ = env.reset()
                done = False
                truncated = False

                # while not (done or truncated):
                #     obs_norm = (obs - obs_mean) / obs_std
                #     obs_tensor_step = torch.tensor(
                #         obs_norm,
                #         dtype=torch.float32
                #     ).unsqueeze(0)

                #     with torch.no_grad():
                #         action = policy(obs_tensor_step).squeeze(0).numpy()

                #     next_obs, reward, done, truncated, _ = env.step(action)

                #     collected_states.append(next_obs.copy())

                #     obs = next_obs
                while not (done or truncated):
                    obs_norm = (obs - obs_mean) / obs_std
                    obs_tensor_step = torch.tensor(
                        obs_norm,
                        dtype=torch.float32
                    ).unsqueeze(0)

                    with torch.no_grad():
                        action = policy(obs_tensor_step).squeeze(0).numpy()

                    next_obs, reward, done, truncated, _ = env.step(action)

                    collected_states.append(next_obs.copy())

                    obs = next_obs

            expert_actions = [expert(s) for s in collected_states]

            expert_actions_np = np.array([
                a.cpu().detach().squeeze().numpy()
                for a in expert_actions
            ])

            new_obs = np.array(collected_states)

            observations = np.concatenate([observations, new_obs], axis=0)
            actions = np.concatenate([actions, expert_actions_np], axis=0)

            obs_mean = np.mean(observations, axis=0)
            obs_std = np.std(observations, axis=0) + 1e-8

            observations_norm = (observations - obs_mean) / obs_std

            obs_tensor = torch.tensor(observations_norm, dtype=torch.float32)
            act_tensor = torch.tensor(actions, dtype=torch.float32)

            loader = DataLoader(
                TensorDataset(obs_tensor, act_tensor),
                batch_size=batch_size,
                shuffle=True
            )

            policy.train()

            for epoch in range(num_epoch):
                total_loss = 0

                for obs_batch, act_batch in loader:
                    pred = policy(obs_batch)
                    loss = loss_fn(pred, act_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                print(
                    f"[DAgger Iter {iter_idx+1}] "
                    f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}"
                )

            # Save checkpoint after each DAgger iteration
            iter_model_path = f"dagger_iter{iter_idx+1}.pth"

            torch.save({
                "model_state_dict": policy.state_dict(),
                "obs_mean": obs_mean,
                "obs_std": obs_std,
                "obs_dim": observations.shape[1],
                "act_dim": actions.shape[1]
            }, iter_model_path)

            print(f"Saved DAgger checkpoint: {iter_model_path}")

        env.close()

    print("=== Saving Final Model ===")

    torch.save({
        "model_state_dict": policy.state_dict(),
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "obs_dim": observations.shape[1],
        "act_dim": actions.shape[1]
    }, model_path)

    print(f"Training complete and model saved as {model_path}!")

    return model_path