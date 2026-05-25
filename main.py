import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from train import train_loop
from eval import eval, eval_expert
from loaded_gaussian_policy import LoadedGaussianPolicy


# ---------------------------------------------------
# Config
# ---------------------------------------------------

ENV_NAME = "HalfCheetah-v4"
DATASET_PATH = "expert_data_HalfCheetah-v2.pkl"
EXPERT_PATH = "HalfCheetah.pkl"

DAGGER_ITERATIONS = 5
ROLLOUT_EPISODES = 2
NUM_EPOCHS = 50
NUM_EVAL_EPISODES = 20


# ---------------------------------------------------
# Load expert dataset
# ---------------------------------------------------

with open(DATASET_PATH, "rb") as f:
    dataset = pickle.load(f)


# ---------------------------------------------------
# Train BC + DAgger
# ---------------------------------------------------

train_loop(
    dataset,
    dagger=True,
    env_name=ENV_NAME,
    dagger_iterations=DAGGER_ITERATIONS,
    rollout_episodes=ROLLOUT_EPISODES,
    num_epoch=NUM_EPOCHS,
    expert_path=EXPERT_PATH,
    save_name="dagger.pth"
)


# ---------------------------------------------------
# Evaluate expert
# ---------------------------------------------------

expert = LoadedGaussianPolicy(EXPERT_PATH)
expert.eval()

expert_stats = eval_expert(
    expert,
    env_name=ENV_NAME,
    num_episodes=NUM_EVAL_EPISODES
)

expert_mean = expert_stats["mean"]
expert_std = expert_stats["std"]

print(f"Expert | mean: {expert_mean:.1f} | std: {expert_std:.1f}")


# ---------------------------------------------------
# Evaluate BC baseline
# ---------------------------------------------------

bc_stats = eval(
    "bc_only.pth",
    num_episodes=NUM_EVAL_EPISODES,
    env_name=ENV_NAME
)

bc_mean = bc_stats["mean"]
bc_std = bc_stats["std"]

print(f"BC | mean: {bc_mean:.1f} | std: {bc_std:.1f}")


# ---------------------------------------------------
# Evaluate DAgger checkpoints
# ---------------------------------------------------

dagger_checkpoints = [
    f"dagger_iter{i}.pth"
    for i in range(1, DAGGER_ITERATIONS + 1)
]

dagger_means = []
dagger_stds = []

for ckpt in dagger_checkpoints:
    if not os.path.exists(ckpt):
        print(f"Skipping missing checkpoint: {ckpt}")
        continue

    stats = eval(
        ckpt,
        num_episodes=NUM_EVAL_EPISODES,
        env_name=ENV_NAME
    )

    dagger_means.append(stats["mean"])
    dagger_stds.append(stats["std"])

    print(
        f"{ckpt} | mean: {stats['mean']:.1f} | std: {stats['std']:.1f}"
    )



# ---------------------------------------------------
# Plot DAgger performance
# ---------------------------------------------------

iterations = np.arange(1, len(dagger_means) + 1)

plt.figure(figsize=(8, 5))

# Expert baseline
plt.axhline(
    y=expert_mean,
    linestyle="--",
    linewidth=2,
    color="green",
    label=f"Expert Mean: {expert_mean:.1f}"
)

# BC baseline
plt.axhline(
    y=bc_mean,
    linestyle="--",
    linewidth=2,
    color="red",
    label=f"BC Mean: {bc_mean:.1f}"
)

plt.ylim(3800, 4250)
# DAgger curve
plt.plot(
    iterations,
    dagger_means,
    marker="o",
    linewidth=2,
    color="blue",
    label="DAgger"
)

plt.xlabel("DAgger Iteration")
plt.ylabel("Average Return")
plt.title("DAgger Performance Improvement on HalfCheetah")

plt.xticks(iterations)

plt.legend()
plt.grid(True)

plt.savefig(
    "dagger_training_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ---------------------------------------------------
# Save rollout videos
# ---------------------------------------------------

eval(
    "bc_only.pth",
    num_episodes=3,
    save_video=True,
    video_tag="bc_HalfCheetah",
    env_name=ENV_NAME
)

eval(
    "dagger.pth",
    num_episodes=3,
    save_video=True,
    video_tag="dagger_HalfCheetah",
    env_name=ENV_NAME
)