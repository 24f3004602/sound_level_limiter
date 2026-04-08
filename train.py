"""
train.py
Train baseline agents on the Sound Limiter environment.

Default behavior prefers a PyTorch DQN baseline and automatically falls back
to tabular Q-learning when torch is unavailable.

Run:
  python train.py
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent.q_agent import QLearningAgent
from agent.dqn_agent import DQNAgent, torch_is_available
from environment.sound_env import SoundLimiterEnv
from environment.tasks import TASK_EASY, TASK_HARD, TASK_MEDIUM, grade_all_tasks


def _sample_curriculum_task(rng: np.random.Generator):
    task_curriculum = [TASK_EASY, TASK_MEDIUM, TASK_HARD]
    task_weights = [0.25, 0.35, 0.40]
    task_idx = int(rng.choice(len(task_curriculum), p=task_weights))
    return task_curriculum[task_idx]


def _resolve_algorithm(requested: str) -> str:
    normalized = requested.strip().lower()
    if normalized not in {"auto", "dqn", "q_table"}:
        raise ValueError("algorithm must be one of: auto, dqn, q_table")

    if normalized == "auto":
        return "dqn" if torch_is_available() else "q_table"

    if normalized == "dqn" and not torch_is_available():
        raise RuntimeError("Requested DQN training but PyTorch is not available")

    return normalized


def train(
    n_episodes: int = 900,
    render_every: int = 100,
    seed: int = 42,
    algorithm: str = "auto",
):
    """Train baseline model and return (algo, trained_agent, rewards, model_path)."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    algo = _resolve_algorithm(algorithm)
    rewards: list[float] = []

    if algo == "dqn":
        agent = DQNAgent(
            learning_rate=8e-4,
            discount=0.98,
            epsilon=1.0,
            epsilon_decay=0.997,
            epsilon_min=0.05,
            batch_size=64,
            replay_capacity=20_000,
            target_sync_every=180,
            seed=seed,
        )
        model_path = "dqn_model.pt"
        print("Training baseline agent (PyTorch DQN)...\n")
    else:
        agent = QLearningAgent(
            learning_rate=0.14,
            discount=0.96,
            epsilon=1.0,
            epsilon_decay=0.996,
            epsilon_min=0.03,
        )
        model_path = "q_table.pkl"
        print("Training baseline agent (Tabular Q-learning)...\n")

    for ep in range(1, n_episodes + 1):
        task = _sample_curriculum_task(rng)

        init_sound = float(np.clip(task.initial_sound + rng.normal(0, 2.0), 0.0, 100.0))
        noise_std = float(max(0.5, task.noise_std + rng.normal(0, 0.5)))
        ep_seed = int(rng.integers(0, 1_000_000))

        env = SoundLimiterEnv(
            initial_sound=init_sound,
            noise_std=noise_std,
            max_steps=task.max_steps,
            seed=ep_seed,
        )

        obs = env.reset(seed=ep_seed)
        done = False
        total_reward = 0.0

        while not done:
            action = agent.choose_action(obs.model_dump())
            next_obs, reward, done, _ = env.step(action)
            agent.learn(obs.model_dump(), action, reward.value, next_obs.model_dump(), done)
            obs = next_obs
            total_reward += reward.value

        agent.decay_epsilon()
        rewards.append(total_reward)

        if ep % render_every == 0:
            avg = float(np.mean(rewards[-render_every:]))
            print(
                f"  Episode {ep:4d}/{n_episodes} | "
                f"Avg Reward: {avg:8.3f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    agent.save(model_path)
    print(f"\nTraining complete! Saved -> {model_path}\n")
    return algo, agent, rewards, model_path


def plot_training(rewards: list[float], algorithm: str, window: int = 50) -> None:
    smoothed = [
        float(np.mean(rewards[max(0, i - window) : i + 1]))
        for i in range(len(rewards))
    ]

    plt.figure(figsize=(10, 4))
    plt.plot(rewards, alpha=0.25, color="#127369", label="Raw reward")
    plt.plot(smoothed, color="#127369", linewidth=2.0, label=f"{window}-ep moving avg")
    plt.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    plt.title(f"Sound Limiter - {algorithm.upper()} Training Progress", fontsize=13)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_progress.png", dpi=120)
    print("Training chart saved -> training_progress.png")


def evaluate_saved_model(algo: str, model_path: str, seed: int = 42):
    if algo == "dqn":
        eval_agent = DQNAgent(epsilon=0.0, seed=seed)
        loaded_ok = eval_agent.load(model_path)
        if not loaded_ok:
            raise RuntimeError(f"Could not load {model_path} for evaluation")
        eval_agent.set_eval_mode()
        return grade_all_tasks(
            agent_fn=lambda obs_dict: eval_agent.choose_action(obs_dict),
            n_episodes=10,
        )

    eval_agent = QLearningAgent(epsilon=0.0)
    loaded_ok = eval_agent.load(model_path)
    if not loaded_ok:
        raise RuntimeError(f"Could not load {model_path} for evaluation")
    eval_agent.epsilon = 0.0
    return grade_all_tasks(
        agent_fn=lambda obs_dict: eval_agent.choose_action(obs_dict),
        n_episodes=10,
    )


if __name__ == "__main__":
    requested_algo = os.environ.get("BASELINE_ALGO", "auto")
    algo, _trained_agent, rewards, model_path = train(
        n_episodes=900,
        render_every=100,
        seed=42,
        algorithm=requested_algo,
    )

    plot_training(rewards, algorithm=algo)

    print(f"Evaluating trained {algo.upper()} baseline on registered tasks...\n")
    results = evaluate_saved_model(algo=algo, model_path=model_path)

    print("\n-- Baseline Scores -------------------------------------")
    for _tid, result in results.items():
        print(f"  {result.difficulty:6s} | {result.task_name:30s} | score = {result.score:.4f}")
    print("--------------------------------------------------------\n")
