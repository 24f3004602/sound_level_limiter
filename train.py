"""
train.py
Train baseline agents on the Sound Limiter environment.

Default behavior prefers a PyTorch DQN baseline and automatically falls back
to tabular Q-learning when torch is unavailable.

Run:
  python train.py
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent.q_agent import QLearningAgent
from agent.dqn_agent import DQNAgent, torch_is_available
from environment.sound_env import SoundLimiterEnv
from environment.tasks import TASK_EASY, TASK_HARD, TASK_MEDIUM, grade_all_tasks, grade_task, list_tasks


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


def _moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    return [
        float(np.mean(values[max(0, i - window + 1) : i + 1]))
        for i in range(len(values))
    ]


@contextmanager
def _evaluation_mode(agent):
    """Temporarily force deterministic action selection for evaluation."""
    previous_epsilon = getattr(agent, "epsilon", None)
    previous_policy_training = None
    previous_target_training = None

    if previous_epsilon is not None:
        agent.epsilon = 0.0

    if hasattr(agent, "policy_net") and hasattr(agent, "target_net"):
        previous_policy_training = bool(agent.policy_net.training)
        previous_target_training = bool(agent.target_net.training)
        agent.policy_net.eval()
        agent.target_net.eval()

    try:
        yield
    finally:
        if previous_epsilon is not None:
            agent.epsilon = float(previous_epsilon)

        if hasattr(agent, "policy_net") and hasattr(agent, "target_net"):
            if previous_policy_training:
                agent.policy_net.train()
            if previous_target_training:
                agent.target_net.train()


def evaluate_agent(agent, n_episodes: int = 6, seed: int = 42) -> dict[str, Any]:
    """Run deterministic evaluation across all registered tasks."""
    task_results = {}

    with _evaluation_mode(agent):
        for idx, task in enumerate(list_tasks()):
            task_seed = seed + (idx * 97)
            result = grade_task(
                task=task,
                agent_fn=lambda obs_dict: agent.choose_action(obs_dict),
                n_episodes=n_episodes,
                seed=task_seed,
            )
            task_results[task.id] = {
                "task_name": result.task_name,
                "difficulty": result.difficulty,
                "score": float(result.score),
                "passed": bool(result.passed),
                "avg_safe_fraction": float(result.avg_safe_fraction),
                "avg_total_reward": float(result.avg_total_reward),
            }

    all_scores = [entry["score"] for entry in task_results.values()]
    all_rewards = [entry["avg_total_reward"] for entry in task_results.values()]
    all_passes = [1.0 if entry["passed"] else 0.0 for entry in task_results.values()]

    return {
        "avg_score": float(np.mean(all_scores)) if all_scores else 0.0,
        "avg_total_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "success_rate": float(np.mean(all_passes)) if all_passes else 0.0,
        "task_results": task_results,
    }


def train(
    n_episodes: int = 900,
    render_every: int = 100,
    eval_every: int = 100,
    eval_episodes: int = 6,
    seed: int = 42,
    algorithm: str = "auto",
):
    """Train baseline model and return metrics-rich training artifacts."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    algo = _resolve_algorithm(algorithm)
    if algorithm.strip().lower() == "auto":
        if algo == "dqn":
            print("Auto mode selected DQN (PyTorch available).")
        else:
            print("Auto mode selected Q-learning fallback (PyTorch unavailable).")
    rewards: list[float] = []
    train_successes: list[float] = []
    eval_history: list[dict[str, Any]] = []

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
        steps_in_safe = 0
        total_steps = 0

        while not done:
            action = agent.choose_action(obs.model_dump())
            next_obs, reward, done, _ = env.step(action)
            agent.learn(obs.model_dump(), action, reward.value, next_obs.model_dump(), done)
            obs = next_obs
            total_reward += reward.value
            total_steps += 1
            if reward.in_safe_zone:
                steps_in_safe += 1

        agent.decay_epsilon()
        rewards.append(total_reward)

        safe_fraction = (steps_in_safe / total_steps) if total_steps > 0 else 0.0
        train_successes.append(1.0 if safe_fraction >= task.success_threshold else 0.0)

        if ep % render_every == 0:
            avg = float(np.mean(rewards[-render_every:]))
            success = float(np.mean(train_successes[-render_every:]))
            print(
                f"  Episode {ep:4d}/{n_episodes} | "
                f"Avg Reward: {avg:8.3f} | "
                f"Success: {success:6.1%} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

        should_eval = (ep % eval_every == 0) or (ep == n_episodes)
        if should_eval:
            snapshot = evaluate_agent(agent=agent, n_episodes=eval_episodes, seed=seed + ep)
            snapshot["episode"] = ep
            eval_history.append(snapshot)

            task_summary = " | ".join(
                f"{tid}:{res['score']:.3f}"
                for tid, res in snapshot["task_results"].items()
            )
            print(
                "  Eval "
                f"@{ep:4d} | "
                f"avg_reward={snapshot['avg_total_reward']:.3f} | "
                f"avg_score={snapshot['avg_score']:.3f} | "
                f"success={snapshot['success_rate']:.1%} | "
                f"{task_summary}"
            )

    agent.save(model_path)
    print(f"\nTraining complete! Saved -> {model_path}\n")
    return algo, agent, rewards, train_successes, eval_history, model_path


def plot_training(
    rewards: list[float],
    train_successes: list[float],
    eval_history: list[dict[str, Any]],
    algorithm: str,
    window: int = 50,
) -> None:
    reward_smoothed = _moving_average(rewards, window)
    success_smoothed = _moving_average(train_successes, window)
    episodes = list(range(1, len(rewards) + 1))

    plt.figure(figsize=(11, 7))

    plt.subplot(2, 1, 1)
    plt.plot(episodes, rewards, alpha=0.20, color="#127369", label="Train reward (raw)")
    plt.plot(episodes, reward_smoothed, color="#127369", linewidth=2.0, label=f"Train reward ({window}-ep avg)")

    if eval_history:
        eval_episodes_axis = [int(item["episode"]) for item in eval_history]
        eval_rewards_axis = [float(item["avg_total_reward"]) for item in eval_history]
        plt.plot(eval_episodes_axis, eval_rewards_axis, marker="o", color="#B23A48", label="Eval avg reward")

    plt.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    plt.title(f"Sound Limiter - {algorithm.upper()} Learning Curves", fontsize=13)
    plt.ylabel("Episode Reward")
    plt.legend(loc="best")

    plt.subplot(2, 1, 2)
    plt.plot(episodes, success_smoothed, color="#1F5DAB", linewidth=2.0, label=f"Train success ({window}-ep avg)")

    if eval_history:
        eval_episodes_axis = [int(item["episode"]) for item in eval_history]
        eval_success_axis = [float(item["success_rate"]) for item in eval_history]
        eval_score_axis = [float(item["avg_score"]) for item in eval_history]

        plt.plot(eval_episodes_axis, eval_success_axis, marker="o", color="#8D5A97", label="Eval success rate")
        plt.plot(eval_episodes_axis, eval_score_axis, marker="s", color="#E07A5F", label="Eval avg score")

        task_ids = list(eval_history[0].get("task_results", {}).keys())
        for task_id in task_ids:
            task_scores = [
                float(item.get("task_results", {}).get(task_id, {}).get("score", 0.0))
                for item in eval_history
            ]
            plt.plot(
                eval_episodes_axis,
                task_scores,
                linestyle="--",
                linewidth=1.1,
                alpha=0.85,
                label=f"Eval {task_id}",
            )

    plt.ylim(0.0, 1.02)
    plt.xlabel("Episode")
    plt.ylabel("Rate / Score")
    plt.legend(loc="best", ncol=2)

    plt.tight_layout()
    plt.savefig("training_progress.png", dpi=120)
    print("Training chart saved -> training_progress.png")


def save_training_metrics(
    rewards: list[float],
    train_successes: list[float],
    eval_history: list[dict[str, Any]],
    algorithm: str,
    path: str = "training_metrics.json",
) -> None:
    payload = {
        "algorithm": algorithm,
        "episodes": len(rewards),
        "train_rewards": [round(float(x), 6) for x in rewards],
        "train_successes": [round(float(x), 6) for x in train_successes],
        "train_reward_moving_avg_50": [
            round(float(x), 6) for x in _moving_average(rewards, 50)
        ],
        "train_success_moving_avg_50": [
            round(float(x), 6) for x in _moving_average(train_successes, 50)
        ],
        "evaluation": eval_history,
    }
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    print(f"Training metrics saved -> {path}")


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
    train_episodes = int(os.environ.get("TRAIN_EPISODES", "900"))
    render_every = int(os.environ.get("RENDER_EVERY", "100"))
    eval_every = int(os.environ.get("EVAL_EVERY", "100"))
    eval_episodes = int(os.environ.get("EVAL_EPISODES", "6"))
    train_seed = int(os.environ.get("TRAIN_SEED", "42"))

    algo, _trained_agent, rewards, train_successes, eval_history, model_path = train(
        n_episodes=train_episodes,
        render_every=render_every,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        seed=train_seed,
        algorithm=requested_algo,
    )

    plot_training(
        rewards=rewards,
        train_successes=train_successes,
        eval_history=eval_history,
        algorithm=algo,
    )
    save_training_metrics(
        rewards=rewards,
        train_successes=train_successes,
        eval_history=eval_history,
        algorithm=algo,
    )

    print(f"Evaluating trained {algo.upper()} baseline on registered tasks...\n")
    results = evaluate_saved_model(algo=algo, model_path=model_path)

    print("\n-- Baseline Scores -------------------------------------")
    for _tid, result in results.items():
        print(f"  {result.difficulty:6s} | {result.task_name:30s} | score = {result.score:.4f}")
    print("--------------------------------------------------------\n")
