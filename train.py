"""
train.py
Train the Q-Learning baseline agent on the Sound Limiter environment.

Run:  python train.py
This creates q_table.pkl used by the Q-agent baseline.
The LLM agent in inference.py does NOT need this file.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (works in Docker)
import matplotlib.pyplot as plt

from environment.sound_env import SoundLimiterEnv
from environment.tasks import TASK_EASY, TASK_MEDIUM, TASK_HARD, grade_all_tasks
from agent.q_agent import QLearningAgent


def train(
    n_episodes: int = 900,
    render_every: int = 100,
    seed: int = 42,
) -> tuple[QLearningAgent, list[float]]:
    # Ensure reproducible Q-learning exploration and training trajectories.
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    task_curriculum = [TASK_EASY, TASK_MEDIUM, TASK_HARD]
    task_weights = [0.25, 0.35, 0.40]

    agent = QLearningAgent(
        learning_rate=0.14,
        discount=0.96,
        epsilon=1.0,
        epsilon_decay=0.996,
        epsilon_min=0.03,
    )

    episode_rewards: list[float] = []
    print("Training Q-Learning agent...\n")

    for ep in range(1, n_episodes + 1):
        task_idx = int(rng.choice(len(task_curriculum), p=task_weights))
        task = task_curriculum[task_idx]

        # Mild domain randomization improves cross-task robustness.
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
        episode_rewards.append(total_reward)

        if ep % render_every == 0:
            avg = float(np.mean(episode_rewards[-render_every:]))
            print(
                f"  Episode {ep:4d}/{n_episodes} | "
                f"Avg Reward: {avg:7.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    agent.save("q_table.pkl")
    print("\nTraining complete! Saved -> q_table.pkl\n")
    return agent, episode_rewards


def plot_training(rewards: list[float], window: int = 50) -> None:
    smoothed = [
        float(np.mean(rewards[max(0, i - window): i + 1]))
        for i in range(len(rewards))
    ]

    plt.figure(figsize=(10, 4))
    plt.plot(rewards,  alpha=0.25, color="#4A90D9", label="Raw reward")
    plt.plot(smoothed, color="#4A90D9", linewidth=2.0, label=f"{window}-ep moving avg")
    plt.axhline(y=0,  color="gray",  linestyle="--", linewidth=0.8)
    plt.axhline(y=30, color="green", linestyle="--", linewidth=0.8, label="Good threshold")
    plt.title("Sound Limiter - Q-Learning Training Progress", fontsize=13)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_progress.png", dpi=120)
    print("Training chart saved -> training_progress.png")


if __name__ == "__main__":
    # ── Train ──────────────────────────────────────────────
    trained_agent, rewards = train(n_episodes=900)

    # ── Plot ───────────────────────────────────────────────
    plot_training(rewards)

    # ── Evaluate on all 3 tasks ────────────────────────────
    print("Evaluating trained agent on all 3 tasks...\n")

    # Load with no exploration for evaluation
    eval_agent = QLearningAgent(epsilon=0.0)
    loaded_ok = eval_agent.load("q_table.pkl")
    if not loaded_ok:
        raise RuntimeError("Could not load q_table.pkl for evaluation")
    eval_agent.epsilon = 0.0

    results = grade_all_tasks(
        agent_fn   = lambda obs_dict: eval_agent.choose_action(obs_dict),
        n_episodes = 10,
    )

    print("\n── Baseline Scores ─────────────────────────────────")
    for tid, r in results.items():
        print(f"  {r.difficulty:6s} | {r.task_name:30s} | score = {r.score:.4f}")
    print("────────────────────────────────────────────────────\n")