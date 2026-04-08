"""
environment/tasks.py
Task definitions, registry, and graders for easy -> medium -> hard benchmarks.

The registry supports static defaults plus runtime task registration for API-based
parameter sweeps.
"""

import threading
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

try:
    from .sound_env import SoundLimiterEnv
except ImportError:
    from sound_env import SoundLimiterEnv


class TaskConfig(BaseModel):
    """Configuration for a single task."""

    id: str
    name: str
    description: str
    difficulty: str
    initial_sound: float
    noise_std: float
    max_steps: int
    success_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Min fraction of safe steps required to pass",
    )


TASK_EASY = TaskConfig(
    id="task_easy",
    name="Quiet the Room",
    description=(
        "Sound starts at a moderately loud 75 dB. "
        "The room noise is low. Keep sound in the 40-70 dB safe zone "
        "for at least 60% of a 30-step episode. "
        "Basic actions (reduce_gain or warn) should work reliably."
    ),
    difficulty="easy",
    initial_sound=75.0,
    noise_std=1.5,
    max_steps=30,
    success_threshold=0.60,
)

TASK_MEDIUM = TaskConfig(
    id="task_medium",
    name="Manage a Noisy Meeting",
    description=(
        "Sound starts at 85 dB (loud meeting) with moderate noise drift. "
        "The agent must bring sound to the safe zone and hold it there "
        "for at least 50% of a 40-step episode. "
        "Avoid over-muting because very low dB is still undesirable."
    ),
    difficulty="medium",
    initial_sound=85.0,
    noise_std=4.0,
    max_steps=40,
    success_threshold=0.50,
)

TASK_HARD = TaskConfig(
    id="task_hard",
    name="Control a Loud Event",
    description=(
        "Sound starts at 92 dB with heavy noise spikes simulating a loud event. "
        "The agent must keep sound safe for at least 40% of a 50-step episode. "
        "Noise frequently pushes sound back up, so robust feedback control is required."
    ),
    difficulty="hard",
    initial_sound=92.0,
    noise_std=7.0,
    max_steps=50,
    success_threshold=0.40,
)


DEFAULT_TASKS: list[TaskConfig] = [TASK_EASY, TASK_MEDIUM, TASK_HARD]
_TASK_LOCK = threading.RLock()

# Backward-compatible globals; these are kept in sync by registry functions.
ALL_TASKS: list[TaskConfig] = [task.model_copy(deep=True) for task in DEFAULT_TASKS]
TASKS_BY_ID: dict[str, TaskConfig] = {task.id: task for task in ALL_TASKS}


def _normalize_task(task: TaskConfig) -> TaskConfig:
    normalized = task.model_copy(deep=True)
    normalized.id = normalized.id.strip()
    normalized.name = normalized.name.strip()
    normalized.description = normalized.description.strip()
    normalized.difficulty = normalized.difficulty.strip().lower()
    if not normalized.id:
        raise ValueError("Task id cannot be empty")
    if not normalized.name:
        raise ValueError("Task name cannot be empty")
    if normalized.noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")
    if normalized.max_steps <= 0:
        raise ValueError("max_steps must be >= 1")
    return normalized


def list_tasks() -> list[TaskConfig]:
    """Return all currently registered tasks as copies."""
    with _TASK_LOCK:
        return [task.model_copy(deep=True) for task in ALL_TASKS]


def get_task(task_id: str) -> Optional[TaskConfig]:
    """Fetch one task by id."""
    with _TASK_LOCK:
        task = TASKS_BY_ID.get(task_id)
        return task.model_copy(deep=True) if task else None


def register_task(task: TaskConfig, overwrite: bool = False) -> TaskConfig:
    """
    Register or update a task in the runtime registry.

    Args:
        task: TaskConfig to register.
        overwrite: If False, duplicate ids are rejected.
    """
    normalized = _normalize_task(task)

    with _TASK_LOCK:
        existing = TASKS_BY_ID.get(normalized.id)
        if existing and not overwrite:
            raise ValueError(f"Task '{normalized.id}' already exists")

        if existing:
            for idx, row in enumerate(ALL_TASKS):
                if row.id == normalized.id:
                    ALL_TASKS[idx] = normalized
                    break
        else:
            ALL_TASKS.append(normalized)

        TASKS_BY_ID[normalized.id] = normalized
        return normalized.model_copy(deep=True)


class GradeResult(BaseModel):
    task_id: str
    task_name: str
    difficulty: str
    score: float = Field(..., ge=0.0, le=1.0)
    reward: float = Field(..., ge=0.0, le=1.0)
    episodes_run: int
    avg_safe_fraction: float
    avg_total_reward: float
    passed: bool
    details: str


def grade_task(
    task: TaskConfig,
    agent_fn,
    n_episodes: int = 5,
    seed: int = 42,
) -> GradeResult:
    """
    Grade an agent on a task.

    Args:
        task: TaskConfig to run.
        agent_fn: Callable(obs_dict) -> action_id in [0, 3].
        n_episodes: Number of episodes to average.
        seed: Fixed seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    episode_scores = []
    episode_rewards = []

    for _ in range(n_episodes):
        ep_seed = int(rng.integers(0, 10_000))
        env = SoundLimiterEnv(
            initial_sound=task.initial_sound,
            noise_std=task.noise_std,
            max_steps=task.max_steps,
            seed=ep_seed,
        )

        obs = env.reset(seed=ep_seed)
        done = False
        steps_in_safe = 0
        total_steps = 0
        total_reward = 0.0

        while not done:
            action_id = int(agent_fn(obs.model_dump()))
            obs, reward, done, _ = env.step(action_id)

            total_reward += reward.value
            total_steps += 1
            if reward.in_safe_zone:
                steps_in_safe += 1

        safe_fraction = steps_in_safe / total_steps if total_steps > 0 else 0.0
        episode_scores.append(safe_fraction)
        episode_rewards.append(total_reward)

    avg_safe = float(np.mean(episode_scores))
    avg_rew = float(np.mean(episode_rewards))

    # Primary score is safe-zone occupancy.
    base_score = avg_safe

    # Consistency bonus scales with margin above threshold (up to +0.20).
    margin = max(0.0, avg_safe - task.success_threshold)
    consistency_bonus = min(0.20, 0.50 * margin)

    final_score = float(np.clip(base_score + consistency_bonus, 0.0, 1.0))
    passed = avg_safe >= task.success_threshold

    details = (
        f"Average safe-zone fraction: {avg_safe:.1%} (threshold: {task.success_threshold:.0%}). "
        f"Consistency bonus: +{consistency_bonus:.3f}. "
        f"Average total reward: {avg_rew:.2f}. "
        f"{'PASSED' if passed else 'FAILED'}"
    )

    return GradeResult(
        task_id=task.id,
        task_name=task.name,
        difficulty=task.difficulty,
        score=round(final_score, 4),
        reward=round(final_score, 4),
        episodes_run=n_episodes,
        avg_safe_fraction=round(avg_safe, 4),
        avg_total_reward=round(avg_rew, 4),
        passed=passed,
        details=details,
    )


def grade_all_tasks(agent_fn, n_episodes: int = 5) -> dict[str, GradeResult]:
    """Grade an agent on all currently registered tasks."""
    results: dict[str, GradeResult] = {}
    print("\nGrading all tasks...\n")
    for task in list_tasks():
        result = grade_task(task, agent_fn, n_episodes=n_episodes)
        results[task.id] = result
        status = "pass" if result.passed else "fail"
        print(
            f"  {status} [{result.difficulty.upper():6s}] {result.task_name:30s} "
            f"score={result.score:.4f}  safe={result.avg_safe_fraction:.1%}"
        )
    print()
    return results