"""
environment/tasks.py
3 Tasks with Graders — Easy → Medium → Hard

Each task defines:
- A concrete objective the agent must accomplish
- A grader that scores performance 0.0–1.0
- Deterministic, reproducible scoring criteria
"""

import numpy as np
from typing import Optional
from pydantic import BaseModel, Field
from sound_env import SoundLimiterEnv, ACTION_MAP


# ══════════════════════════════════════════════════════════
# Task definitions
# ══════════════════════════════════════════════════════════

class TaskConfig(BaseModel):
    """Configuration for a single task."""
    id:             str
    name:           str
    description:    str
    difficulty:     str   # "easy", "medium", "hard"
    initial_sound:  float
    noise_std:      float
    max_steps:      int
    success_threshold: float = Field(default=0.6, description="Min fraction of safe steps to 'pass'")


# ── Task 1: Easy ───────────────────────────────────────────
TASK_EASY = TaskConfig(
    id            = "task_easy",
    name          = "Quiet the Room",
    description   = (
        "Sound starts at a moderately loud 75 dB. "
        "The room noise is low. Keep sound in the 40–70 dB safe zone "
        "for at least 60% of a 30-step episode. "
        "Basic actions (reduce_gain or warn) should work reliably."
    ),
    difficulty    = "easy",
    initial_sound = 75.0,
    noise_std     = 1.5,
    max_steps     = 30,
    success_threshold = 0.60,
)

# ── Task 2: Medium ─────────────────────────────────────────
TASK_MEDIUM = TaskConfig(
    id            = "task_medium",
    name          = "Manage a Noisy Meeting",
    description   = (
        "Sound starts at 85 dB (loud meeting) with moderate noise drift. "
        "The agent must bring sound to the safe zone and hold it there "
        "for at least 50% of a 40-step episode. "
        "Avoid over-muting — muting below 20 dB is penalised."
    ),
    difficulty    = "medium",
    initial_sound = 85.0,
    noise_std     = 4.0,
    max_steps     = 40,
    success_threshold = 0.50,
)

# ── Task 3: Hard ───────────────────────────────────────────
TASK_HARD = TaskConfig(
    id            = "task_hard",
    name          = "Control a Loud Event",
    description   = (
        "Sound starts at 92 dB with heavy noise spikes simulating a loud event. "
        "The agent must keep sound safe for at least 40% of a 50-step episode. "
        "Hard: noise frequently pushes sound back up; muting is too blunt "
        "(penalised), so the agent must use graduated responses."
    ),
    difficulty    = "hard",
    initial_sound = 92.0,
    noise_std     = 7.0,
    max_steps     = 50,
    success_threshold = 0.40,
)

ALL_TASKS: list[TaskConfig] = [TASK_EASY, TASK_MEDIUM, TASK_HARD]
TASKS_BY_ID: dict[str, TaskConfig] = {t.id: t for t in ALL_TASKS}


# ══════════════════════════════════════════════════════════
# Grader
# ══════════════════════════════════════════════════════════

class GradeResult(BaseModel):
    task_id:           str
    task_name:         str
    difficulty:        str
    score:             float = Field(..., ge=0.0, le=1.0)
    reward:            float = Field(..., ge=0.0, le=1.0)   # Alias for score (required by spec)
    episodes_run:      int
    avg_safe_fraction: float
    avg_total_reward:  float
    passed:            bool
    details:           str


def grade_task(
    task: TaskConfig,
    agent_fn,           # Callable: (obs_dict) -> action_id (int)
    n_episodes: int = 5,
    seed: int = 42,
) -> GradeResult:
    """
    Grade an agent on a task.

    Args:
        task:       TaskConfig — which task to run
        agent_fn:   function that takes an observation dict and returns action_id (0–3)
        n_episodes: how many episodes to average over
        seed:       fixed seed for reproducibility

    Returns:
        GradeResult with score in 0.0–1.0
    """
    rng = np.random.default_rng(seed)
    episode_scores   = []
    episode_rewards  = []

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 10_000))
        env = SoundLimiterEnv(
            initial_sound = task.initial_sound,
            noise_std     = task.noise_std,
            max_steps     = task.max_steps,
            seed          = ep_seed,
        )

        obs  = env.reset(seed=ep_seed)
        done = False

        steps_in_safe = 0
        total_steps   = 0
        total_reward  = 0.0

        while not done:
            action_id = agent_fn(obs.model_dump())
            obs, reward, done, info = env.step(action_id)

            total_reward  += reward.value
            total_steps   += 1
            if reward.in_safe_zone:
                steps_in_safe += 1

        safe_fraction = steps_in_safe / total_steps if total_steps > 0 else 0.0
        episode_scores.append(safe_fraction)
        episode_rewards.append(total_reward)

    avg_safe  = float(np.mean(episode_scores))
    avg_rew   = float(np.mean(episode_rewards))

    # ── Score formula ─────────────────────────────────────
    # Primary: fraction of steps in the safe zone (0.0–1.0)
    # Bonus:   if avg_safe > success_threshold, small bonus for consistency
    base_score    = avg_safe
    bonus         = 0.05 if avg_safe >= task.success_threshold else 0.0
    final_score   = float(np.clip(base_score + bonus, 0.0, 1.0))
    passed        = avg_safe >= task.success_threshold

    details = (
        f"Average safe-zone fraction: {avg_safe:.1%} "
        f"(threshold: {task.success_threshold:.0%}). "
        f"Average total reward: {avg_rew:.2f}. "
        f"{'PASSED ' if passed else 'FAILED '}"
    )

    return GradeResult(
        task_id           = task.id,
        task_name         = task.name,
        difficulty        = task.difficulty,
        score             = round(final_score, 4),
        reward            = round(final_score, 4),
        episodes_run      = n_episodes,
        avg_safe_fraction = round(avg_safe, 4),
        avg_total_reward  = round(avg_rew, 4),
        passed            = passed,
        details           = details,
    )


def grade_all_tasks(agent_fn, n_episodes: int = 5) -> dict[str, GradeResult]:
    """Grade an agent on all 3 tasks. Returns dict keyed by task_id."""
    results = {}
    print("\n Grading all tasks...\n")
    for task in ALL_TASKS:
        result = grade_task(task, agent_fn, n_episodes=n_episodes)
        results[task.id] = result
        status = "pass" if result.passed else "fail"
        print(
            f"  {status} [{result.difficulty.upper():6s}] {result.task_name:30s} "
            f"score={result.score:.4f}  safe={result.avg_safe_fraction:.1%}"
        )
    print()
    return results