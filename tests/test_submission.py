import json
import random
import subprocess
import sys

from environment.sound_env import SoundLimiterEnv, SoundObservation, SoundReward
from environment.tasks import ALL_TASKS, TaskConfig, get_task, grade_all_tasks, register_task


def test_pydantic_models() -> None:
    obs = SoundObservation(
        sound_level=65.0,
        gain=0.8,
        step_count=5,
        above_safe=False,
        below_safe=False,
        loud_streak=0,
    )
    assert obs.model_dump()["sound_level"] == 65.0

    reward = SoundReward(value=1.0, reason="In safe zone", in_safe_zone=True)
    assert reward.value == 1.0


def test_environment_reset_and_step() -> None:
    env = SoundLimiterEnv(initial_sound=75.0, noise_std=2.0, seed=42)
    obs = env.reset(seed=42)

    assert isinstance(obs, SoundObservation)
    assert 0 <= obs.sound_level <= 100
    assert 0 <= obs.gain <= 1

    for action in range(4):
        obs2, reward, done, _ = env.step(action)
        assert isinstance(obs2, SoundObservation)
        assert isinstance(reward, SoundReward)
        assert -2.0 <= reward.value <= 1.25
        assert len(obs2.source_levels) == 3
        assert isinstance(done, bool)


def test_reward_shaping_has_gradient() -> None:
    env = SoundLimiterEnv(seed=0)
    env.safe_streak = 0

    env.sound_level = 71.0
    near_safe = env._compute_reward().value

    env.sound_level = 84.0
    far_from_safe = env._compute_reward().value

    assert near_safe > far_from_safe


def test_dynamic_task_registry_adds_task() -> None:
    task = TaskConfig(
        id="task_custom_registry_test",
        name="Custom Registry Test",
        description="Registry insertion test task",
        difficulty="custom",
        initial_sound=78.0,
        noise_std=2.0,
        max_steps=25,
        success_threshold=0.55,
    )

    registered = register_task(task, overwrite=True)
    assert registered.id == task.id

    loaded = get_task(task.id)
    assert loaded is not None
    assert loaded.initial_sound == 78.0


def test_tasks_and_grader_bounds() -> None:
    assert len(ALL_TASKS) >= 3

    difficulties = {task.difficulty for task in ALL_TASKS}
    assert "easy" in difficulties
    assert "medium" in difficulties
    assert "hard" in difficulties

    rng = random.Random(42)

    def random_agent(_obs_dict: dict) -> int:
        return rng.randint(0, 3)

    results = grade_all_tasks(agent_fn=random_agent, n_episodes=3)
    for result in results.values():
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.reward <= 1.0


def test_inference_log_format() -> None:
    test_code = """
import json
from environment.sound_env import SoundLimiterEnv, ACTION_MAP
from environment.tasks import ALL_TASKS

task = ALL_TASKS[0]
env = SoundLimiterEnv(
    initial_sound=task.initial_sound,
    noise_std=task.noise_std,
    max_steps=task.max_steps,
    seed=42,
)
obs = env.reset(seed=42)

print(json.dumps({
    'type': '[START]',
    'task_id': task.id,
    'task': task.name,
    'difficulty': task.difficulty,
    'state': env.state(),
}), flush=True)

done = False
step_num = 0
total_reward = 0.0
steps_in_safe = 0

while not done:
    step_num += 1
    action = 2 if obs.sound_level > 70 else 0
    obs, reward, done, info = env.step(action)
    total_reward += reward.value
    if reward.in_safe_zone:
        steps_in_safe += 1

    print(json.dumps({
        'type': '[STEP]',
        'task_id': task.id,
        'step': step_num,
        'action': ACTION_MAP[action],
        'action_id': action,
        'sound_level': round(info['sound_level'], 2),
        'gain': round(info['gain'], 3),
        'reward': round(reward.value, 4),
        'in_safe_zone': reward.in_safe_zone,
        'reward_reason': reward.reason,
    }), flush=True)

score = round(steps_in_safe / step_num, 4) if step_num > 0 else 0.0
print(json.dumps({
    'type': '[END]',
    'task_id': task.id,
    'task': task.name,
    'difficulty': task.difficulty,
    'total_steps': step_num,
    'total_reward': round(total_reward, 4),
    'score': score,
    'reward': score,
    'passed': score >= task.success_threshold,
}), flush=True)
"""

    result = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )

    assert result.returncode == 0, (
        "Inference format subprocess failed.\n"
        f"Return code: {result.returncode}\n"
        f"Stdout:\n{result.stdout[-2000:]}\n"
        f"Stderr:\n{result.stderr[-2000:]}"
    )

    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    assert len(lines) >= 3

    first = json.loads(lines[0])
    last = json.loads(lines[-1])
    assert first["type"] == "[START]"
    assert last["type"] == "[END]"
    assert 0.0 <= float(last["score"]) <= 1.0
    assert 0.0 <= float(last["reward"]) <= 1.0

    for step_line in lines[1:-1]:
        step = json.loads(step_line)
        assert step["type"] == "[STEP]"
        assert "action" in step
        assert "reward" in step
