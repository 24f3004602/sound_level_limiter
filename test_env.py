"""
test_env.py
Quick sanity check script to run before submitting.
Tests: environment, tasks, graders, and log format.

Run: python test_env.py
"""

import json
import sys


def test_environment() -> None:
    print("\n-- Test 1: Environment reset and step --")
    from environment.sound_env import SoundLimiterEnv, SoundObservation, SoundReward

    env = SoundLimiterEnv(initial_sound=75.0, noise_std=2.0, seed=42)
    obs = env.reset(seed=42)

    assert isinstance(obs, SoundObservation), "reset() must return SoundObservation"
    assert 0 <= obs.sound_level <= 100, "sound_level out of range"
    assert 0 <= obs.gain <= 1, "gain out of range"
    print(f"  reset() -> sound_level={obs.sound_level}, gain={obs.gain} [PASS]")

    for action in range(4):
        obs2, reward, done, _ = env.step(action)
        assert isinstance(reward, SoundReward), "step() reward must be SoundReward"
        assert reward.value in (-2.0, -0.5, 1.0), f"Unexpected reward: {reward.value}"
        assert isinstance(done, bool)
        print(f"  step(action={action}) -> reward={reward.value:+.1f}, sound={obs2.sound_level:.1f} dB [PASS]")

    state = env.state()
    assert "sound_level" in state, "state() must contain sound_level"
    print(f"  state() -> {state} [PASS]")
    print("  [PASS] Environment\n")


def test_tasks() -> None:
    print("-- Test 2: Tasks and graders --")
    from environment.tasks import ALL_TASKS, grade_all_tasks
    import random

    assert len(ALL_TASKS) >= 3, "Must have at least 3 tasks"
    difficulties = [t.difficulty for t in ALL_TASKS]
    assert "easy" in difficulties, "Missing easy task"
    assert "medium" in difficulties, "Missing medium task"
    assert "hard" in difficulties, "Missing hard task"
    print(f"  Found {len(ALL_TASKS)} tasks: {[t.id for t in ALL_TASKS]} [PASS]")

    def random_agent(_obs_dict: dict) -> int:
        return random.randint(0, 3)

    results = grade_all_tasks(agent_fn=random_agent, n_episodes=3)
    for tid, result in results.items():
        assert 0.0 <= result.score <= 1.0, f"Score out of range for {tid}: {result.score}"
        assert 0.0 <= result.reward <= 1.0, f"Reward out of range for {tid}: {result.reward}"
        print(f"  {tid}: score={result.score:.4f}, reward={result.reward:.4f} [PASS]")

    print("  [PASS] Tasks and graders\n")


def test_inference_format() -> None:
    print("-- Test 3: Inference log format --")
    import subprocess

    test_code = '''
import json

from environment.sound_env import ACTION_MAP, SoundLimiterEnv
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
    "type": "[START]",
    "task_id": task.id,
    "task": task.name,
    "difficulty": task.difficulty,
    "state": env.state(),
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
        "type": "[STEP]",
        "task_id": task.id,
        "step": step_num,
        "action": ACTION_MAP[action],
        "action_id": action,
        "sound_level": round(info["sound_level"], 2),
        "gain": round(info["gain"], 3),
        "reward": round(reward.value, 4),
        "in_safe_zone": reward.in_safe_zone,
        "reward_reason": reward.reason,
    }), flush=True)

score = round(steps_in_safe / step_num, 4) if step_num > 0 else 0.0
print(json.dumps({
    "type": "[END]",
    "task_id": task.id,
    "task": task.name,
    "difficulty": task.difficulty,
    "total_steps": step_num,
    "total_reward": round(total_reward, 4),
    "score": score,
    "reward": score,
    "passed": score >= task.success_threshold,
}), flush=True)
'''

    try:
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        partial_stdout = (exc.stdout or "")[-1200:]
        partial_stderr = (exc.stderr or "")[-1200:]
        raise AssertionError(
            "Inference format subprocess timed out after 45s.\n"
            f"Partial stdout:\n{partial_stdout}\n"
            f"Partial stderr:\n{partial_stderr}"
        )

    if result.returncode != 0:
        raise AssertionError(
            "Inference format subprocess failed.\n"
            f"Return code: {result.returncode}\n"
            f"Stdout:\n{result.stdout[-2000:]}\n"
            f"Stderr:\n{result.stderr[-2000:]}"
        )

    lines = [line for line in result.stdout.strip().split("\n") if line.strip()]
    assert len(lines) >= 3, f"Expected >= 3 log lines, got {len(lines)}"

    first = json.loads(lines[0])
    assert first["type"] == "[START]", f"First line must be [START], got {first['type']}"
    assert "task_id" in first

    last = json.loads(lines[-1])
    assert last["type"] == "[END]", f"Last line must be [END], got {last['type']}"
    assert "score" in last
    assert "reward" in last
    assert 0.0 <= last["score"] <= 1.0

    step_lines = [json.loads(line) for line in lines[1:-1]]
    for step in step_lines:
        assert step["type"] == "[STEP]"
        assert "action" in step
        assert "reward" in step

    print(f"  Log lines: {len(lines)} ({len(step_lines)} steps) [PASS]")
    print(f"  [START] -> task_id={first['task_id']} [PASS]")
    print(f"  [END] -> score={last['score']:.4f} [PASS]")
    print("  [PASS] Inference log format\n")


def test_pydantic_models() -> None:
    print("-- Test 4: Pydantic typed models --")
    from environment.sound_env import SoundObservation, SoundReward

    obs = SoundObservation(
        sound_level=65.0,
        gain=0.8,
        step_count=5,
        above_safe=False,
        below_safe=False,
        loud_streak=0,
    )
    assert obs.model_dump()["sound_level"] == 65.0
    print("  SoundObservation serializes correctly [PASS]")

    reward = SoundReward(value=1.0, reason="In safe zone", in_safe_zone=True)
    assert reward.value == 1.0
    print("  SoundReward serializes correctly [PASS]")
    print("  [PASS] Pydantic models\n")


if __name__ == "__main__":
    print("Sound Limiter Env - Pre-submission Tests")
    print("=" * 52)

    errors: list[str] = []
    for test_fn in [test_pydantic_models, test_environment, test_tasks, test_inference_format]:
        try:
            test_fn()
        except Exception as exc:
            print(f"  [FAIL] {exc}\n")
            errors.append(str(exc))

    print("=" * 52)
    if errors:
        print(f"[FAIL] {len(errors)} test(s) failed:")
        for err in errors:
            print(f"   - {err}")
        sys.exit(1)

    print("[PASS] All tests passed. Ready to submit.\n")