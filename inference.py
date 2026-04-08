import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from environment.sound_env import SoundLimiterEnv, ACTION_MAP
from environment.tasks import ALL_TASKS, TaskConfig


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get(
    "HF_TOKEN",
    os.environ.get("OPENAI_API_KEY", os.environ.get("OPEAI_API_KEY", "")),
)
LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "8"))
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "1"))
LLM_FAIL_OPEN_AFTER = int(os.environ.get("LLM_FAIL_OPEN_AFTER", "5"))
HEURISTIC_ONLY = os.environ.get("HEURISTIC_ONLY", "0").strip() in ("1", "true", "True")

if not HF_TOKEN and not HEURISTIC_ONLY:
    print(
        json.dumps({"error": "HF_TOKEN or OPENAI_API_KEY not set. Cannot run inference."}),
        flush=True
    )
    sys.exit(1)


def _configure_metrics_logger() -> logging.Logger:
    logger = logging.getLogger("sound_limiter_inference")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    return logger


def _log_metric(logger: logging.Logger, event: str, **payload) -> None:
    logger.info(json.dumps({"event": event, **payload}, separators=(",", ":"), default=str))


metrics_logger = _configure_metrics_logger()

client = None
if not HEURISTIC_ONLY:
    client = OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
        max_retries=LLM_MAX_RETRIES,
        timeout=LLM_TIMEOUT_S,
    )

_consecutive_llm_errors = 0
_use_heuristic_only = HEURISTIC_ONLY

SYSTEM_PROMPT = """You control room sound.
Keep sound_level in the 40-70 dB safe range.
Reply with one integer only:
0 do_nothing
1 warn
2 reduce_gain
3 mute
Prefer 1 or 2 above 70 dB and avoid 3 unless extreme."""


def llm_choose_action(obs_dict: dict, step_num: int) -> int:
    """Ask the LLM to choose an action given the current observation."""
    global _consecutive_llm_errors, _use_heuristic_only

    if _use_heuristic_only or client is None:
        return _stability_guard(obs_dict, _heuristic_fallback(obs_dict))

    user_message = (
        f"Step {step_num}. Current state:\n"
        f"  sound_level = {obs_dict['sound_level']:.1f} dB\n"
        f"  gain        = {obs_dict['gain']:.3f}\n"
        f"  above_safe  = {obs_dict['above_safe']} (>70 dB)\n"
        f"  below_safe  = {obs_dict['below_safe']} (<40 dB)\n"
        f"  loud_streak = {obs_dict['loud_streak']}\n\n"
        f"Which action? Reply with 0, 1, 2, or 3 only."
    )

    try:
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        raw_content = response.choices[0].message.content or ""
        raw = raw_content.strip()

        if not raw or raw[0] not in "0123":
            action = 2
        else:
            action = int(raw[0])
            if action not in ACTION_MAP:
                action = 2
        _consecutive_llm_errors = 0
        _log_metric(
            metrics_logger,
            "llm_call",
            step=step_num,
            latency_ms=round(latency_ms, 2),
            model=MODEL_NAME,
        )

    except Exception:
        _consecutive_llm_errors += 1
        if _consecutive_llm_errors >= LLM_FAIL_OPEN_AFTER and not _use_heuristic_only:
            _use_heuristic_only = True
            _log_metric(
                metrics_logger,
                "llm_circuit_open",
                consecutive_errors=_consecutive_llm_errors,
                fail_open_after=LLM_FAIL_OPEN_AFTER,
            )
        action = _heuristic_fallback(obs_dict)

    return _stability_guard(obs_dict, action)


def _heuristic_fallback(obs_dict: dict) -> int:
    sl = obs_dict.get("sound_level", 60.0)
    gain = obs_dict.get("gain", 1.0)

    if sl > 90:
        return 2
    if sl > 72:
        return 2 if gain > 0.3 else 1
    if sl > 68:
        return 1
    return 0


def _stability_guard(obs_dict: dict, proposed_action: int) -> int:
    sl = obs_dict.get("sound_level", 60.0)
    gain = obs_dict.get("gain", 1.0)

    if proposed_action == 3:
        proposed_action = 2

    if sl > 90:
        return 2
    if sl > 72:
        return 2 if gain > 0.3 else 1
    if sl > 68:
        return 1
    if sl < 40 and proposed_action in (2, 3):
        return 0

    return proposed_action

def run_task_inference(task: TaskConfig) -> dict:
    task_started = time.perf_counter()
    env = SoundLimiterEnv(
        initial_sound=task.initial_sound,
        noise_std=task.noise_std,
        max_steps=task.max_steps,
        seed=42,
    )

    obs = env.reset(seed=42)

    start_log = {
        "type": "[START]",
        "task_id": task.id,
        "task":       task.name,
        "difficulty": task.difficulty,
        "model": MODEL_NAME,
        "state": env.state(),
    }
    print(json.dumps(start_log), flush=True)

    total_reward = 0.0
    steps_in_safe = 0
    step_num = 0
    done = False

    while not done:
        step_num += 1
        action = llm_choose_action(obs.model_dump(), step_num)
        obs, reward, done, info = env.step(action)

        total_reward += reward.value
        if reward.in_safe_zone:
            steps_in_safe += 1

        step_log = {
            "type": "[STEP]",
            "task_id": task.id,
            "step": step_num,
            "action": ACTION_MAP[action],
            "action_id": action,
            "sound_level": round(info["sound_level"], 2),
            "gain": round(info["gain"], 3),
            "reward":       round(reward.value, 4),
            "in_safe_zone": reward.in_safe_zone,
            "reward_reason": reward.reason,
        }
        print(json.dumps(step_log), flush=True)

    score = round(steps_in_safe / step_num, 4) if step_num > 0 else 0.0

    end_log = {
        "type": "[END]",
        "task_id": task.id,
        "task":         task.name,
        "difficulty":   task.difficulty,
        "total_steps": step_num,
        "total_reward": round(total_reward, 4),
        "score": score,
        "reward": score,
        "passed": score >= task.success_threshold,
    }
    print(json.dumps(end_log), flush=True)

    _log_metric(
        metrics_logger,
        "task_score",
        task_id=task.id,
        difficulty=task.difficulty,
        score=score,
        passed=end_log["passed"],
        total_steps=step_num,
        total_reward=round(total_reward, 4),
        latency_ms=round((time.perf_counter() - task_started) * 1000.0, 2),
        model=MODEL_NAME,
        heuristic_only=_use_heuristic_only,
    )

    return end_log


def run_inference():
    started = time.perf_counter()
    final_results = []

    for task in ALL_TASKS:
        result = run_task_inference(task)
        final_results.append(result)

    avg_score = sum(r["score"] for r in final_results) / len(final_results) if final_results else 0.0
    passed_count = sum(1 for r in final_results if r["passed"])
    _log_metric(
        metrics_logger,
        "inference_summary",
        tasks=len(final_results),
        passed=passed_count,
        avg_score=round(avg_score, 4),
        total_latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
        model=MODEL_NAME,
        heuristic_only=_use_heuristic_only,
    )


if __name__ == "__main__":
    run_inference()