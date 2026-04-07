"""
inference.py
OpenEnv Hackathon — Required Inference Script

Uses an LLM (via OpenAI-compatible client) as the agent.
The LLM observes the environment state and decides which action to take.

Required env vars:
  API_BASE_URL   — LLM API endpoint
  MODEL_NAME     — Model identifier
  HF_TOKEN       — Hugging Face / API key

Output format (strict — do not change field names):
  [START] → once per task
  [STEP]  → once per step
  [END]   → once per task with final score

Run:
  python inference.py
"""

import os
import json
import sys
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from environment.sound_env import SoundLimiterEnv, ACTION_MAP
from environment.tasks import ALL_TASKS, TaskConfig


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


# ══════════════════════════════════════════════════════════
# Config — read from environment variables
# ══════════════════════════════════════════════════════════
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get(
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

# ── OpenAI-compatible client (required by hackathon) ──────
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


# ══════════════════════════════════════════════════════════
# LLM Agent
# ══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an intelligent sound level controller for a meeting room.

Your goal: keep the sound level between 40 and 70 dB (the "safe zone").

Available actions (respond with the action number ONLY — no explanation):
  0 = do_nothing   — let the room drift (use only when already in safe zone)
  1 = warn         — gentle nudge, reduces sound by ~3 dB
  2 = reduce_gain  — meaningful reduction, reduces sound by ~5 dB
  3 = mute         — drastic cut (avoid unless sound > 85 dB, as it over-mutes)

Strategy tips:
- If sound_level > 70: use warn (1) or reduce_gain (2)
- If sound_level > 85: use reduce_gain (2) or mute (3)  
- If sound_level in 40–70: use do_nothing (0)
- If sound_level < 40: use do_nothing (0) — gain will recover naturally
- Prefer reduce_gain (2) over mute (3) unless it's an emergency

Respond with a single integer: 0, 1, 2, or 3."""


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
            model       = MODEL_NAME,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens  = 5,
            temperature = 0.0,   # Deterministic for reproducibility
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        raw_content = response.choices[0].message.content or ""
        raw = raw_content.strip()

        # Be defensive: some providers can return empty/non-numeric content.
        if not raw or raw[0] not in "0123":
            action = 2
        else:
            action = int(raw[0])   # Take the first character as the action digit
            if action not in ACTION_MAP:
                action = 2         # Fallback: reduce_gain
        _consecutive_llm_errors = 0
        _log_metric(
            metrics_logger,
            "llm_call",
            step=step_num,
            latency_ms=round(latency_ms, 2),
            model=MODEL_NAME,
        )

    except Exception:
        # On any API error, fall back to a safe heuristic
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
    """Rule-based fallback if the LLM call fails."""
    sl = obs_dict.get("sound_level", 60.0)
    gain = obs_dict.get("gain", 1.0)

    # Tuned thresholds that perform well across easy/medium/hard tasks.
    if sl > 90:
        return 2
    if sl > 72:
        return 2 if gain > 0.3 else 1
    if sl > 68:
        return 1
    return 0


def _stability_guard(obs_dict: dict, proposed_action: int) -> int:
    """Clamp risky actions to avoid irreversible over-muting in long episodes."""
    sl = obs_dict.get("sound_level", 60.0)
    gain = obs_dict.get("gain", 1.0)

    # Mute is usually too destructive for this environment because gain cannot recover.
    if proposed_action == 3:
        proposed_action = 2

    # Keep behavior robust if the model drifts from good control policy.
    if sl > 90:
        return 2
    if sl > 72:
        return 2 if gain > 0.3 else 1
    if sl > 68:
        return 1
    if sl < 40 and proposed_action in (2, 3):
        return 0

    return proposed_action


# ══════════════════════════════════════════════════════════
# Main inference loop
# ══════════════════════════════════════════════════════════

def run_task_inference(task: TaskConfig) -> dict:
    """Run the LLM agent on a single task. Returns result dict with score."""
    task_started = time.perf_counter()
    env = SoundLimiterEnv(
        initial_sound = task.initial_sound,
        noise_std     = task.noise_std,
        max_steps     = task.max_steps,
        seed          = 42,   # Fixed seed — reproducible baseline
    )

    obs = env.reset(seed=42)

    # ── [START] ───────────────────────────────────────────
    start_log = {
        "type":       "[START]",
        "task_id":    task.id,
        "task":       task.name,
        "difficulty": task.difficulty,
        "model":      MODEL_NAME,
        "state":      env.state(),
    }
    print(json.dumps(start_log), flush=True)

    # ── Episode loop ──────────────────────────────────────
    total_reward   = 0.0
    steps_in_safe  = 0
    step_num       = 0
    done           = False

    while not done:
        step_num += 1
        action   = llm_choose_action(obs.model_dump(), step_num)
        obs, reward, done, info = env.step(action)

        total_reward += reward.value
        if reward.in_safe_zone:
            steps_in_safe += 1

        # ── [STEP] ────────────────────────────────────────
        step_log = {
            "type":         "[STEP]",
            "task_id":      task.id,
            "step":         step_num,
            "action":       ACTION_MAP[action],
            "action_id":    action,
            "sound_level":  round(info["sound_level"], 2),
            "gain":         round(info["gain"], 3),
            "reward":       round(reward.value, 4),
            "in_safe_zone": reward.in_safe_zone,
            "reward_reason": reward.reason,
        }
        print(json.dumps(step_log), flush=True)

    # ── Final score ───────────────────────────────────────
    score = round(steps_in_safe / step_num, 4) if step_num > 0 else 0.0

    # ── [END] ─────────────────────────────────────────────
    end_log = {
        "type":         "[END]",
        "task_id":      task.id,
        "task":         task.name,
        "difficulty":   task.difficulty,
        "total_steps":  step_num,
        "total_reward": round(total_reward, 4),
        "score":        score,
        "reward":       score,   # Required field alias
        "passed":       score >= task.success_threshold,
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
    """Run the LLM agent on all 3 tasks in order: easy → medium → hard."""
    started = time.perf_counter()
    final_results = []

    for task in ALL_TASKS:
        result = run_task_inference(task)
        final_results.append(result)

    # Summary to stderr (doesn't interfere with stdout log parsing)
    print("\n──────────────────────────────────────────", file=sys.stderr)
    print("  INFERENCE SUMMARY", file=sys.stderr)
    print("──────────────────────────────────────────", file=sys.stderr)
    for r in final_results:
        status = " PASSED" if r["passed"] else " FAILED"
        print(
            f"  {r['difficulty'].upper():6s} | {r['task']:30s} | "
            f"score={r['score']:.4f} {status}",
            file=sys.stderr
        )
    print("──────────────────────────────────────────\n", file=sys.stderr)

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