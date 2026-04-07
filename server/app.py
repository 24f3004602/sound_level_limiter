"""
server/app.py
FastAPI server that exposes OpenEnv-compatible endpoints.

Run locally:
  python server.py
  or
  uvicorn server:app --host 0.0.0.0 --port 7860
"""

import json
import logging
import os
import threading
import time
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import openenv
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from environment.sound_env import ACTION_MAP, SoundLimiterEnv, SoundObservation
    from environment.tasks import ALL_TASKS, TASKS_BY_ID, grade_task
except ModuleNotFoundError as exc:
    if exc.name != "environment":
        raise
    # Allow direct execution: python server/app.py
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from environment.sound_env import ACTION_MAP, SoundLimiterEnv, SoundObservation
    from environment.tasks import ALL_TASKS, TASKS_BY_ID, grade_task


load_dotenv(PROJECT_ROOT / ".env", override=False)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_client_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _resolve_session_id(request: Request) -> str:
    session_id = request.headers.get("x-session-id", "").strip()
    if session_id:
        return session_id[:128]
    return f"client:{_resolve_client_key(request)}"


class EnvironmentStore:
    """Thread-safe in-memory environment store keyed by session id."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._env_by_session: dict[str, SoundLimiterEnv] = {}

    def _build_env(self, req: "ResetRequest") -> SoundLimiterEnv:
        if req.task_id:
            task = TASKS_BY_ID.get(req.task_id)
            if not task:
                raise HTTPException(status_code=404, detail=f"Task '{req.task_id}' not found")
            return SoundLimiterEnv(
                initial_sound=task.initial_sound,
                noise_std=task.noise_std,
                max_steps=task.max_steps,
                seed=req.seed,
            )
        return SoundLimiterEnv(seed=req.seed)

    def reset(self, session_id: str, req: "ResetRequest") -> tuple[SoundObservation, dict]:
        env = self._build_env(req)
        obs = env.reset(seed=req.seed)
        with self._lock:
            self._env_by_session[session_id] = env
        return obs, env.state()

    def step(self, session_id: str, action_id: int):
        with self._lock:
            env = self._env_by_session.get(session_id)
            if env is None:
                env = SoundLimiterEnv()
                self._env_by_session[session_id] = env
            obs, reward, done, info = env.step(action_id)
            state = env.state()
        return obs, reward, done, info, state

    def state(self, session_id: str) -> dict:
        with self._lock:
            env = self._env_by_session.get(session_id)
            if env is None:
                env = SoundLimiterEnv()
                self._env_by_session[session_id] = env
            return env.state()

    def active_sessions(self) -> int:
        with self._lock:
            return len(self._env_by_session)


class RateLimiter:
    """Simple fixed-window rate limiter keyed by client id."""

    def __init__(self, limit_per_minute: int) -> None:
        self.limit_per_minute = max(0, limit_per_minute)
        self.window_seconds = 60
        self._lock = threading.RLock()
        self._history: dict[str, deque[float]] = defaultdict(deque)

    def check(self, client_key: str) -> tuple[bool, int]:
        if self.limit_per_minute == 0:
            return True, 0

        now = time.time()
        cutoff = now - self.window_seconds
        with self._lock:
            events = self._history[client_key]
            while events and events[0] < cutoff:
                events.popleft()

            if len(events) >= self.limit_per_minute:
                retry_after = max(1, int(self.window_seconds - (now - events[0])))
                return False, retry_after

            events.append(now)
        return True, 0


class ApiMetrics:
    """In-memory API call counters and latency aggregates."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stats = defaultdict(lambda: {"count": 0, "errors": 0, "latency_ms_total": 0.0})

    def record(self, method: str, path: str, status_code: int, latency_ms: float) -> None:
        key = f"{method} {path}"
        with self._lock:
            item = self._stats[key]
            item["count"] += 1
            if status_code >= 400:
                item["errors"] += 1
            item["latency_ms_total"] += latency_ms

    def snapshot(self) -> dict:
        with self._lock:
            out = {}
            for key, item in self._stats.items():
                count = item["count"]
                avg_latency_ms = item["latency_ms_total"] / count if count else 0.0
                out[key] = {
                    "count": count,
                    "errors": item["errors"],
                    "avg_latency_ms": round(avg_latency_ms, 3),
                }
            return out


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger("sound_limiter_api")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    return logger


def _log_json(logger: logging.Logger, **payload) -> None:
    logger.info(json.dumps(payload, separators=(",", ":"), default=str))


cors_origins = _split_csv(
    os.environ.get("CORS_ALLOW_ORIGINS", "http://localhost,http://127.0.0.1")
)
cors_methods = _split_csv(os.environ.get("CORS_ALLOW_METHODS", "GET,POST,OPTIONS"))
cors_headers = _split_csv(
    os.environ.get(
        "CORS_ALLOW_HEADERS",
        "Authorization,Content-Type,X-API-Key,X-Session-Id",
    )
)

if not cors_origins:
    cors_origins = ["http://localhost", "http://127.0.0.1"]
if not cors_methods:
    cors_methods = ["GET", "POST", "OPTIONS"]
if not cors_headers:
    cors_headers = ["Authorization", "Content-Type", "X-API-Key", "X-Session-Id"]

api_auth_token = os.environ.get("API_AUTH_TOKEN", "").strip()
rate_limit_per_minute = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "0"))
logger = _configure_logger()


app = FastAPI(
    title="Sound Limiter RL Environment",
    description=(
        "An OpenEnv-compatible environment for training agents to manage "
        "meeting room sound levels. The agent keeps dB within 40-70 dB "
        "using actions: do_nothing, warn, reduce_gain, mute."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=cors_methods,
    allow_headers=cors_headers,
)

app.state.env_store = EnvironmentStore()
app.state.rate_limiter = RateLimiter(rate_limit_per_minute)
app.state.api_metrics = ApiMetrics()


@app.middleware("http")
async def security_and_metrics_middleware(request: Request, call_next):
    if api_auth_token:
        provided_key = request.headers.get("x-api-key", "")
        if provided_key != api_auth_token:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    client_key = _resolve_client_key(request)
    allowed, retry_after = app.state.rate_limiter.check(client_key)
    if not allowed:
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(retry_after)},
            content={"detail": "Rate limit exceeded"},
        )

    started = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - started) * 1000.0

    app.state.api_metrics.record(request.method, request.url.path, response.status_code, latency_ms)
    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
    if rate_limit_per_minute > 0:
        response.headers["X-RateLimit-Limit"] = str(rate_limit_per_minute)

    _log_json(
        logger,
        event="api_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=round(latency_ms, 2),
        session_id=_resolve_session_id(request),
    )

    return response


class StepRequest(BaseModel):
    action: int


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    task_id: Optional[str] = None


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    terminated: bool
    truncated: bool
    info: dict
    state: dict


@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "environment": "sound-limiter-env",
        "version": "1.0.0",
        "openenv_version": getattr(openenv, "__version__", "unknown"),
        "active_sessions": app.state.env_store.active_sessions(),
    }


@app.post("/reset")
def reset(request: Request, req: Optional[ResetRequest] = None):
    req = req or ResetRequest()
    session_id = _resolve_session_id(request)
    obs, state = app.state.env_store.reset(session_id, req)
    return {
        "observation": obs.model_dump(),
        "state": state,
    }


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest, request: Request):
    if req.action not in ACTION_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action {req.action}. Must be 0-3: {ACTION_MAP}",
        )

    session_id = _resolve_session_id(request)
    obs, reward, done, info, state = app.state.env_store.step(session_id, req.action)

    return StepResponse(
        observation=obs.model_dump(),
        reward=reward.value,
        done=done,
        terminated=info.get("terminated", False),
        truncated=info.get("truncated", False),
        info={**info, "reward_reason": reward.reason, "in_safe_zone": reward.in_safe_zone},
        state=state,
    )


@app.get("/state")
def state(request: Request):
    session_id = _resolve_session_id(request)
    return app.state.env_store.state(session_id)


@app.get("/metrics")
def metrics():
    return {
        "rate_limit_per_minute": rate_limit_per_minute,
        "metrics": app.state.api_metrics.snapshot(),
    }


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [t.model_dump() for t in ALL_TASKS],
        "action_space": {
            "type": "Discrete",
            "n": 4,
            "actions": ACTION_MAP,
        },
    }


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    task = TASKS_BY_ID.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return task.model_dump()


@app.post("/tasks/{task_id}/grade")
def grade_task_endpoint(task_id: str, n_episodes: int = 5):
    task = TASKS_BY_ID.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    import random

    def random_agent(_obs_dict: dict) -> int:
        return random.randint(0, 3)

    result = grade_task(task, agent_fn=random_agent, n_episodes=n_episodes)
    return result.model_dump()


@app.get("/actions")
def list_actions():
    return {
        "action_space": {
            "type": "Discrete",
            "n": 4,
        },
        "actions": [
            {"id": 0, "name": "do_nothing", "effect": "No intervention; room drifts louder"},
            {"id": 1, "name": "warn", "effect": "Gentle -3 dB nudge"},
            {"id": 2, "name": "reduce_gain", "effect": "Reduces gain by 0.1; drops sound by ~5 dB"},
            {"id": 3, "name": "mute", "effect": "Hard mute; gain -> 0, sound drops to ~10%"},
        ],
        "observation_space": {
            "type": "Box",
            "fields": {
                "sound_level": "float [0, 100] - current dB level",
                "gain": "float [0, 1] - current gain setting",
                "step_count": "int - steps in current episode",
                "above_safe": "bool - True if > 70 dB",
                "below_safe": "bool - True if < 40 dB",
                "loud_streak": "int - consecutive steps above 95 dB",
            },
        },
    }


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
