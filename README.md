---
title: Sound Level Limiter
emoji: "🎯"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Sound Limiter RL Environment

An OpenEnv-compatible reinforcement learning environment where an agent manages meeting-room sound levels and keeps dB in a safe range.

## Project layout

- `server/app.py`: FastAPI server with OpenEnv endpoints.
- `server.py`: compatibility launcher for local runs.
- `environment/sound_env.py`: core environment dynamics.
- `environment/tasks.py`: task definitions and graders.
- `train.py`: baseline training (PyTorch DQN by default, Q-learning fallback).
- `inference.py`: submission-time inference script.
- `scripts/validate-submission.sh`: Bash validator.
- `scripts/validate-submission.ps1`: PowerShell validator.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```dotenv
HF_TOKEN=your_hf_or_openai_token
OPENAI_API_KEY=your_openai_key
```

Both `inference.py` and `server/app.py` auto-load values from `.env`.

## Run API server

```powershell
python app.py
```

Or with Uvicorn:

```powershell
uvicorn server:app --host 0.0.0.0 --port 7860
```

## Train baseline agent

```powershell
python train.py
```

Default mode trains a PyTorch DQN baseline and creates `dqn_model.pt`.

To force tabular Q-learning:

```powershell
$env:BASELINE_ALGO = "q_table"
python train.py
```

Training always writes `training_progress.png`.

## Run inference

Set required variables for LLM mode:

```powershell
$env:API_BASE_URL = "https://api.openai.com/v1"
$env:MODEL_NAME = "gpt-4o-mini"
$env:HF_TOKEN = "<your-token>"
python inference.py
```

If those variables are in `.env`, you can run `python inference.py` directly.

Heuristic-only mode (no HF token required):

```powershell
$env:HEURISTIC_ONLY = "1"
python inference.py
```

## Validate submission

### PowerShell (Windows)

```powershell
.\scripts\validate-submission.ps1 https://your-space.hf.space
```

### Bash (WSL/Git Bash/macOS/Linux)

```bash
./scripts/validate-submission.sh https://your-space.hf.space
```

## Testing

Run pytest suite:

```powershell
pip install pytest
pytest
```

Run legacy smoke script:

```powershell
python test_env.py
```

## OpenEnv checks

```powershell
openenv validate
```

If `openenv --version` fails, use:

```powershell
openenv --help
```

## Real-time and extensibility endpoints

- `POST /tasks`: register or overwrite a custom task config at runtime.
- `GET /tasks`: list all registered tasks (default + custom).
- `GET /tasks/{task_id}`: inspect one task.
- `POST /tasks/{task_id}/grade`: run grader for a given task.
- `WS /ws`: low-overhead streaming endpoint for `reset`, `step`, and `state` messages.

## Security and rate limiting

Optional environment variables:

- `API_AUTH_TOKEN`: require matching `X-API-Key` header for all API requests.
- `RATE_LIMIT_PER_MINUTE`: per-client request limit (0 disables rate limiting).
- `CORS_ALLOW_ORIGINS`: comma-separated allowed origins.
- `CORS_ALLOW_METHODS`: comma-separated allowed HTTP methods.
- `CORS_ALLOW_HEADERS`: comma-separated allowed request headers.

## CI

GitHub Actions workflow in `.github/workflows/ci.yml` runs:

1. `pytest`
2. `openenv validate`
