# pyright: reportMissingImports=false
"""
agent/dqn_agent.py
PyTorch DQN baseline agent for the Sound Limiter environment.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_TORCH_AVAILABLE = False
_TORCH_IMPORT_ERROR: Exception | None = None

torch: Any = None
nn: Any = None
optim: Any = None

try:
    import torch as _torch
    import torch.nn as _nn
    import torch.optim as _optim

    torch = _torch
    nn = _nn
    optim = _optim
    _TORCH_AVAILABLE = True
except Exception as exc:  # pragma: no cover - runtime availability depends on env
    _TORCH_IMPORT_ERROR = exc


def torch_is_available() -> bool:
    return _TORCH_AVAILABLE


def vectorize_observation(obs_dict: dict) -> np.ndarray:
    """Convert env observation dict into a dense feature vector for DQN."""
    source_levels = obs_dict.get("source_levels", [0.0, 0.0, 0.0])
    if not isinstance(source_levels, list) or len(source_levels) != 3:
        source_levels = [0.0, 0.0, 0.0]

    return np.array(
        [
            float(obs_dict.get("sound_level", 60.0)) / 100.0,
            float(obs_dict.get("gain", 1.0)),
            1.0 if bool(obs_dict.get("above_safe", False)) else 0.0,
            1.0 if bool(obs_dict.get("below_safe", False)) else 0.0,
            float(obs_dict.get("loud_streak", 0.0)) / 10.0,
            float(obs_dict.get("bass_level", 0.0)) / 100.0,
            float(obs_dict.get("mid_level", 0.0)) / 100.0,
            float(obs_dict.get("treble_level", 0.0)) / 100.0,
            float(obs_dict.get("reverb_energy", 0.0)),
            float(source_levels[0]) / 100.0,
            float(source_levels[1]) / 100.0,
            float(source_levels[2]) / 100.0,
        ],
        dtype=np.float32,
    )


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class DQNAgent:
    N_ACTIONS = 4

    def __init__(
        self,
        learning_rate: float = 1e-3,
        discount: float = 0.98,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        batch_size: int = 64,
        replay_capacity: int = 10_000,
        target_sync_every: int = 200,
        seed: int = 42,
    ):
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for DQNAgent but is not available. "
                f"Original import error: {_TORCH_IMPORT_ERROR}"
            )

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device("cpu")
        self.input_dim = len(vectorize_observation({}))

        class QNetwork(nn.Module):
            def __init__(self, input_dim: int, output_dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, output_dim),
                )

            def forward(self, x):
                return self.net(x)

        self._network_cls = QNetwork

        self.policy_net = self._network_cls(self.input_dim, self.N_ACTIONS).to(self.device)
        self.target_net = self._network_cls(self.input_dim, self.N_ACTIONS).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.batch_size = batch_size
        self.target_sync_every = target_sync_every
        self.replay = deque(maxlen=replay_capacity)
        self.learn_steps = 0

    def choose_action(self, obs_dict: dict) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.N_ACTIONS - 1)

        state_vec = vectorize_observation(obs_dict)
        state_tensor = torch.from_numpy(state_vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def remember(self, obs: dict, action: int, reward: float, next_obs: dict, done: bool) -> None:
        self.replay.append(
            Transition(
                state=vectorize_observation(obs),
                action=int(action),
                reward=float(reward),
                next_state=vectorize_observation(next_obs),
                done=bool(done),
            )
        )

    def learn(self, obs: dict, action: int, reward: float, next_obs: dict, done: bool) -> None:
        self.remember(obs, action, reward, next_obs, done)
        if len(self.replay) < self.batch_size:
            return

        batch = random.sample(self.replay, self.batch_size)

        states = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([1.0 if t.done else 0.0 for t in batch], dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1).values
            target_values = rewards + (1.0 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_sync_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str = "dqn_model.pt") -> None:
        payload = {
            "model_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "input_dim": self.input_dim,
            "n_actions": self.N_ACTIONS,
        }
        torch.save(payload, path)

    def load(self, path: str = "dqn_model.pt") -> bool:
        model_path = Path(path)
        if not model_path.exists():
            return False

        payload = torch.load(model_path, map_location=self.device)
        self.policy_net.load_state_dict(payload["model_state_dict"])
        self.target_net.load_state_dict(payload.get("target_state_dict", payload["model_state_dict"]))

        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)

        self.epsilon = float(payload.get("epsilon", self.epsilon_min))
        return True

    def set_eval_mode(self) -> None:
        self.epsilon = 0.0
        self.policy_net.eval()
        self.target_net.eval()
