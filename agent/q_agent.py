import os
import pickle
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


class QLearningAgent:
    N_SOUND_BINS = 10
    N_GAIN_BINS  = 5
    N_ACTIONS    = 4

    def __init__(
        self,
        learning_rate:  float = 0.15,
        discount:       float = 0.95,
        epsilon:        float = 1.0,
        epsilon_decay:  float = 0.995,
        epsilon_min:    float = 0.05,
    ):
        self.lr            = learning_rate
        self.gamma         = discount
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min

        self.q_table = np.zeros((self.N_SOUND_BINS, self.N_GAIN_BINS, self.N_ACTIONS))

    def _discretize(self, obs_dict: dict) -> tuple[int, int]:
        s = obs_dict.get("sound_level", 50.0)
        g = obs_dict.get("gain", 1.0)
        sound_bin = int(np.clip(s / 10, 0, self.N_SOUND_BINS - 1))
        gain_bin  = int(np.clip(g / 0.2, 0, self.N_GAIN_BINS - 1))
        return sound_bin, gain_bin

    def _stability_guard(self, obs_dict: dict, action: int) -> int:
        sl = float(obs_dict.get("sound_level", 60.0))
        gain = float(obs_dict.get("gain", 1.0))

        if action == 3:
            action = 2 if (sl > 92 and gain > 0.25) else 1

        if sl < 42 and action in (1, 2, 3):
            return 0

        if gain < 0.20 and sl < 80 and action == 2:
            return 1 if sl > 70 else 0

        if sl > 92:
            return 2 if gain > 0.20 else 1
        if sl > 78 and action in (0, 1):
            return 2 if gain > 0.35 else 1
        if sl > 70 and action == 0:
            return 1

        return action

    def choose_action(self, obs_dict: dict) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.N_ACTIONS)
        sb, gb = self._discretize(obs_dict)
        action = int(np.argmax(self.q_table[sb, gb]))
        return self._stability_guard(obs_dict, action)

    def learn(
        self,
        obs:      dict,
        action:   int,
        reward:   float,
        next_obs: dict,
        done:     bool,
    ) -> None:
        sb,  gb  = self._discretize(obs)
        sb2, gb2 = self._discretize(next_obs)

        current_q  = self.q_table[sb, gb, action]
        max_next_q = 0.0 if done else float(np.max(self.q_table[sb2, gb2]))
        target_q   = reward + self.gamma * max_next_q

        self.q_table[sb, gb, action] += self.lr * (target_q - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str = "q_table.pkl") -> None:
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table, "epsilon": self.epsilon}, f)

    def load(self, path: str = "q_table.pkl") -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        loaded = data["q_table"]
        if loaded.shape != self.q_table.shape:
            return False
        self.q_table = loaded
        self.epsilon = data.get("epsilon", self.epsilon_min)
        return True