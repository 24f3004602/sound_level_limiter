"""
agent/q_agent.py
Simple Q-Learning Agent — Baseline for local training/testing

This trains a policy purely from environment interaction.
It is NOT used in inference.py (which uses an LLM agent).
It's here to: (a) verify the env works, (b) show learning progress,
(c) provide a non-LLM baseline score for comparison.
"""

import numpy as np
import pickle
import os
import environment.sound_env
from environment.sound_env import SoundLimiterEnv, ACTION_MAP


class QLearningAgent:
    """
    Tabular Q-Learning with discretized (sound_level, gain) states.

    State space:  10 sound bins × 5 gain bins = 50 discrete states
    Action space: 4 actions (do_nothing, warn, reduce_gain, mute)
    """

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

        # Q-table: shape [sound_bins, gain_bins, actions]
        self.q_table = np.zeros((self.N_SOUND_BINS, self.N_GAIN_BINS, self.N_ACTIONS))

    def _discretize(self, obs_dict: dict) -> tuple[int, int]:
        """Map continuous observation to discrete (sound_bin, gain_bin)."""
        s = obs_dict.get("sound_level", 50.0)
        g = obs_dict.get("gain", 1.0)
        sound_bin = int(np.clip(s / 10, 0, self.N_SOUND_BINS - 1))
        gain_bin  = int(np.clip(g / 0.2, 0, self.N_GAIN_BINS - 1))
        return sound_bin, gain_bin

    def choose_action(self, obs_dict: dict) -> int:
        """Epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.N_ACTIONS)
        sb, gb = self._discretize(obs_dict)
        return int(np.argmax(self.q_table[sb, gb]))

    def learn(
        self,
        obs:      dict,
        action:   int,
        reward:   float,
        next_obs: dict,
        done:     bool,
    ) -> None:
        """Bellman update."""
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
        self.q_table = data["q_table"]
        self.epsilon = data.get("epsilon", self.epsilon_min)
        return True