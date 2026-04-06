"""
environment/sound_env.py
Meeting Room Sound Level Limiter — Core RL Environment

The agent observes sound levels (dB) in a meeting room and takes actions
to keep them in the safe zone (40–70 dB). This simulates a real-world
acoustic management system used in conference rooms.
"""

import numpy as np
import random
from typing import Optional
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════
# Typed Models (OpenEnv spec requires Pydantic models)
# ══════════════════════════════════════════════════════════

class SoundObservation(BaseModel):
    """What the agent can observe about the environment."""
    sound_level:    float = Field(..., ge=0.0, le=100.0, description="Current sound level in dB (0–100)")
    gain:           float = Field(..., ge=0.0, le=1.0,   description="Current gain setting (0=muted, 1=full)")
    step_count:     int   = Field(..., ge=0,              description="Steps elapsed in this episode")
    above_safe:     bool  = Field(...,                    description="True if sound is above 70 dB")
    below_safe:     bool  = Field(...,                    description="True if sound is below 40 dB")
    loud_streak:    int   = Field(..., ge=0,              description="Consecutive steps above 95 dB (danger)")


class SoundAction(BaseModel):
    """An action the agent can take."""
    action_id:   int = Field(..., ge=0, le=3, description="0=do_nothing, 1=warn, 2=reduce_gain, 3=mute")
    action_name: str = Field(...,             description="Human-readable action name")


class SoundReward(BaseModel):
    """Reward signal with explanation."""
    value:       float = Field(..., description="Reward value for this step")
    reason:      str   = Field(..., description="Why this reward was given")
    in_safe_zone: bool = Field(..., description="Whether sound is in the 40–70 dB safe zone")


# ══════════════════════════════════════════════════════════
# Action constants
# ══════════════════════════════════════════════════════════
ACTION_MAP = {
    0: "do_nothing",
    1: "warn",
    2: "reduce_gain",
    3: "mute",
}

SAFE_MIN = 40.0
SAFE_MAX = 70.0
MAX_STEPS = 50


# ══════════════════════════════════════════════════════════
# Environment
# ══════════════════════════════════════════════════════════
class SoundLimiterEnv:
    """
    Meeting Room Sound Level Limiter RL Environment.

    State:   [sound_level, gain, step_count, above_safe, below_safe, loud_streak]
    Actions: do_nothing (0), warn (1), reduce_gain (2), mute (3)
    Reward:  +1.0 if in safe zone, -0.5 if slightly off, -2.0 if dangerously loud
    Done:    After MAX_STEPS steps, or 3 consecutive steps above 95 dB
    """

    def __init__(
        self,
        initial_sound: Optional[float] = None,
        noise_std: float = 3.0,
        max_steps: int = MAX_STEPS,
        seed: Optional[int] = None,
    ):
        self.initial_sound = initial_sound
        self.noise_std = noise_std
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # State variables (initialized in reset)
        self.sound_level: float = 60.0
        self.gain:        float = 1.0
        self.step_count:  int   = 0
        self.loud_streak: int   = 0

        self.reset()

    # ── Core OpenEnv interface ──────────────────────────────

    def reset(self, seed: Optional[int] = None) -> SoundObservation:
        """Reset the environment and return the initial observation."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.initial_sound is not None:
            self.sound_level = float(self.initial_sound)
        else:
            self.sound_level = float(self.rng.uniform(30, 90))

        self.gain        = 1.0
        self.step_count  = 0
        self.loud_streak = 0

        return self._make_observation()

    def step(self, action_id: int) -> tuple[SoundObservation, SoundReward, bool, dict]:
        """
        Take one step in the environment.

        Returns:
            observation: SoundObservation
            reward:      SoundReward
            done:        bool
            info:        dict with extra details
        """
        if action_id not in ACTION_MAP:
            raise ValueError(f"Invalid action {action_id}. Must be 0–3.")

        # ── Apply action effect ─────────────────────────────
        if action_id == 0:   # do_nothing — let room drift
            pass

        elif action_id == 1:  # warn — slight nudge
            self.sound_level = max(0.0, self.sound_level - 3.0)

        elif action_id == 2:  # reduce_gain — meaningful reduction
            self.gain        = max(0.0, self.gain - 0.1)
            self.sound_level = max(0.0, self.sound_level - 5.0)

        elif action_id == 3:  # mute — drastic cut
            self.gain        = 0.0
            self.sound_level = max(0.0, self.sound_level * 0.1)

        # ── Natural drift (room noise) ──────────────────────
        noise = float(self.rng.normal(0, self.noise_std))
        self.sound_level = float(np.clip(self.sound_level + noise, 0.0, 100.0))

        # Room gets louder over time if not actively managed
        if action_id == 0 and self.gain > 0.5:
            drift = float(self.rng.uniform(0.5, 2.5))
            self.sound_level = min(100.0, self.sound_level + drift)

        self.step_count += 1

        # ── Compute reward ──────────────────────────────────
        reward = self._compute_reward()

        # ── Check loud streak ───────────────────────────────
        if self.sound_level > 95:
            self.loud_streak += 1
        else:
            self.loud_streak = 0

        terminated = self.loud_streak >= 3
        truncated  = self.step_count >= self.max_steps
        done       = terminated or truncated

        obs  = self._make_observation()
        info = {
            "action":       ACTION_MAP[action_id],
            "sound_level":  self.sound_level,
            "gain":         self.gain,
            "terminated":   terminated,
            "truncated":    truncated,
        }

        return obs, reward, done, info

    def state(self) -> dict:
        """Return current environment state as a plain dict (OpenEnv spec)."""
        return {
            "sound_level": round(self.sound_level, 2),
            "gain":        round(self.gain, 3),
            "step_count":  self.step_count,
            "loud_streak": self.loud_streak,
            "above_safe":  self.sound_level > SAFE_MAX,
            "below_safe":  self.sound_level < SAFE_MIN,
            "in_safe_zone": SAFE_MIN <= self.sound_level <= SAFE_MAX,
            "max_steps":   self.max_steps,
        }

    # ── Internal helpers ────────────────────────────────────

    def _make_observation(self) -> SoundObservation:
        return SoundObservation(
            sound_level  = round(self.sound_level, 2),
            gain         = round(self.gain, 3),
            step_count   = self.step_count,
            above_safe   = self.sound_level > SAFE_MAX,
            below_safe   = self.sound_level < SAFE_MIN,
            loud_streak  = self.loud_streak,
        )

    def _compute_reward(self) -> SoundReward:
        s = self.sound_level
        if SAFE_MIN <= s <= SAFE_MAX:
            return SoundReward(value=1.0,  reason="Sound in safe zone (40–70 dB)", in_safe_zone=True)
        elif s > 85:
            return SoundReward(value=-2.0, reason=f"Dangerously loud: {s:.1f} dB", in_safe_zone=False)
        elif s < 20:
            return SoundReward(value=-0.5, reason=f"Over-muted: {s:.1f} dB (bad for meeting)", in_safe_zone=False)
        else:
            return SoundReward(value=-0.5, reason=f"Outside safe range: {s:.1f} dB", in_safe_zone=False)

    def render(self) -> str:
        """ASCII render of the current state."""
        bar_len = int(self.sound_level / 2)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        zone = "SAFE" if SAFE_MIN <= self.sound_level <= SAFE_MAX else "  LOUD"
        return (
            f"Step {self.step_count:03d} | "
            f"[{bar}] {self.sound_level:5.1f} dB | "
            f"Gain: {self.gain:.2f} | {zone}"
        )