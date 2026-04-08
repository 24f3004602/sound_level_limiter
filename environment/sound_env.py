from typing import Optional

import numpy as np
from openenv.core import Action as OpenEnvAction
from openenv.core import Observation as OpenEnvObservation
from openenv.core import State as OpenEnvState
from pydantic import BaseModel, Field


class SoundObservation(OpenEnvObservation):
    sound_level: float = Field(..., ge=0.0, le=100.0)
    gain: float = Field(..., ge=0.0, le=1.0)
    step_count: int = Field(..., ge=0)
    above_safe: bool
    below_safe: bool
    loud_streak: int = Field(..., ge=0)
    source_levels: list[float] = Field(..., description="dB level of each of the 3 sound sources")

    bass_level: float = Field(default=0.0, ge=0.0, le=100.0)
    mid_level: float = Field(default=0.0, ge=0.0, le=100.0)
    treble_level: float = Field(default=0.0, ge=0.0, le=100.0)
    reverb_energy: float = Field(default=0.0, ge=0.0, le=1.0)


class SoundAction(OpenEnvAction):
    action_id: int = Field(..., ge=0, le=3, description="0=do_nothing, 1=warn, 2=reduce_gain, 3=mute")
    action_name: str = Field(..., description="Human-readable action name")


class SoundReward(BaseModel):
    value: float = Field(..., description="Reward value for this step")
    reason: str = Field(..., description="Why this reward was given")
    in_safe_zone: bool = Field(..., description="Whether sound is in the 40-70 dB safe zone")


class SoundState(OpenEnvState):
    sound_level: float = Field(..., ge=0.0, le=100.0)
    gain: float = Field(..., ge=0.0, le=1.0)
    step_count: int = Field(default=0, ge=0)
    loud_streak: int = Field(..., ge=0)
    safe_streak: int = Field(..., ge=0)
    above_safe: bool
    below_safe: bool
    in_safe_zone: bool
    max_steps: int = Field(..., ge=1)
    source_levels: list[float] = Field(..., description="dB level of each sound source")
    bass_level: float = Field(..., ge=0.0, le=100.0)
    mid_level: float = Field(..., ge=0.0, le=100.0)
    treble_level: float = Field(..., ge=0.0, le=100.0)
    reverb_energy: float = Field(..., ge=0.0, le=1.0)


ACTION_MAP = {
    0: "do_nothing",
    1: "warn",
    2: "reduce_gain",
    3: "mute",
}

SAFE_MIN = 40.0
SAFE_MAX = 70.0
MAX_STEPS = 50


class SoundLimiterEnv:
    def __init__(
        self,
        initial_sound: Optional[float] = None,
        noise_std: float = 3.0,
        max_steps: int = MAX_STEPS,
        seed: Optional[int] = None,
        n_sources: int = 3,
    ):
        self.initial_sound = initial_sound
        self.noise_std = noise_std
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.n_sources = max(1, int(n_sources))
        self.source_levels: list[float] = [0.0] * self.n_sources
        self.safe_streak: int = 0

        self.sound_level: float = 60.0
        self.gain: float = 1.0
        self.step_count: int = 0
        self.loud_streak: int = 0

        self.source_profiles = np.full((self.n_sources, 3), 1.0 / 3.0, dtype=np.float32)
        self.band_levels_db = np.zeros(3, dtype=np.float32)
        self.reverb_energy = 0.0
        self._room_tail_db = 0.0

        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None) -> SoundObservation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.initial_sound is not None:
            base_sound = float(self.initial_sound)
        else:
            base_sound = float(self.rng.uniform(30.0, 90.0))

        self.source_levels = np.clip(
            self.rng.normal(loc=base_sound, scale=4.0, size=self.n_sources),
            0.0,
            100.0,
        ).astype(np.float32).tolist()

        self.source_profiles = self.rng.dirichlet([2.3, 2.4, 2.0], size=self.n_sources).astype(np.float32)

        self.gain = 1.0
        self.step_count = 0
        self.loud_streak = 0
        self.safe_streak = 0

        mixed = self._mix_sources()
        self._room_tail_db = mixed
        self.sound_level = float(np.clip(mixed, 0.0, 100.0))
        self._update_band_features()

        return self._make_observation()

    def _mix_sources(self) -> float:
        linear = [10 ** (s / 10.0) for s in self.source_levels]
        total_power = sum(linear) * max(self.gain, 0.0)
        return float(10.0 * np.log10(total_power + 1e-9))

    def step(self, action_id: int) -> tuple[SoundObservation, SoundReward, bool, dict]:
        if action_id not in ACTION_MAP:
            raise ValueError(f"Invalid action {action_id}. Must be 0-3.")

        if action_id == 1:  # warn
            self.source_levels = [max(0.0, s - 2.0) for s in self.source_levels]
        elif action_id == 2:  # reduce_gain
            self.gain = max(0.0, self.gain - 0.1)
            self.source_levels = [max(0.0, s - 4.0) for s in self.source_levels]
        elif action_id == 3:  # mute
            self.gain = 0.0
            self.source_levels = [s * 0.1 for s in self.source_levels]

        for i in range(self.n_sources):
            noise = float(self.rng.normal(0.0, self.noise_std))
            self.source_levels[i] = float(np.clip(self.source_levels[i] + noise, 0.0, 100.0))

        mixed_now = self._mix_sources()

        self._room_tail_db = (0.7 * self._room_tail_db) + (0.3 * mixed_now)
        self.sound_level = float(np.clip((0.85 * mixed_now) + (0.15 * self._room_tail_db), 0.0, 100.0))
        self._update_band_features()

        self.step_count += 1
        reward = self._compute_reward()

        if self.sound_level > 95.0:
            self.loud_streak += 1
        else:
            self.loud_streak = 0

        terminated = self.loud_streak >= 3
        truncated = self.step_count >= self.max_steps
        done = terminated or truncated

        obs = self._make_observation()
        info = {
            "action": ACTION_MAP[action_id],
            "sound_level": self.sound_level,
            "gain": self.gain,
            "source_levels": [float(x) for x in self.source_levels],
            "band_levels": {
                "bass": float(self.band_levels_db[0]),
                "mid": float(self.band_levels_db[1]),
                "treble": float(self.band_levels_db[2]),
            },
            "reverb_energy": self.reverb_energy,
            "terminated": terminated,
            "truncated": truncated,
        }

        return obs, reward, done, info

    def state(self) -> dict:
        state = SoundState.model_validate(
            {
                "sound_level": round(self.sound_level, 2),
                "gain": round(self.gain, 3),
                "step_count": self.step_count,
                "loud_streak": self.loud_streak,
                "safe_streak": self.safe_streak,
                "above_safe": self.sound_level > SAFE_MAX,
                "below_safe": self.sound_level < SAFE_MIN,
                "in_safe_zone": SAFE_MIN <= self.sound_level <= SAFE_MAX,
                "max_steps": self.max_steps,
                "source_levels": [round(float(x), 2) for x in self.source_levels],
                "bass_level": round(float(self.band_levels_db[0]), 2),
                "mid_level": round(float(self.band_levels_db[1]), 2),
                "treble_level": round(float(self.band_levels_db[2]), 2),
                "reverb_energy": round(float(self.reverb_energy), 4),
            }
        )
        return state.model_dump()

    def _update_band_features(self) -> None:
        powers = np.zeros(3, dtype=np.float64)
        for source_db, profile in zip(self.source_levels, self.source_profiles):
            source_power = 10 ** (float(source_db) / 10.0)
            powers += source_power * profile

        room_tail_power = 10 ** (self._room_tail_db / 10.0)
        powers += 0.08 * room_tail_power

        band_db = 10.0 * np.log10(powers + 1e-9)
        self.band_levels_db = np.clip(band_db, 0.0, 100.0).astype(np.float32)

        diff = max(0.0, self._room_tail_db - self.sound_level)
        self.reverb_energy = float(np.clip(diff / 40.0, 0.0, 1.0))

    def _make_observation(self) -> SoundObservation:
        return SoundObservation.model_validate(
            {
                "sound_level": round(self.sound_level, 2),
                "gain": round(self.gain, 3),
                "step_count": self.step_count,
                "above_safe": self.sound_level > SAFE_MAX,
                "below_safe": self.sound_level < SAFE_MIN,
                "loud_streak": self.loud_streak,
                "source_levels": [round(float(x), 2) for x in self.source_levels],
                "bass_level": round(float(self.band_levels_db[0]), 2),
                "mid_level": round(float(self.band_levels_db[1]), 2),
                "treble_level": round(float(self.band_levels_db[2]), 2),
                "reverb_energy": round(float(self.reverb_energy), 4),
            }
        )

    def _compute_reward(self) -> SoundReward:
        s = float(self.sound_level)
        centre = (SAFE_MIN + SAFE_MAX) / 2.0
        half_band = (SAFE_MAX - SAFE_MIN) / 2.0

        if SAFE_MIN <= s <= SAFE_MAX:
            self.safe_streak += 1

            dist_to_centre = abs(s - centre)
            centre_bonus = max(0.0, 1.0 - (dist_to_centre / half_band))
            streak_bonus = min(0.20, self.safe_streak * 0.02)
            value = 0.95 + (0.10 * centre_bonus) + streak_bonus

            return SoundReward(
                value=round(float(np.clip(value, -2.0, 1.25)), 4),
                reason=(
                    f"In safe zone ({s:.1f} dB), "
                    f"center_bonus={centre_bonus:.2f}, streak={self.safe_streak}"
                ),
                in_safe_zone=True,
            )

        self.safe_streak = 0

        if s < SAFE_MIN:
            boundary_distance = SAFE_MIN - s
            side = "below"
            side_multiplier = 1.0
        else:
            boundary_distance = s - SAFE_MAX
            side = "above"
            side_multiplier = 1.15

        normalized = min(1.0, boundary_distance / 40.0)
        penalty = -0.20 - side_multiplier * (1.20 * normalized + 0.60 * (normalized ** 2))
        value = float(np.clip(penalty, -2.0, -0.05))

        return SoundReward(
            value=round(value, 4),
            reason=(
                f"Outside safe range ({side}): {s:.1f} dB, "
                f"distance={boundary_distance:.1f}, scaled={normalized:.2f}"
            ),
            in_safe_zone=False,
        )

    def render(self) -> str:
        bar_len = int(self.sound_level / 2)
        bar = "#" * bar_len + "-" * (50 - bar_len)
        zone = "SAFE" if SAFE_MIN <= self.sound_level <= SAFE_MAX else "LOUD"
        return (
            f"Step {self.step_count:03d} | "
            f"[{bar}] {self.sound_level:5.1f} dB | "
            f"Gain: {self.gain:.2f} | "
            f"Sources: {[round(s, 1) for s in self.source_levels]} | "
            f"{zone}"
        )
