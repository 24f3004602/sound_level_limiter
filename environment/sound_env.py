"""
environment/sound_env.py
Meeting Room Sound Level Limiter - Core RL Environment

The dynamics are intentionally acoustic-aware:
- 3 independent sound sources
- band-sensitive control (bass, mid, treble)
- FFT-based band analysis
- room impulse carry-over so muting is not instantaneous silence
"""

from typing import Optional

import numpy as np
from openenv.core import Action as OpenEnvAction
from openenv.core import Observation as OpenEnvObservation
from openenv.core import State as OpenEnvState
from pydantic import BaseModel, Field


# Typed models (OpenEnv spec requires Pydantic models)
class SoundObservation(OpenEnvObservation):
    """What the agent can observe about the environment."""

    sound_level: float = Field(..., ge=0.0, le=100.0, description="Current mixed sound level in dB")
    gain: float = Field(..., ge=0.0, le=1.0, description="Current gain setting (0=muted, 1=full)")
    step_count: int = Field(..., ge=0, description="Steps elapsed in this episode")
    above_safe: bool = Field(..., description="True if sound is above 70 dB")
    below_safe: bool = Field(..., description="True if sound is below 40 dB")
    loud_streak: int = Field(..., ge=0, description="Consecutive steps above 95 dB (danger)")

    # Backward-compatible additions for richer acoustics.
    bass_level: float = Field(default=0.0, ge=0.0, le=100.0, description="Estimated bass band level")
    mid_level: float = Field(default=0.0, ge=0.0, le=100.0, description="Estimated mid band level")
    treble_level: float = Field(default=0.0, ge=0.0, le=100.0, description="Estimated treble band level")
    source_levels: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        min_length=3,
        max_length=3,
        description="Per-source loudness levels for 3 speakers",
    )
    reverb_energy: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalized residual room energy from impulse carry-over",
    )


class SoundAction(OpenEnvAction):
    """An action the agent can take."""

    action_id: int = Field(..., ge=0, le=3, description="0=do_nothing, 1=warn, 2=reduce_gain, 3=mute")
    action_name: str = Field(..., description="Human-readable action name")


class SoundReward(BaseModel):
    """Reward signal with explanation."""

    value: float = Field(..., description="Reward value for this step")
    reason: str = Field(..., description="Why this reward was given")
    in_safe_zone: bool = Field(..., description="Whether sound is in the 40-70 dB safe zone")


class SoundState(OpenEnvState):
    """Internal state model compatible with OpenEnv state typing."""

    sound_level: float = Field(..., ge=0.0, le=100.0, description="Current mixed sound level")
    gain: float = Field(..., ge=0.0, le=1.0, description="Current gain setting")
    step_count: int = Field(..., ge=0, description="Steps elapsed in this episode")
    loud_streak: int = Field(..., ge=0, description="Consecutive dangerously loud steps")
    safe_streak: int = Field(..., ge=0, description="Consecutive steps inside safe zone")
    above_safe: bool = Field(..., description="True if sound is above 70 dB")
    below_safe: bool = Field(..., description="True if sound is below 40 dB")
    in_safe_zone: bool = Field(..., description="True if sound is in the safe 40-70 dB band")
    max_steps: int = Field(..., ge=1, description="Maximum steps allowed in an episode")
    bass_level: float = Field(..., ge=0.0, le=100.0)
    mid_level: float = Field(..., ge=0.0, le=100.0)
    treble_level: float = Field(..., ge=0.0, le=100.0)
    source_levels: list[float] = Field(..., min_length=3, max_length=3)
    reverb_energy: float = Field(..., ge=0.0, le=1.0)


# Action constants
ACTION_MAP = {
    0: "do_nothing",
    1: "warn",
    2: "reduce_gain",
    3: "mute",
}

SAFE_MIN = 40.0
SAFE_MAX = 70.0
TARGET_LEVEL = 55.0
MAX_STEPS = 50

N_SOURCES = 3
N_BANDS = 3


class SoundLimiterEnv:
    """
    Meeting Room Sound Level Limiter RL Environment.

    State (core):  [sound_level, gain, step_count, above_safe, below_safe, loud_streak]
    Actions:       do_nothing (0), warn (1), reduce_gain (2), mute (3)
    Reward:        distance-shaped reward around a 55 dB target with smooth gradients
    Done:          After MAX_STEPS steps, or 3 consecutive steps above 95 dB
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

        self.sample_rate = 8_000
        self.frame_size = 256
        self.time_axis = np.arange(self.frame_size, dtype=np.float32) / float(self.sample_rate)
        self.fft_freqs = np.fft.rfftfreq(self.frame_size, d=1.0 / float(self.sample_rate))
        self._bass_mask = (self.fft_freqs >= 20.0) & (self.fft_freqs < 250.0)
        self._mid_mask = (self.fft_freqs >= 250.0) & (self.fft_freqs < 2_000.0)
        self._treble_mask = (self.fft_freqs >= 2_000.0) & (self.fft_freqs <= 4_000.0)

        self.room_impulse = np.array([1.0, 0.58, 0.34, 0.18, 0.09], dtype=np.float32)
        self.room_memory = np.zeros(len(self.room_impulse) - 1, dtype=np.float32)

        self.sound_level: float = 60.0
        self.gain: float = 1.0
        self.step_count: int = 0
        self.loud_streak: int = 0
        self.safe_streak: int = 0

        self.source_levels_db = np.full(N_SOURCES, 60.0, dtype=np.float32)
        self.source_profiles = np.full((N_SOURCES, N_BANDS), 1.0 / float(N_BANDS), dtype=np.float32)
        self.source_frequencies = np.zeros((N_SOURCES, N_BANDS), dtype=np.float32)
        self.source_phases = np.zeros((N_SOURCES, N_BANDS), dtype=np.float32)
        self.eq_profile = np.ones(N_BANDS, dtype=np.float32)
        self.band_levels_db = np.zeros(N_BANDS, dtype=np.float32)
        self.reverb_energy = 0.0
        self.level_bias = 0.0

        self.reset(seed=seed)

    # Core OpenEnv interface
    def reset(self, seed: Optional[int] = None) -> SoundObservation:
        """Reset the environment and return the initial observation."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.initial_sound is not None:
            base_level = float(self.initial_sound)
        else:
            base_level = float(self.rng.uniform(30.0, 90.0))

        self.source_levels_db = np.clip(
            self.rng.normal(loc=base_level, scale=4.5, size=N_SOURCES),
            10.0,
            100.0,
        ).astype(np.float32)
        self.source_profiles = self.rng.dirichlet([2.2, 2.4, 2.0], size=N_SOURCES).astype(np.float32)
        self.source_frequencies = np.column_stack(
            (
                self.rng.uniform(70.0, 220.0, size=N_SOURCES),
                self.rng.uniform(350.0, 1_500.0, size=N_SOURCES),
                self.rng.uniform(2_200.0, 3_600.0, size=N_SOURCES),
            )
        ).astype(np.float32)
        self.source_phases = self.rng.uniform(0.0, 2.0 * np.pi, size=(N_SOURCES, N_BANDS)).astype(np.float32)

        self.gain = 1.0
        self.step_count = 0
        self.loud_streak = 0
        self.safe_streak = 0
        self.eq_profile = np.ones(N_BANDS, dtype=np.float32)
        self.room_memory.fill(0.0)

        self.level_bias = 0.0
        self._refresh_sound_metrics()

        # If an explicit initial level is requested, calibrate close to it.
        if self.initial_sound is not None:
            self.level_bias = float(np.clip(base_level - self.sound_level, -20.0, 20.0))
            self._refresh_sound_metrics()

        return self._make_observation()

    def step(self, action_id: int) -> tuple[SoundObservation, SoundReward, bool, dict]:
        """
        Take one step in the environment.

        Returns:
            observation: SoundObservation
            reward: SoundReward
            done: bool
            info: dict with extra details
        """
        if action_id not in ACTION_MAP:
            raise ValueError(f"Invalid action {action_id}. Must be 0-3.")

        self._apply_action(action_id)
        self._apply_natural_drift(action_id)
        self._refresh_sound_metrics()

        self.step_count += 1
        in_safe_zone = SAFE_MIN <= self.sound_level <= SAFE_MAX
        if in_safe_zone:
            self.safe_streak += 1
        else:
            self.safe_streak = 0

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
            "band_levels": {
                "bass": float(self.band_levels_db[0]),
                "mid": float(self.band_levels_db[1]),
                "treble": float(self.band_levels_db[2]),
            },
            "source_levels": [float(x) for x in self.source_levels_db.tolist()],
            "reverb_energy": self.reverb_energy,
            "terminated": terminated,
            "truncated": truncated,
        }

        return obs, reward, done, info

    def state(self) -> dict:
        """Return current environment state as a plain dict (OpenEnv spec)."""
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
                "bass_level": round(float(self.band_levels_db[0]), 2),
                "mid_level": round(float(self.band_levels_db[1]), 2),
                "treble_level": round(float(self.band_levels_db[2]), 2),
                "source_levels": [round(float(x), 2) for x in self.source_levels_db.tolist()],
                "reverb_energy": round(self.reverb_energy, 4),
            }
        )
        return state.model_dump()

    # Internal helpers
    def _apply_action(self, action_id: int) -> None:
        band_profiles = {
            0: np.array([1.0, 1.0, 1.0], dtype=np.float32),
            1: np.array([0.98, 0.93, 0.90], dtype=np.float32),
            2: np.array([0.94, 0.86, 0.78], dtype=np.float32),
            3: np.array([0.72, 0.55, 0.35], dtype=np.float32),
        }

        if action_id == 1:
            loudest = int(np.argmax(self.source_levels_db))
            self.source_levels_db[loudest] = max(
                0.0,
                float(self.source_levels_db[loudest] - self.rng.uniform(2.5, 4.5)),
            )
        elif action_id == 2:
            self.gain = max(0.05, self.gain - 0.08)
            self.source_levels_db = np.maximum(
                0.0,
                self.source_levels_db - self.rng.uniform(1.5, 3.0, size=N_SOURCES),
            ).astype(np.float32)
        elif action_id == 3:
            self.gain = max(0.02, self.gain * 0.40)
            self.source_levels_db = np.maximum(
                0.0,
                self.source_levels_db - self.rng.uniform(6.0, 10.0, size=N_SOURCES),
            ).astype(np.float32)

        target_profile = band_profiles[action_id]
        self.eq_profile = np.clip(0.82 * self.eq_profile + 0.18 * target_profile, 0.2, 1.10)

    def _apply_natural_drift(self, action_id: int) -> None:
        baseline_push = 0.7 if (action_id == 0 and self.gain > 0.40) else 0.1
        drift = self.rng.normal(loc=baseline_push, scale=max(0.2, self.noise_std * 0.35), size=N_SOURCES)
        self.source_levels_db = np.clip(self.source_levels_db + drift, 0.0, 100.0).astype(np.float32)

        # Let spectra evolve slightly, then renormalize rows.
        profile_jitter = self.rng.normal(0.0, 0.02, size=(N_SOURCES, N_BANDS))
        self.source_profiles = np.clip(self.source_profiles + profile_jitter, 0.05, 0.90)
        self.source_profiles = self.source_profiles / self.source_profiles.sum(axis=1, keepdims=True)

        # Occasional speaking spike when unmanaged.
        if action_id == 0 and self.gain > 0.60 and self.rng.random() < 0.35:
            idx = int(self.rng.integers(0, N_SOURCES))
            self.source_levels_db[idx] = min(
                100.0,
                float(self.source_levels_db[idx] + self.rng.uniform(0.8, 2.8)),
            )

        # Gradual recovery of EQ profile if action pressure is low.
        if action_id == 0:
            self.eq_profile = np.clip(0.96 * self.eq_profile + 0.04 * np.ones(N_BANDS), 0.2, 1.10)

    def _synthesize_dry_frame(self) -> np.ndarray:
        dry = np.zeros(self.frame_size, dtype=np.float32)
        for src_idx in range(N_SOURCES):
            src_level = float(self.source_levels_db[src_idx])
            amplitude = 10.0 ** ((src_level - 55.0) / 20.0)

            for band_idx in range(N_BANDS):
                coeff = (
                    amplitude
                    * float(self.source_profiles[src_idx, band_idx])
                    * float(self.eq_profile[band_idx])
                )
                frequency = float(self.source_frequencies[src_idx, band_idx])
                phase = float(self.source_phases[src_idx, band_idx])

                dry += coeff * np.sin((2.0 * np.pi * frequency * self.time_axis) + phase)

                self.source_phases[src_idx, band_idx] = float(
                    (phase + self.rng.normal(0.0, 0.12)) % (2.0 * np.pi)
                )

        dry *= float(self.gain / float(N_SOURCES))
        return dry

    def _apply_room_impulse(self, dry_frame: np.ndarray) -> np.ndarray:
        padded = np.concatenate((self.room_memory, dry_frame))
        convolved = np.convolve(padded, self.room_impulse, mode="full")
        start = len(self.room_memory)
        wet = convolved[start : start + self.frame_size]
        self.room_memory = padded[-(len(self.room_impulse) - 1) :].astype(np.float32)
        return wet.astype(np.float32)

    def _fft_band_powers(self, frame: np.ndarray) -> np.ndarray:
        windowed = frame * np.hanning(self.frame_size)
        spectrum = np.abs(np.fft.rfft(windowed)) ** 2

        def _mean_power(mask: np.ndarray) -> float:
            if not np.any(mask):
                return 0.0
            return float(np.mean(spectrum[mask]))

        return np.array(
            [
                _mean_power(self._bass_mask),
                _mean_power(self._mid_mask),
                _mean_power(self._treble_mask),
            ],
            dtype=np.float32,
        )

    def _refresh_sound_metrics(self) -> None:
        dry = self._synthesize_dry_frame()
        wet = self._apply_room_impulse(dry)

        band_powers = self._fft_band_powers(wet)
        raw_band_db = (10.0 * np.log10(band_powers + 1e-8)) + 65.0 + self.level_bias
        self.band_levels_db = np.clip(raw_band_db, 0.0, 100.0).astype(np.float32)

        rms = float(np.sqrt(np.mean(np.square(wet))) + 1e-8)
        raw_sound_db = (20.0 * np.log10(rms)) + 75.0 + self.level_bias
        self.sound_level = float(np.clip(raw_sound_db, 0.0, 100.0))

        self.reverb_energy = float(
            np.clip(np.sqrt(np.mean(np.square(self.room_memory))) * 3.5, 0.0, 1.0)
        )

    def _make_observation(self) -> SoundObservation:
        return SoundObservation.model_validate(
            {
                "sound_level": round(self.sound_level, 2),
                "gain": round(self.gain, 3),
                "step_count": self.step_count,
                "above_safe": self.sound_level > SAFE_MAX,
                "below_safe": self.sound_level < SAFE_MIN,
                "loud_streak": self.loud_streak,
                "bass_level": round(float(self.band_levels_db[0]), 2),
                "mid_level": round(float(self.band_levels_db[1]), 2),
                "treble_level": round(float(self.band_levels_db[2]), 2),
                "source_levels": [round(float(x), 2) for x in self.source_levels_db.tolist()],
                "reverb_energy": round(self.reverb_energy, 4),
            }
        )

    def _compute_reward(self) -> SoundReward:
        sound = float(self.sound_level)
        distance = abs(sound - TARGET_LEVEL)

        # Smooth shaping everywhere: closer to 55 dB means better reward.
        shaped_value = 1.0 - (distance / 30.0)

        # Strong non-linear penalties near unsafe extremes.
        if sound > 90.0:
            shaped_value -= 0.6 + ((sound - 90.0) / 15.0)
        elif sound < 15.0:
            shaped_value -= 0.35 + ((15.0 - sound) / 20.0)

        # Meaningful consistency bonus for sustained safe control.
        consistency_bonus = min(0.30, 0.04 * float(self.safe_streak))
        value = float(np.clip(shaped_value + consistency_bonus, -2.0, 1.25))

        in_safe_zone = SAFE_MIN <= sound <= SAFE_MAX
        direction = "in safe zone" if in_safe_zone else ("too loud" if sound > SAFE_MAX else "too quiet")
        reason = (
            f"Distance-shaped reward: level={sound:.1f} dB, "
            f"target={TARGET_LEVEL:.1f} dB, safe_streak={self.safe_streak} ({direction})"
        )

        return SoundReward(value=value, reason=reason, in_safe_zone=in_safe_zone)

    def render(self) -> str:
        """ASCII render of the current state."""
        bar_len = int(self.sound_level / 2)
        bar = "#" * bar_len + "-" * (50 - bar_len)
        if SAFE_MIN <= self.sound_level <= SAFE_MAX:
            zone = "SAFE"
        elif self.sound_level > SAFE_MAX:
            zone = "LOUD"
        else:
            zone = "QUIET"
        return (
            f"Step {self.step_count:03d} | "
            f"[{bar}] {self.sound_level:5.1f} dB | "
            f"Gain: {self.gain:.2f} | "
            f"Bands(B/M/T): {self.band_levels_db[0]:5.1f}/{self.band_levels_db[1]:5.1f}/{self.band_levels_db[2]:5.1f} | "
            f"{zone}"
        )