"""Synthetic Smoke-Test Racing Environment for Pipeline Verification.

CRITICAL WARNING:
This environment is SOLELY a lightweight local software test fixture designed
to verify package imports, configuration loading, Stable-Baselines3 algorithm
instantiation, training step loops, checkpoint saving/loading, and evaluation runners.
IT DOES NOT REPRESENT OFFICIAL SIMULATION DYNAMICS, REAL CAR PHYSICS, OR RACING PERFORMANCE.
NEVER REPORT REWARDS OR COMPLETION METRICS FROM THIS ENVIRONMENT AS RACING RESULTS.
"""

from typing import Any, Dict, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from synapse_arena.envs.base import BaseRacingEnv


class SyntheticSmokeTestRacingEnv(BaseRacingEnv):
    """Isolated, minimal synthetic racing environment for automated smoke tests.

    Observation Space:
        Box(shape=(6,), dtype=float32)
        - [0..4]: 5 synthetic distance sensor readings [0.0, 50.0] meters.
        - [5]: Synthetic forward speed [0.0, 1.0] normalized.

    Action Space:
        Box(shape=(2,), dtype=float32)
        - [0]: Steering angle [-1.0, 1.0] (negative = left, positive = right).
        - [1]: Throttle / Brake [-1.0, 1.0] (positive = accelerate, negative = brake).
    """

    def __init__(self, track: str | None = None, render_mode: str | None = None) -> None:
        super().__init__(track=track, render_mode=render_mode)

        # 5 rangefinder sensors + 1 speed indicator
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([50.0, 50.0, 50.0, 50.0, 50.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Steering [-1, 1], Throttle/Brake [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._step_count = 0
        self._max_steps = 100
        self._speed = 0.0
        self._distance_traveled = 0.0
        self._x = 0.0
        self._y = 0.0
        self._heading = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        self._speed = 0.1
        self._distance_traveled = 0.0
        self._x = 0.0
        self._y = 0.0
        self._heading = 0.0

        obs = self._get_observation()
        info = {
            "speed": self._speed,
            "distance_traveled": self._distance_traveled,
            "x": self._x,
            "y": self._y,
            "heading": self._heading,
            "is_synthetic": True,
            "track": self.track or "synthetic_default",
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1

        # Clip action to valid bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        steering, throttle = float(action[0]), float(action[1])

        # Simple synthetic speed & kinematic pose update
        self._speed = float(np.clip(self._speed + 0.1 * throttle - 0.02 * abs(steering), 0.0, 1.0))
        self._distance_traveled += self._speed
        self._heading += float(0.08 * steering)
        self._x += float(self._speed * np.cos(self._heading))
        self._y += float(self._speed * np.sin(self._heading))

        obs = self._get_observation()

        # Synthetic baseline reward: speed minus severe steering penalty
        reward = float(self._speed - 0.1 * abs(steering))

        # Termination: step limit reached or synthetic boundary
        terminated = False
        truncated = self._step_count >= self._max_steps

        info = {
            "speed": self._speed,
            "distance_traveled": self._distance_traveled,
            "x": self._x,
            "y": self._y,
            "heading": self._heading,
            "is_synthetic": True,
            "track": self.track or "synthetic_default",
            "step_count": self._step_count,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        # Generate stable synthetic sensor readings
        sensors = np.array([25.0, 30.0, 35.0, 30.0, 25.0], dtype=np.float32)
        # Add tiny variance depending on step count for dynamic observation
        perturb = 0.5 * np.sin(np.arange(5) + self._step_count * 0.1)
        sensors = np.clip(sensors + perturb, 0.0, 50.0).astype(np.float32)
        speed_obs = np.array([self._speed], dtype=np.float32)
        return np.concatenate([sensors, speed_obs])

    @property
    def is_official(self) -> bool:
        return False
