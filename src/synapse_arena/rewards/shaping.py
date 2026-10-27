"""Reward Shaping Interface for SYNAPSE ARENA.

MANDATORY ENGINEERING RULE:
Do not invent unverified signals or assume fields such as 'centerline_distance'
or 'checkpoint_index' exist before the official environment is inspected.
By default, the raw environment reward is passed through completely untouched.
Any custom reward modification must be explicitly enabled, documented, and tested.
"""

from typing import Any, Dict
import gymnasium as gym
from synapse_arena.config.schema import RewardConfig
from synapse_arena.utils.logger import get_logger

logger = get_logger(__name__)


class RewardShaper(gym.RewardWrapper):
    """Gymnasium RewardWrapper providing a modular boundary for reward shaping.

    If config.use_custom_shaping is False (default):
        Returns the raw simulator reward unmodified.

    If config.use_custom_shaping is True:
        Applies configurable, verified reward terms.
    """

    def __init__(self, env: gym.Env, config: RewardConfig) -> None:
        super().__init__(env)
        self.config = config
        self._last_info: Dict[str, Any] = {}

        if self.config.use_custom_shaping:
            logger.info(
                "Custom reward shaping ENABLED with: speed_weight=%.2f, collision_penalty=%.2f",
                self.config.speed_weight,
                self.config.collision_penalty,
            )
        else:
            logger.debug("Reward shaping disabled; passing through raw simulator reward.")

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_info = info
        shaped_reward = self.reward(reward)
        return obs, shaped_reward, terminated, truncated, info

    def reward(self, reward: float) -> float:
        """Computes the shaped reward.

        Args:
            reward: Raw reward from the underlying environment.

        Returns:
            float: Modified or unmodified reward signal.
        """
        if not self.config.use_custom_shaping:
            return float(reward)

        shaped = float(reward)

        # Example: only apply terms if info signals are confirmed in the environment
        if "speed" in self._last_info and self.config.speed_weight > 0.0:
            speed = float(self._last_info["speed"])
            shaped += self.config.speed_weight * speed

        if "is_collision" in self._last_info and self._last_info["is_collision"]:
            shaped -= abs(self.config.collision_penalty)

        return float(shaped)
