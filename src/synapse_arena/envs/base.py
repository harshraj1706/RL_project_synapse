"""Abstract Base Racing Environment interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import gymnasium as gym


class BaseRacingEnv(gym.Env, ABC):
    """Abstract base class establishing the contract for all racing environments in SYNAPSE ARENA.

    All concrete implementations (both the official adapter and the synthetic test fixture)
    must adhere to standard Gymnasium conventions:
    - reset() -> (observation, info)
    - step(action) -> (observation, reward, terminated, truncated, info)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, track: str | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        self.track = track
        self.render_mode = render_mode

    @abstractmethod
    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Resets the environment to an initial state.

        Returns:
            Tuple[Any, Dict[str, Any]]: (initial_observation, info_dict)
        """
        super().reset(seed=seed)

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Applies an action and advances the simulator.

        Returns:
            Tuple[Any, float, bool, bool, Dict[str, Any]]:
            (observation, reward, terminated, truncated, info_dict)
        """
        pass

    @property
    @abstractmethod
    def is_official(self) -> bool:
        """Returns True if this is the official competition environment, False if synthetic."""
        pass
