"""Official Simulator Adapter Boundary for SYNAPSE ARENA.

This module provides the integration boundary between the competition organizer's
simulator and our Gymnasium-based training and evaluation pipeline.
"""

from typing import Any, Dict, Tuple
from synapse_arena.envs.base import BaseRacingEnv


class EnvironmentNotIntegratedError(RuntimeError):
    """Raised when attempting to instantiate the official simulator before integration."""
    pass


class OfficialEnvironmentAdapter(BaseRacingEnv):
    """Adapter bridging the organizer's official simulator to the Gymnasium API.

    NOTE: During Phase 1, the official environment package has not yet been released.
    This adapter intentionally raises EnvironmentNotIntegratedError when instantiated
    to prevent accidental mock runs being mistaken for real environment integration.

    Integration steps for Phase 2:
    1. Inspect organizer package imports and API in docs/ENVIRONMENT_CONTRACT.md.
    2. Import the official environment class here.
    3. Initialize official simulator instance in __init__.
    4. Implement observation and action conversion in reset() and step().
    5. Map collision / lap completion signals to terminated / truncated flags.
    """

    def __init__(self, track: str | None = None, render_mode: str | None = None) -> None:
        super().__init__(track=track, render_mode=render_mode)
        raise EnvironmentNotIntegratedError(
            "The official SYNAPSE ARENA racing simulator has not yet been integrated into "
            "synapse_arena.envs.official_adapter.OfficialEnvironmentAdapter.\n"
            "Organizers are expected to supply the simulator files around October 26-27, 2026.\n"
            "To run the Phase 1 software pipeline smoke test, set env.name: 'synthetic' in your config.\n"
            "Refer to docs/ENVIRONMENT_CONTRACT.md for the step-by-step integration checklist."
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError("Pending Phase 2 official integration.")

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        raise NotImplementedError("Pending Phase 2 official integration.")

    @property
    def is_official(self) -> bool:
        return True
