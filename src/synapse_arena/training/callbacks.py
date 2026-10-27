"""Stable-Baselines3 callbacks for training monitoring and checkpointing."""

from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from synapse_arena.utils.logger import get_logger

logger = get_logger(__name__)


class ProgressLoggerCallback(BaseCallback):
    """Custom callback to log training progress every N steps."""

    def __init__(self, log_freq: int = 50, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            logger.info("Training step: %d / %d", self.num_timesteps, self.locals.get("total_timesteps", 0))
        return True
