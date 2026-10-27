"""Environment Factory for instantiating racing environments."""

import gymnasium as gym
from synapse_arena.config.schema import EnvConfig
from synapse_arena.envs.base import BaseRacingEnv
from synapse_arena.envs.official_adapter import OfficialEnvironmentAdapter
from synapse_arena.envs.synthetic_env import SyntheticSmokeTestRacingEnv


def make_racing_env(config: EnvConfig, seed: int | None = None) -> BaseRacingEnv:
    """Instantiates a racing environment based on configuration.

    Args:
        config: Environment configuration instance.
        seed: Optional random seed.

    Returns:
        BaseRacingEnv: The instantiated environment adhering to the BaseRacingEnv contract.

    Raises:
        EnvironmentNotIntegratedError: If config.name is 'official' and official env is pending.
        ValueError: If config.name is unrecognized.
    """
    if config.name == "synthetic":
        env = SyntheticSmokeTestRacingEnv(track=config.track, render_mode=config.render_mode)
    elif config.name == "official":
        env = OfficialEnvironmentAdapter(track=config.track, render_mode=config.render_mode)
    else:
        raise ValueError(f"Unknown environment name: '{config.name}'. Choose 'synthetic' or 'official'.")

    if seed is not None:
        env.reset(seed=seed)

    return env
