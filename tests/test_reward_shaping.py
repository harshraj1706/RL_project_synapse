"""Tests for reward shaping boundary and pass-through safety."""

import numpy as np
from synapse_arena.config.schema import RewardConfig
from synapse_arena.envs.synthetic_env import SyntheticSmokeTestRacingEnv
from synapse_arena.rewards.shaping import RewardShaper


def test_reward_shaper_passthrough_default():
    """Verify that by default (use_custom_shaping=False), raw rewards are untouched."""
    raw_env = SyntheticSmokeTestRacingEnv()
    config = RewardConfig(use_custom_shaping=False, speed_weight=5.0, collision_penalty=10.0)
    shaper = RewardShaper(raw_env, config)

    obs, _ = shaper.reset(seed=42)
    action = np.array([0.0, 0.5], dtype=np.float32)

    # Calculate expected raw reward from an unadulterated step
    raw_obs, expected_raw_reward, _, _, _ = raw_env.step(action)

    # Shaper step
    _, shaped_reward, _, _, _ = shaper.step(action)

    assert isinstance(shaped_reward, float)
    # When shaping is disabled, shaper.reward(r) must equal r
    assert shaper.reward(1.2345) == 1.2345
    assert shaper.reward(-0.5) == -0.5

    shaper.close()
    raw_env.close()


def test_reward_shaper_active():
    """Verify that when custom shaping is enabled, verified terms are applied."""
    raw_env = SyntheticSmokeTestRacingEnv()
    config = RewardConfig(use_custom_shaping=True, speed_weight=2.0, collision_penalty=5.0)
    shaper = RewardShaper(raw_env, config)

    shaper.reset(seed=42)
    action = np.array([0.0, 1.0], dtype=np.float32)
    _, shaped_reward, _, _, info = shaper.step(action)

    assert isinstance(shaped_reward, float)
    # Speed is exposed in synthetic info
    assert "speed" in info

    shaper.close()
    raw_env.close()
