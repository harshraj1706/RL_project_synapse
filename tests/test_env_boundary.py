"""Tests for environment boundaries, adapter guards, and synthetic environment compliance."""

import pytest
import numpy as np
from gymnasium.utils.env_checker import check_env
from synapse_arena.config.schema import EnvConfig
from synapse_arena.envs.base import BaseRacingEnv
from synapse_arena.envs.official_adapter import (
    OfficialEnvironmentAdapter,
    EnvironmentNotIntegratedError,
)
from synapse_arena.envs.synthetic_env import SyntheticSmokeTestRacingEnv
from synapse_arena.envs.factory import make_racing_env


def test_official_adapter_guard_raises_error():
    """Verify that unintegrated official adapter strictly raises EnvironmentNotIntegratedError."""
    with pytest.raises(EnvironmentNotIntegratedError) as exc_info:
        OfficialEnvironmentAdapter()
    assert "The official SYNAPSE ARENA racing simulator has not yet been integrated" in str(exc_info.value)


def test_factory_with_official_env_raises_error():
    """Verify that factory raises EnvironmentNotIntegratedError when official env requested."""
    config = EnvConfig(name="official")
    with pytest.raises(EnvironmentNotIntegratedError):
        make_racing_env(config)


def test_synthetic_env_contract_and_gymnasium_compliance():
    """Verify that the synthetic smoke test env satisfies BaseRacingEnv contract and Gymnasium API."""
    env = SyntheticSmokeTestRacingEnv()
    assert isinstance(env, BaseRacingEnv)
    assert env.is_official is False, "Synthetic environment must never be flagged as official."

    # Validate against standard Gymnasium environment specifications
    check_env(env.unwrapped)

    # Validate step and reset semantics
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (6,)
    assert info.get("is_synthetic") is True

    action = np.array([0.0, 0.5], dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == (6,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert step_info.get("is_synthetic") is True

    env.close()


def test_factory_creates_synthetic_env():
    """Verify that factory creates synthetic environment when configured."""
    config = EnvConfig(name="synthetic")
    env = make_racing_env(config, seed=123)
    assert isinstance(env, SyntheticSmokeTestRacingEnv)
    env.close()
