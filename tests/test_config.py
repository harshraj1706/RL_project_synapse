"""Tests for configuration loading, dataclass parsing, and validation."""

import pytest
from pathlib import Path
from synapse_arena.config.schema import (
    AppConfig,
    load_config,
    save_config,
    EnvConfig,
    AlgorithmConfig,
    TrainingConfig,
    EvaluationConfig,
    RewardConfig,
)


def test_default_config_loads_successfully():
    """Verify that the default YAML configuration is valid and parses properly."""
    config_path = Path("configs/default.yaml")
    assert config_path.is_file(), "configs/default.yaml should exist."
    config = load_config(config_path)

    assert isinstance(config, AppConfig)
    assert config.seed == 42
    assert config.env.name == "synthetic"
    assert config.algorithm.name == "PPO"
    assert config.training.total_timesteps == 64
    assert config.evaluation.episodes == 2
    assert config.reward.use_custom_shaping is False


def test_missing_config_file_raises_error():
    """Verify that a nonexistent configuration path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_path_to_config.yaml")


def test_invalid_seed_raises_error():
    """Verify that negative seed fails validation."""
    cfg = AppConfig(seed=-1)
    with pytest.raises(ValueError, match="seed must be >= 0"):
        cfg.validate()


def test_invalid_env_name_raises_error():
    """Verify that unapproved environment names fail validation."""
    cfg = AppConfig(env=EnvConfig(name="unknown_simulator"))
    with pytest.raises(ValueError, match="Invalid env.name"):
        cfg.validate()


def test_invalid_algorithm_raises_error():
    """Verify that unsupported algorithms fail validation."""
    cfg = AppConfig(algorithm=AlgorithmConfig(name="DQN"))
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        cfg.validate()


def test_invalid_timesteps_raises_error():
    """Verify that non-positive training timesteps fail validation."""
    cfg = AppConfig(training=TrainingConfig(total_timesteps=0))
    with pytest.raises(ValueError, match="total_timesteps must be positive"):
        cfg.validate()


def test_invalid_device_raises_error():
    """Verify that invalid compute device fails validation."""
    cfg = AppConfig(training=TrainingConfig(device="tpu"))
    with pytest.raises(ValueError, match="device must be one of"):
        cfg.validate()


def test_save_and_reload_config(tmp_path):
    """Verify that configuration can be serialized to YAML and reloaded."""
    original_cfg = AppConfig(seed=123)
    save_dest = tmp_path / "test_config.yaml"
    save_config(original_cfg, save_dest)

    assert save_dest.is_file()
    reloaded_cfg = load_config(save_dest)
    assert reloaded_cfg.seed == 123
    assert reloaded_cfg.env.name == "synthetic"
