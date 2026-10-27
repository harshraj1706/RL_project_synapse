"""Configuration schema and validation for SYNAPSE ARENA."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict
import yaml


@dataclass
class EnvConfig:
    """Environment configuration."""
    name: str = "synthetic"  # "synthetic" (for smoke test) or "official"
    track: str | None = None
    render_mode: str | None = None

    def validate(self) -> None:
        valid_envs = {"synthetic", "official"}
        if self.name not in valid_envs:
            raise ValueError(f"Invalid env.name '{self.name}'. Must be one of {valid_envs}.")


@dataclass
class AlgorithmConfig:
    """RL Algorithm hyperparameters."""
    name: str = "PPO"
    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    n_steps: int = 64
    batch_size: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0

    def validate(self) -> None:
        supported_algos = {"PPO"}
        if self.name.upper() not in supported_algos:
            raise ValueError(f"Unsupported algorithm '{self.name}'. Supported candidate: {supported_algos}.")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}.")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be > 0, got {self.n_steps}.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}.")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"gamma must be between 0 and 1, got {self.gamma}.")


@dataclass
class TrainingConfig:
    """Training runner parameters."""
    total_timesteps: int = 64
    checkpoint_dir: str = "checkpoints/smoke_test"
    log_dir: str = "logs/smoke_test"
    save_freq: int = 64
    device: str = "auto"
    resume_checkpoint: str | None = None

    def validate(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be positive, got {self.total_timesteps}.")
        valid_devices = {"auto", "cpu", "cuda"}
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}, got '{self.device}'.")


@dataclass
class EvaluationConfig:
    """Evaluation runner parameters."""
    episodes: int = 2
    deterministic: bool = True
    results_dir: str = "eval_results/smoke_test"
    save_csv: bool = True

    def validate(self) -> None:
        if self.episodes <= 0:
            raise ValueError(f"episodes must be positive, got {self.episodes}.")


@dataclass
class RewardConfig:
    """Reward shaping configuration."""
    use_custom_shaping: bool = False
    speed_weight: float = 0.0
    collision_penalty: float = 0.0

    def validate(self) -> None:
        pass


@dataclass
class AppConfig:
    """Root configuration holding all sub-configurations."""
    seed: int = 42
    env: EnvConfig = field(default_factory=EnvConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}.")
        self.env.validate()
        self.algorithm.validate()
        self.training.validate()
        self.evaluation.validate()
        self.reward.validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(config_path: str | Path) -> AppConfig:
    """Loads and validates a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        AppConfig: Validated application configuration.

    Raises:
        FileNotFoundError: If configuration file does not exist.
        ValueError: If YAML is invalid or values fail validation.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML file at {path}: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise ValueError(f"Configuration root in {path} must be a dictionary/mapping.")

    env_cfg = EnvConfig(**raw_data.get("env", {}))
    algo_cfg = AlgorithmConfig(**raw_data.get("algorithm", {}))
    train_cfg = TrainingConfig(**raw_data.get("training", {}))
    eval_cfg = EvaluationConfig(**raw_data.get("evaluation", {}))
    reward_cfg = RewardConfig(**raw_data.get("reward", {}))
    seed = raw_data.get("seed", 42)

    app_cfg = AppConfig(
        seed=seed,
        env=env_cfg,
        algorithm=algo_cfg,
        training=train_cfg,
        evaluation=eval_cfg,
        reward=reward_cfg,
    )
    app_cfg.validate()
    return app_cfg


def save_config(config: AppConfig, save_path: str | Path) -> None:
    """Saves an AppConfig to YAML format.

    Args:
        config: AppConfig object to save.
        save_path: Destination path for YAML file.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
