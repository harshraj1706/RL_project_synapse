"""End-to-end integration smoke test for the SYNAPSE ARENA software pipeline.

Verifies:
1. Environment instantiation via factory.
2. PPO algorithm initialization.
3. Micro-training loop execution (64 steps).
4. Model checkpoint saving to disk.
5. Model reloading from disk.
6. Deterministic evaluation execution.
7. Metric generation and CSV output.
"""

from pathlib import Path
from synapse_arena.config.schema import (
    AppConfig,
    EnvConfig,
    AlgorithmConfig,
    TrainingConfig,
    EvaluationConfig,
    RewardConfig,
)
from synapse_arena.training.trainer import train_policy
from synapse_arena.evaluation.evaluator import run_evaluation


def test_end_to_end_synthetic_smoke_pipeline(tmp_path):
    """Executes a full micro-loop verifying software integration without consuming competition budget."""
    checkpoint_dir = tmp_path / "checkpoints"
    results_dir = tmp_path / "results"
    log_dir = tmp_path / "logs"

    config = AppConfig(
        seed=101,
        env=EnvConfig(name="synthetic"),
        algorithm=AlgorithmConfig(
            name="PPO",
            policy="MlpPolicy",
            learning_rate=0.0003,
            n_steps=64,
            batch_size=32,
        ),
        training=TrainingConfig(
            total_timesteps=64,
            checkpoint_dir=str(checkpoint_dir),
            log_dir=str(log_dir),
            save_freq=64,
            device="cpu",
        ),
        evaluation=EvaluationConfig(
            episodes=2,
            deterministic=True,
            results_dir=str(results_dir),
            save_csv=True,
        ),
        reward=RewardConfig(use_custom_shaping=False),
    )

    # 1. Train micro model
    saved_model_path = train_policy(config)
    assert saved_model_path.is_file(), f"Model file should exist at {saved_model_path}"
    assert (checkpoint_dir / "config.yaml").is_file(), "Saved config should exist in checkpoint dir"

    # 2. Evaluate saved model
    summary = run_evaluation(config, checkpoint_path=saved_model_path)
    assert summary.num_episodes == 2
    assert len(summary.episodes) == 2
    assert isinstance(summary.mean_return, float)

    # 3. Verify CSV output
    csv_file = results_dir / "eval_metrics.csv"
    assert csv_file.is_file(), f"CSV results file should exist at {csv_file}"
    content = csv_file.read_text(encoding="utf-8")
    assert "episode_index" in content
    assert "episode_return" in content
