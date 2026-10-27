"""Evaluation runner for trained racing policies."""

from pathlib import Path
from typing import Any
import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm
from synapse_arena.config.schema import AppConfig
from synapse_arena.envs.factory import make_racing_env
from synapse_arena.rewards.shaping import RewardShaper
from synapse_arena.evaluation.metrics import (
    EpisodeMetric,
    EvaluationSummary,
    save_metrics_to_csv,
)
from synapse_arena.utils.logger import get_logger
from synapse_arena.utils.seed import set_seed

logger = get_logger(__name__)


def evaluate_policy(
    model: BaseAlgorithm,
    env: gym.Env,
    num_episodes: int = 2,
    deterministic: bool = True,
) -> EvaluationSummary:
    """Runs deterministic evaluation of a policy over a specified number of episodes.

    Args:
        model: Trained policy model supporting .predict(obs, deterministic=...).
        env: Target evaluation environment.
        num_episodes: Number of complete episodes to evaluate.
        deterministic: Whether to use deterministic action selection.

    Returns:
        EvaluationSummary: Aggregated statistics and per-episode metrics.
    """
    episodes: list[EpisodeMetric] = []

    for ep_idx in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_return = 0.0
        step_count = 0
        last_info = info

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            step_count += 1
            last_info = info
            done = bool(terminated or truncated)

        # Extract only confirmed metrics from environment info
        lap_time = float(last_info["lap_time"]) if "lap_time" in last_info else None
        is_completed = bool(last_info["is_completed"]) if "is_completed" in last_info else None
        is_crash = bool(last_info["is_crash"]) if "is_crash" in last_info else None
        track = str(last_info.get("track", "unknown"))

        ep_metric = EpisodeMetric(
            episode_index=ep_idx + 1,
            episode_return=total_return,
            episode_length=step_count,
            lap_time=lap_time,
            is_completed=is_completed,
            is_crash=is_crash,
            track=track,
        )
        episodes.append(ep_metric)
        logger.info(
            "Episode %d/%d: Return=%.2f, Length=%d steps, Track=%s",
            ep_idx + 1,
            num_episodes,
            total_return,
            step_count,
            track,
        )

    summary = EvaluationSummary.from_episodes(episodes)
    logger.info(
        "Evaluation Complete: Mean Return=%.2f +/- %.2f | Mean Length=%.1f +/- %.1f over %d episodes",
        summary.mean_return,
        summary.std_return,
        summary.mean_length,
        summary.std_length,
        summary.num_episodes,
    )
    return summary


def run_evaluation(config: AppConfig, checkpoint_path: str | Path) -> EvaluationSummary:
    """Loads a saved checkpoint and runs evaluation using configuration parameters.

    Args:
        config: Application configuration.
        checkpoint_path: Path to the saved model (.zip).

    Returns:
        EvaluationSummary: Aggregated evaluation results.
    """
    from stable_baselines3 import PPO

    set_seed(config.seed)
    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint model not found at {ckpt.resolve()}")

    logger.info("Loading model checkpoint from %s", ckpt)
    model = PPO.load(ckpt, device=config.training.device)

    raw_env = make_racing_env(config.env, seed=config.seed)
    env = RewardShaper(raw_env, config.reward)

    summary = evaluate_policy(
        model=model,
        env=env,
        num_episodes=config.evaluation.episodes,
        deterministic=config.evaluation.deterministic,
    )

    if config.evaluation.save_csv:
        results_dir = Path(config.evaluation.results_dir)
        csv_path = results_dir / "eval_metrics.csv"
        saved_path = save_metrics_to_csv(summary, csv_path)
        logger.info("Evaluation metrics successfully written to: %s", saved_path)

    env.close()
    return summary
