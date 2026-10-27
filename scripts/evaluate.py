#!/usr/bin/env python
"""CLI entry point for evaluating a trained SYNAPSE ARENA racing policy."""

import argparse
import sys
from pathlib import Path

# Add src to sys.path so scripts can run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from synapse_arena.config.schema import load_config
from synapse_arena.evaluation.evaluator import run_evaluation
from synapse_arena.utils.logger import get_logger

logger = get_logger("synapse_arena.evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SYNAPSE ARENA racing policy.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/smoke_test/model.zip",
        help="Path to saved model checkpoint (.zip).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Optional override for number of evaluation episodes.",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        if args.episodes is not None:
            config.evaluation.episodes = args.episodes

        summary = run_evaluation(config, checkpoint_path=args.checkpoint)
        logger.info(
            "Evaluation completed successfully: %d episodes, Mean Return: %.2f",
            summary.num_episodes,
            summary.mean_return,
        )
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
