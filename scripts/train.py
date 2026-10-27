#!/usr/bin/env python
"""CLI entry point for training a SYNAPSE ARENA racing policy."""

import argparse
import sys
from pathlib import Path

# Add src to sys.path so scripts can run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from synapse_arena.config.schema import load_config
from synapse_arena.training.trainer import train_policy
from synapse_arena.utils.logger import get_logger

logger = get_logger("synapse_arena.train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a SYNAPSE ARENA racing policy.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml).",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        saved_model = train_policy(config)
        logger.info("Training complete. Artifact: %s", saved_model)
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
