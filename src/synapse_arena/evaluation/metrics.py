"""Evaluation metrics definitions and serialization."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np


@dataclass
class EpisodeMetric:
    """Metrics recorded for a single evaluation episode.

    Fields like lap_time and is_crash are recorded ONLY if exposed by the environment.
    Never fabricated.
    """
    episode_index: int
    episode_return: float
    episode_length: int
    lap_time: Optional[float] = None
    is_completed: Optional[bool] = None
    is_crash: Optional[bool] = None
    track: Optional[str] = None


@dataclass
class EvaluationSummary:
    """Aggregated evaluation metrics across multiple episodes."""
    num_episodes: int
    mean_return: float
    std_return: float
    mean_length: float
    std_length: float
    episodes: List[EpisodeMetric] = field(default_factory=list)

    @classmethod
    def from_episodes(cls, episodes: List[EpisodeMetric]) -> "EvaluationSummary":
        if not episodes:
            return cls(num_episodes=0, mean_return=0.0, std_return=0.0, mean_length=0.0, std_length=0.0)

        returns = [ep.episode_return for ep in episodes]
        lengths = [ep.episode_length for ep in episodes]

        return cls(
            num_episodes=len(episodes),
            mean_return=float(np.mean(returns)),
            std_return=float(np.std(returns)),
            mean_length=float(np.mean(lengths)),
            std_length=float(np.std(lengths)),
            episodes=episodes,
        )


def save_metrics_to_csv(summary: EvaluationSummary, csv_path: str | Path) -> Path:
    """Saves per-episode metrics and summary to a CSV file.

    Args:
        summary: Aggregated evaluation summary.
        csv_path: Destination path for the CSV.

    Returns:
        Path: Resolved path to the written CSV.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    records = [asdict(ep) for ep in summary.episodes]
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    return path
