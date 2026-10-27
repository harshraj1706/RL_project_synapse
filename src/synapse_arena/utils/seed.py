"""Deterministic seed utilities for reproducibility across libraries."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for Python random, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Guarantee deterministic algorithms where available
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
