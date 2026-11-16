"""Reproducibility: penyetelan seed untuk torch, numpy, random, dan CUDA."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from configs.base_config import RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED, deterministic: bool = True) -> None:
    """Set seed untuk reproducibility di torch, numpy, random, dan CUDA.

    Args:
        seed: Nilai seed. Default mengikuti ``RANDOM_SEED`` dari config.
        deterministic: Bila ``True``, paksa cuDNN deterministik (lebih lambat,
            tetapi hasil konsisten antar-run).

    Example:
        >>> set_seed(42)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
