"""Run-level reproducibility helpers."""

import os
import random

import numpy as np
import torch


def seed_everything(seed=0):
    """Seed training RNGs and enable inexpensive deterministic Torch settings.

    Schedule generation deliberately does not use any of these global RNGs; the
    task scheduler owns an independent ``random.Random`` instance.
    """

    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass
    return seed
