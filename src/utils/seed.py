"""Reproducibility utilities for JointGuardian.

Set deterministic seeds for ``random``, ``numpy``, and (optionally)
``torch`` so that experiments are reproducible across runs.  ``torch``
is imported lazily — if it is not installed, only ``random`` and
``numpy`` are seeded.
"""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for all RNG backends used by the project.

    ``torch`` seeding is attempted but silently skipped if torch is not
    installed, so this function can be called from headless data-
    generation scripts that do not depend on PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
