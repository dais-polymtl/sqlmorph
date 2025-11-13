import os

import numpy as np


def ensure_directory(path):
    """Ensure a directory exists, creating it if necessary."""
    if not os.path.exists(path):
        os.makedirs(path)


def set_random_seed(seed=42):
    """Set the random seed for reproducibility."""
    np.random.seed(seed)
