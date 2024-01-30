"""
Functions for getting the best available device.
"""

import torch


def get_device() -> torch.device:
    """Return the most approriate available device"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")
