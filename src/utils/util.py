from typing import Tuple
import torch
import numpy as np


def get_dct_matrix(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get DCT matrix.

    Args:
        N (int): DCT matrix size.

    Returns:
        Tuple[np.ndarray, np.ndarray]: DCT matrix and inverse DCT matrix.
    """
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


def update_lr_multistep(
    nb_iter: int, optimizer: torch.optim.Optimizer
) -> Tuple[torch.optim.Optimizer, float]:
    """Update learning rate.

    Args:
        nb_iter (int): Iteration number.
        optimizer (torch.optim.Optimizer): Optimizer.

    Returns:
        Tuple[torch.optim.Optimizer, float]: Optimizer and current learning rate.
    """
    if nb_iter > 30000:
        current_lr = 1e-5
    else:
        current_lr = 3e-4

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm
