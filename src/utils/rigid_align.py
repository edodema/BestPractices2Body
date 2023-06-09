from typing import Tuple
import numpy as np
import torch

# from IPython import embed


def rigid_transform_3D(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions. # ?

    Args:
        A (np.ndarray): Source points.
        B (np.ndarray): Target points.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Rotation matrix and translation vector.
    """
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B)
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))
    t = -np.dot(R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return R, t


def rigid_align(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Align A to B.

    Args:
        A (torch.Tensor): Source points (ba, nb_frames, 36, 3).
        B (torch.Tensor): Target points (ba, nb_frames, 36, 3).

    Returns:
        torch.Tensor: Aligned points.
    """
    R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(R, np.transpose(A))) + t
    return A2


def rigid_align_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Align A to B.

    Args:
        A (torch.Tensor): Source points (ba, nb_frames, 36, 3).
        B (torch.Tensor): Target points (ba, nb_frames, 36, 3).

    Returns:
        torch.Tensor: Aligned points.
    """
    A = A.detach().cpu().numpy()
    B = B.detach().cpu().numpy()
    bz, nb_f, nb_kpts, _ = A.shape
    A2 = A
    assert nb_kpts == 18 or nb_kpts == 36
    for b in range(bz):
        for n in range(nb_f):
            # A2[b][n] = rigid_align(A[b][n],B[b][n])
            if nb_kpts == 18:
                A2[b][n] = rigid_align(A[b][n], B[b][n])
            elif nb_kpts == 36:
                A2[b][n][:18] = rigid_align(A[b][n][:18], B[b][n][:18])
                A2[b][n][18:] = rigid_align(A[b][n][18:], B[b][n][18:])

    return torch.from_numpy(A2).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
