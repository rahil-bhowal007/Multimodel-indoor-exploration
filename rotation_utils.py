import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

ANGLE_EPS = 1e-3


def unit_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a numpy vector to unit length.
    """
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rodrigues' rotation formula: rotate around 'axis' by 'angle' (radians).
    """
    ax = unit_vector(axis)
    if abs(angle) <= ANGLE_EPS:
        return np.eye(3, dtype=ax.dtype)
    x, y, z = ax
    c, s, C = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    return np.array([
        [c + x*x*C,    x*y*C - z*s,  x*z*C + y*s],
        [y*x*C + z*s,  c + y*y*C,    y*z*C - x*s],
        [z*x*C - y*s,  z*y*C + x*s,  c + z*z*C   ],
    ], dtype=ax.dtype)


def rotation_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Minimal rotation matrix aligning v1 to v2.
    """
    u, w = unit_vector(v1), unit_vector(v2)
    axis = unit_vector(np.cross(u, w))
    angle = np.arccos(np.clip(np.dot(u, w), -1.0, 1.0))
    return rotation_matrix(axis, angle)


def align_camera(up_from: np.ndarray, look_from: np.ndarray,
                 up_to: np.ndarray, look_to: np.ndarray) -> np.ndarray:
    """
    Compute rotation to align camera coordinate frames using up and look vectors.
    """
    R1 = rotation_between(look_from, look_to)
    new_x = R1 @ np.array([1.0, 0.0, 0.0])
    target_x = unit_vector(np.cross(look_to, up_to))
    angle = np.arccos(np.clip(np.dot(new_x, target_x), -1.0, 1.0))
    if abs(angle) > ANGLE_EPS and abs(angle - np.pi) > ANGLE_EPS:
        flip = np.dot(look_to, np.cross(new_x, target_x))
        R2 = rotation_matrix(look_to, angle if flip > 0 else -angle)
    else:
        R2 = np.eye(3, dtype=R1.dtype)
    return R2 @ R1


def build_affine_grids(pose: Tensor, grid_size: tuple, dtype: torch.dtype):
    """
    Create rotation and translation grids for affine transformations.

    Args:
        pose: (B,3) tensor [x, y, theta_deg]
        grid_size: target shape (B, C, H, W)
        dtype: output grid dtype
    Returns:
        rot_grid, trans_grid: sampling grids for grid_sample
    """
    x, y, theta = pose[:,0], pose[:,1], pose[:,2] * np.pi/180.0
    cos_t, sin_t = theta.cos(), theta.sin()

    # Rotation matrix components
    rot_params = torch.stack([
        cos_t, -sin_t, torch.zeros_like(cos_t),
        sin_t,  cos_t, torch.zeros_like(cos_t)
    ], dim=1).view(-1,2,3)
    rot_grid = F.affine_grid(rot_params, grid_size, align_corners=False).to(dtype)

    # Translation matrix components
    trans_params = torch.stack([
        torch.ones_like(x), torch.zeros_like(x), x,
        torch.zeros_like(y), torch.ones_like(y), y
    ], dim=1).view(-1,2,3)
    trans_grid = F.affine_grid(trans_params, grid_size, align_corners=False).to(dtype)

    return rot_grid, trans_grid


def heading_angle_to_target(rel_pos: np.ndarray) -> float:
    """
    Compute heading angle (radians) to target from local frame.
    """
    forward = np.array([1.0, 0.0])
    vec = np.array([rel_pos[0], rel_pos[2]])
    if np.linalg.norm(vec) == 0:
        return 0.0
    dot = np.clip(np.dot(forward, vec)/np.linalg.norm(vec), -1.0, 1.0)
    angle = np.arccos(dot)
    return -angle if vec[1] < 0 else angle