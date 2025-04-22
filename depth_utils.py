import torch
import numpy as np
from typing import Tuple

from constants_utils import MIN_DEPTH_VALUE, MAX_DEPTH_VALUE
from rotation_utils import get_r_matrix


def valid_depth_mask(depth: np.ndarray) -> np.ndarray:
    """
    Return mask of valid depth pixels (not equal to min or max placeholders).
    """
    return (depth != MIN_DEPTH_VALUE) & (depth != MAX_DEPTH_VALUE)


def get_camera_matrix(width: int, height: int, fov: float):
    """
    Compute camera intrinsics from image size and horizontal field of view.
    Returns namespace with xc, zc, f.
    """
    xc = (width - 1) / 2.0
    zc = (height - 1) / 2.0
    f = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
    return type('Cam', (), {'xc': xc, 'zc': zc, 'f': f})()


def project_depth_to_point_cloud(
    depth: torch.Tensor,
    cam: object,
    device: torch.device,
    scale: int = 1
) -> torch.Tensor:
    """
    Projects depth image to 3D point cloud tensor (..., H, W, 3).
    """
    B, H, W = depth.shape
    ys = torch.arange(H-1, -1, -1, device=device).view(1, H, 1).expand(B, H, W)
    xs = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W)
    if scale > 1:
        ys = ys[:, ::scale, ::scale]
        xs = xs[:, ::scale, ::scale]
        depth = depth[:, ::scale, ::scale]
    X = (xs - cam.xc) * depth / cam.f
    Z = (ys - cam.zc) * depth / cam.f
    return torch.stack((X, depth, Z), dim=-1)


def transform_camera_to_base_frame(
    pc: torch.Tensor,
    sensor_height: float,
    elevation_deg: float,
    device: torch.device
) -> torch.Tensor:
    """
    Rotate point cloud about X-axis by elevation and raise by sensor_height.
    """
    R = get_r_matrix([1,0,0], np.deg2rad(elevation_deg))
    R = torch.from_numpy(R).float().to(device)
    B = pc.shape[0]
    flat = pc.reshape(-1,3)
    rotated = flat @ R.T
    rotated = rotated.view(pc.shape)
    rotated[...,2] += sensor_height
    return rotated


def transform_base_pose(
    pc: torch.Tensor,
    pose: Tuple[float,float,float],
    device: torch.device
) -> torch.Tensor:
    """
    Rotate point cloud about Z-axis by pose[2]-pi/2 and translate by pose[:2].
    """
    angle = pose[2] - np.pi/2
    R = get_r_matrix([0,0,1], angle)
    R = torch.from_numpy(R).float().to(device)
    flat = pc.reshape(-1,3)
    rotated = flat @ R.T
    rotated = rotated.view(pc.shape)
    rotated[...,0] += pose[0]
    rotated[...,1] += pose[1]
    return rotated


def splat_features(
    init_grid: torch.Tensor,
    features: torch.Tensor,
    coords: torch.Tensor
) -> torch.Tensor:
    """
    Scatter feature vectors into a grid using bilinear weights.
    init_grid: (B, F, *dims)
    features: (B, F, N)
    coords: (B, D, N) in normalized [-1,1]
    """
    B, F = init_grid.shape[:2]
    dims = init_grid.shape[2:]
    flat = init_grid.view(B, F, -1)
    D = len(dims)
    # compute positions and weights per dim
    pos_list, wt_list = [], []
    for d, size in enumerate(dims):
        c = coords[:, d] * size/2 + size/2
        flo = torch.floor(c)
        w0 = 1 - (c - flo)
        w1 = c - flo
        pos_list.append((flo.long(), (flo+1).long()))
        wt_list.append((w0, w1))
    # iterate corners
    for idx in range(2**D):
        mask = ''
        indices = []
        weights = None
        for d in range(D):
            bit = (idx>>d)&1
            pos = pos_list[d][bit]
            wt = wt_list[d][bit]
            indices.append(pos)
            weights = wt if weights is None else weights*wt
        # compute linear index
        lin_idx = indices[0]
        stride=1
        for d in range(1,D):
            stride *= dims[d-1]
            lin_idx += indices[d]*stride
        lin_idx = lin_idx.unsqueeze(1).expand(B,F,-1)
        vals = features * weights.unsqueeze(1)
        flat.scatter_add_(2, lin_idx, vals)
    return flat.view(init_grid.shape)