import torch
from torch import Tensor

class MapSizeParameters:
    """
    Holds map sizing parameters for converting between centimeters and grid cells.
    """
    def __init__(self, resolution: int, map_size_cm: int, global_downscaling: int):
        self.resolution = resolution
        self.global_map_size_cm = map_size_cm
        self.global_downscaling = global_downscaling
        # Local map covers a fraction of the global map
        self.local_map_size_cm = map_size_cm // global_downscaling
        self.global_map_size = map_size_cm // resolution
        self.local_map_size = self.local_map_size_cm // resolution


def init_map_and_pose_for_env(
    env_idx: int,
    local_map: Tensor,
    global_map: Tensor,
    local_pose: Tensor,
    global_pose: Tensor,
    bounds: Tensor,
    origins: Tensor,
    params: MapSizeParameters,
) -> None:
    """
    Reset maps and poses for a specific environment index.
    """
    # Place agent at center of global map
    global_pose[env_idx].zero_()
    half = params.global_map_size_cm / 100.0 / 2.0
    global_pose[env_idx, :2] = half

    # Clear global map and mark initial agent location
    global_map[env_idx].zero_()
    x, y = (global_pose[env_idx, :2] * 100 / params.resolution).int()
    global_map[env_idx, 2:4, y-1:y+2, x-1:x+2] = 1.0

    # Recenter local map based on new global pose
    recenter_local_map_and_pose_for_env(
        env_idx, local_map, global_map, local_pose, global_pose, bounds, origins, params
    )


def recenter_local_map_and_pose_for_env(
    env_idx: int,
    local_map: Tensor,
    global_map: Tensor,
    local_pose: Tensor,
    global_pose: Tensor,
    bounds: Tensor,
    origins: Tensor,
    params: MapSizeParameters,
) -> None:
    """
    Update local map window and adjust local pose accordingly.
    """
    # Compute integer cell location in global grid
    global_cell = (global_pose[env_idx, :2] * 100 / params.resolution).int()
    # Determine local window boundaries
    bounds[env_idx] = get_local_map_boundaries(global_cell, params)
    y1, y2, x1, x2 = bounds[env_idx]
    # Compute origin offset for local pose
    origins[env_idx] = torch.tensor([
        x1 * params.resolution / 100.0,
        y1 * params.resolution / 100.0,
        0.0,
    ], device=origins.device)
    # Extract local map slice from global
    local_map[env_idx] = global_map[env_idx, :, y1:y2, x1:x2]
    # Adjust local pose relative to origin
    local_pose[env_idx] = global_pose[env_idx] - origins[env_idx]


def get_local_map_boundaries(
    global_cell: Tensor,
    params: MapSizeParameters,
) -> Tensor:
    """
    Compute the [y1, y2, x1, x2] boundaries of the local map window.
    """
    device, dtype = global_cell.device, global_cell.dtype
    y, x = global_cell.tolist()
    size = params.local_map_size
    half = size // 2

    # Compute window edges with clamping
    y1 = max(0, y - half)
    x1 = max(0, x - half)
    y2 = min(params.global_map_size, y1 + size)
    x2 = min(params.global_map_size, x1 + size)

    # Adjust if window falls off global map edges
    if y2 - y1 < size:
        y1 = y2 - size
    if x2 - x1 < size:
        x1 = x2 - size

    return torch.tensor([y1, y2, x1, x2], device=device, dtype=dtype)
