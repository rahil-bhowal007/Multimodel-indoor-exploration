from collections import defaultdict
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, IntTensor
from torch.nn import functional as F
import skimage.morphology as morph

from map_utils import MapSizeParameters, init_map_and_pose_for_env, recenter_local_map_and_pose_for_env
from depth_utils import get_camera_matrix, get_point_cloud_from_depth, transform_camera_to_base, transform_pose
from pose_utils import get_new_pose_batch
from rotation_utils import get_grid
from constants import MapConstants as MC
from memory import InstanceTracker

class CategoricalMap2D(nn.Module):
    """
    Stateless module for building and updating a 2D semantic map with separate channels
    for obstacles, explored areas, agent location, and semantic categories.
    """
    def __init__(
        self,
        frame_h: int,
        frame_w: int,
        cam_h: float,
        hfov: float,
        categories: int,
        map_size: int,
        resolution: int,
        vision_range: int,
        explore_radius: int,
        proximity_radius: int,
        downscale: int=4,
        thresholds: dict=None,
        record_instances: bool=False,
        max_instances: int=0,
        instance_tracker: Optional[InstanceTracker]=None,
        exploration_mode: str='default',
    ):
        super().__init__()
        self.screen_h = frame_h
        self.screen_w = frame_w
        self.camera_matrix = get_camera_matrix(frame_w, frame_h, hfov)
        self.num_categories = categories
        self.map_params = MapSizeParameters(resolution, map_size, downscale)
        self.vision_range = vision_range
        self.explore_radius = explore_radius
        self.proximity_radius = proximity_radius
        self.thresholds = thresholds or {}
        self.record_instances = record_instances
        self.max_instances = max_instances
        self.instance_tracker = instance_tracker
        self.exploration_mode = exploration_mode

    @torch.no_grad()
    def forward(
        self,
        obs_seq: Tensor,
        pose_delta: Tensor,
        done_flags: Tensor,
        update_flags: Tensor,
        cam_poses: Tensor,
        init_local_map: Tensor,
        init_global_map: Tensor,
        init_local_pose: Tensor,
        init_global_pose: Tensor,
        init_bounds: Tensor,
        init_origins: Tensor,
        obstacle_locs: Optional[Tensor]=None,
        free_locs: Optional[Tensor]=None,
        blacklist: bool=False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, IntTensor, Tensor]:
        # Initialize sequences
        B, T = obs_seq.shape[:2]
        local_map = init_local_map.clone()
        global_map = init_global_map.clone()
        local_pose = init_local_pose.clone()
        global_pose = init_global_pose.clone()
        bounds = init_bounds.clone()
        origins = init_origins.clone()
        # Calculate feature channels
        feat_ch = 2 * MC.NON_SEM_CHANNELS + self.num_categories
        seq_feats = torch.zeros(B, T, feat_ch, *local_map.shape[-2:], device=obs_seq.device)
        seq_loc_pose = torch.zeros(B, T, 3, device=obs_seq.device)
        seq_glob_pose = torch.zeros(B, T, 3, device=obs_seq.device)
        seq_bounds = torch.zeros(B, T, 4, dtype=torch.int32, device=obs_seq.device)
        seq_origins = torch.zeros(B, T, 3, device=obs_seq.device)

        for t in range(T):
            # Reset per-environment maps on done
            for e in range(B):
                if done_flags[e, t]:
                    init_map_and_pose_for_env(e, local_map, global_map,
                                              local_pose, global_pose,
                                              bounds, origins,
                                              self.map_params)
            # Update local map and pose
            current_obs = obs_seq[:, t]
            current_delta = pose_delta[:, t]
            local_map, local_pose = self._update_local_map(
                current_obs, current_delta, local_map,
                local_pose, cam_poses[:, t], origins, bounds,
                obstacle_locs[:, t] if obstacle_locs is not None else None,
                free_locs[:, t] if free_locs is not None else None,
                blacklist)
            # Update global if flagged
            for e in range(B):
                if update_flags[e, t]:
                    self._update_global_env(e, local_map, global_map,
                                            local_pose, global_pose,
                                            bounds[e], origins[e])
            # Store sequences
            seq_loc_pose[:, t] = local_pose
            seq_glob_pose[:, t] = global_pose
            seq_bounds[:, t] = bounds
            seq_origins[:, t] = origins
            seq_feats[:, t] = self._get_map_features(local_map, global_map)
        return seq_feats, local_map, global_map, seq_loc_pose, seq_glob_pose, seq_bounds, seq_origins


