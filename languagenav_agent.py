import time
import torch.nn as nn
from categorical_map import CategoricalMap2D
from frontier_policy import FrontierExplorer

class NavigationAgent(nn.Module):
    def __init__(self, cfg):
        super(NavigationAgent, self).__init__()
        # map processor
        self.map_module = CategoricalMap2D(
            height=cfg.environment.frame_height,
            width=cfg.environment.frame_width,
            camera_height=cfg.environment.camera_height,
            hfov=cfg.environment.hfov,
            num_classes=cfg.map.num_categories,
            map_extent=cfg.map.size,
            resolution=cfg.map.resolution,
            vision_range=cfg.map.vision_range,
            explore_radius=cfg.map.explore_radius,
            proximity_radius=cfg.map.proximity_radius,
        )
        # exploration policy
        self.explorer = FrontierExplorer(strategy=cfg.agent.exploration_strategy)

    @property
    def update_interval(self):
        return self.explorer.update_steps

    def forward(self,
                observations,
                pose_deltas,
                done_flags,
                update_global_flags,
                camera_matrices,
                init_local_map,
                init_global_map,
                init_local_pose,
                init_global_pose,
                init_bounds,
                init_origins,
                object_goals=None,
                reject_seen=False,
                blacklist=False):
        # Update semantic maps
        batch, seq_len = observations.shape[:2]
        (map_feats, final_local, final_global,
         local_pose_seq, global_pose_seq,
         bounds_seq, origin_seq) = self.map_module(
            observations, pose_deltas, done_flags,
            update_global_flags, camera_matrices,
            init_local_map, init_global_map,
            init_local_pose, init_global_pose,
            init_bounds, init_origins,
            blacklist
        )
        # Flatten for policy
        feats = map_feats.view(-1, *map_feats.shape[2:])
        if object_goals is not None:
            object_goals = object_goals.view(-1)

        # Compute goal distribution
        goal_map, found = self.explorer(feats,
                                        object_goals,
                                        reject_seen)
        # Restore batch and sequence dims
        goal_seq = goal_map.view(batch, seq_len, *goal_map.shape[-2:])
        found_seq = found.view(batch, seq_len)

        # Frontier mask
        frontier = self.explorer.get_frontier(feats)
        frontier_seq = frontier.view(batch, seq_len, *frontier.shape[-2:])

        return (goal_seq, found_seq, frontier_seq,
                final_local, final_global,
                local_pose_seq, global_pose_seq,
                bounds_seq, origin_seq)
