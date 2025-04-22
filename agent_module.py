
import time
import torch
import torch.nn as nn


from categorical_map import CategoricalMap2D
from frontier_policy import FrontierExplorer
from matching import GoalMatcher
from memory import InstanceTracker

class Agent(nn.Module):
    def __init__(self, cfg):
        super(Agent, self).__init__()
        # semantic map builder
        self.map_builder = CategoricalMap2D(
            frame_h=cfg.env.frame_h,
            frame_w=cfg.env.frame_w,
            cam_h=cfg.env.cam_h,
            hfov=cfg.env.hfov,
            categories=cfg.map.num_cat,
            map_size=cfg.map.size,
            resolution=cfg.map.res,
            vision_range=cfg.map.vision_range,
            explore_radius=cfg.map.explore_rad,
            proximity_radius=cfg.map.prox_rad,
        )
        # frontier exploration policy
        self.explorer = FrontierExplorer(strategy=cfg.agent.explore_strat)
        # instance matcher and memory (optional)
        self.matcher = GoalMatcher(cfg.match)
        self.tracker = InstanceTracker(cfg.mem) if cfg.mem.enabled else None

    @property
    def update_steps(self):
        return self.explorer.update_interval

    def reset(self):
        if self.tracker:
            self.tracker.reset()

    def forward(self, obs_seq, pose_delta, done_flags, update_flags,
                cam_poses, init_loc_map, init_glob_map,
                init_loc_pose, init_glob_pose, init_bounds,
                init_origins, goal_labels=None, **kwargs):
        # Build/update semantic maps
        B, T = obs_seq.shape[:2]
        (feat_seq, final_loc, final_glob,
         loc_pose_seq, glob_pose_seq,
         bounds_seq, origin_seq) = self.map_builder(
            obs_seq, pose_delta, done_flags,
            update_flags, cam_poses,
            init_loc_map, init_glob_map,
            init_loc_pose, init_glob_pose,
            init_bounds, init_origins,
        )
        # Flatten for policy
        feats = feat_seq.view(-1, *feat_seq.shape[2:])
        if goal_labels is not None:
            goal_labels_flat = goal_labels.view(-1)
        else:
            goal_labels_flat = None

        # Default goal map and found flags
        goal_map, found = self.explorer(feats, goal_labels_flat)

        # Match or track instances if tracker available
        if self.tracker or self.matcher:
            goal_map, found = self.matcher.match(
                goal_map, found,
                final_loc, bounds_seq[0],
                tracker=self.tracker, **kwargs
            )

        # Reshape outputs
        goal_seq = goal_map.view(B, T, *goal_map.shape[-2:])
        found_seq = found.view(B, T)
        frontier = self.explorer.get_frontier(feats)
        frontier_seq = frontier.view(B, T, *frontier.shape[-2:])

        return (goal_seq, found_seq, frontier_seq,
                final_loc, final_glob,
                loc_pose_seq, glob_pose_seq,
                bounds_seq, origin_seq)