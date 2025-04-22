
# Fast Marching Method path planner using skfmm

import os
from typing import List, Tuple

import cv2
import numpy as np
import skfmm
from numpy import ma

class FMMPlanner:
    """
    Fast Marching Method planner for selecting short-term goals.
    """
    def __init__(
        self,
        traversible: np.ndarray,
        scale: int = 1,
        step_size: int = 5,
        goal_tolerance: float = 2.0,
        vis_dir: str = "planner_vis",
        visualize: bool = False,
        save_images: bool = False,
        debug: bool = False,
    ):
        self.scale = scale
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        self.visualize = visualize
        self.save_images = save_images
        self.debug = debug
        self.vis_dir = vis_dir
        os.makedirs(self.vis_dir, exist_ok=True)

        if scale != 1:
            h, w = traversible.shape
            self.traversible = cv2.resize(
                traversible, (w // scale, h // scale),
                interpolation=cv2.INTER_NEAREST
            ).round().astype(traversible.dtype)
        else:
            self.traversible = traversible.copy()

        self.du = int(self.step_size / scale)
        self.fmm_dist = None

    def set_multi_goal(
        self,
        goal_map: np.ndarray,
        timestep: int = 0,
        prev_dist: np.ndarray = None,
        update_freq: int = 1,
        downsample: float = 1.0,
    ) -> np.ndarray:
        """
        Compute distance map from goal_map to all traversible cells.
        """
        tv = self.traversible
        h, w = tv.shape
        if downsample > 1.0:
            tv = cv2.resize(tv, (int(w/downsample), int(h/downsample)), interpolation=cv2.INTER_NEAREST)
            gm = cv2.resize(goal_map, (int(w/downsample), int(h/downsample)), interpolation=cv2.INTER_NEAREST)
            if gm.sum() == 0:
                gm = cv2.dilate(goal_map, np.ones((2,2), np.uint8), iterations=1)
                gm = cv2.resize(gm, (int(w/downsample), int(h/downsample)), interpolation=cv2.INTER_NEAREST)
        else:
            gm = goal_map

        traversible_ma = ma.masked_values(tv, 0)
        traversible_ma[gm == 1] = 0

        if prev_dist is None or (timestep % update_freq == 0):
            dist = skfmm.distance(traversible_ma, dx=1 * downsample)
            dist = ma.filled(dist, np.max(dist) + 1)
        else:
            dist = prev_dist

        if downsample > 1.0:
            dist = cv2.resize(dist, (w, h), interpolation=cv2.INTER_NEAREST)

        if (self.visualize or self.save_images) and timestep > 0:
            vis = np.zeros((h, w*3))
            vis[:, :w] = np.flipud(tv)
            vis[:, w:2*w] = np.flipud(gm)
            vis[:, 2*w:] = np.flipud(dist / dist.max())
            if self.visualize:
                cv2.imshow("FMMPlanner", vis)
                cv2.waitKey(1)
            if self.save_images:
                cv2.imwrite(os.path.join(self.vis_dir, f"fmm_{timestep}.png"), (vis*255).astype(np.uint8))

        self.fmm_dist = dist
        return dist

    def get_short_term_goal(
        self,
        position: Tuple[float, float],
        continuous: bool = True,
        timestep: int = 0
    ) -> Tuple[float, float, bool, bool]:
        """
        Select next waypoint within 'step_size' based on distance gradients.
        Returns (x, y, replan, stop).
        """
        sx, sy = position
        sx_s = sx / self.scale
        sy_s = sy / self.scale
        size = self.du*2 + 1

        ext = np.pad(self.fmm_dist, self.du,
                     mode='constant', constant_values=self.fmm_dist.max()*10)
        subset = ext[
            int(sx_s):int(sx_s)+size,
            int(sy_s):int(sy_s)+size
        ]
        if subset.shape != (size, size):
            raise ValueError(f"Unexpected subset shape {subset.shape}")

        mask = self._build_mask(sx_s, sy_s)
        masked = subset.copy()
        masked[mask == 0] = subset.max()

        center_val = masked[self.du, self.du]
        stop = center_val < self.goal_tolerance

        normed = masked - center_val
        ratio = normed / mask
        ratio[ratio < -1.5] = subset.max()
        idx = np.unravel_index(np.argmin(ratio), ratio.shape)
        nx = (idx[0] + int(sx_s) - self.du) * self.scale
        ny = (idx[1] + int(sy_s) - self.du) * self.scale
        replan = ratio[idx] > -1e-6
        return nx, ny, replan, stop

    @staticmethod
    def _build_mask(sx: float, sy: float, scale: float = 1.0, step: float = 5.0) -> np.ndarray:
        size = int(step/scale)*2 + 1
        mask = np.zeros((size, size))
        center = size//2
        for i in range(size):
            for j in range(size):
                d2 = ((i+0.5-center-sx)**2 + (j+0.5-center-sy)**2)
                if d2 <= step**2:
                    mask[i,j] = max(1e-10, np.sqrt(d2))
        mask[center, center] = max(mask[center, center], 1e-10)
        return mask
