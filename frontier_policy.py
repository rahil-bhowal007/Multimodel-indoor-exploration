import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
import skimage.morphology as morph

from constants import MapConstants as MC
from morphology_utils import binary_dilation

class FrontierExplorer(nn.Module):
    """
    Selects high-level goals by reaching mapped targets or exploring frontiers.
    """
    def __init__(self, strategy: str = 'seen_frontier'):
        super(FrontierExplorer, self).__init__()
        assert strategy in ['seen_frontier', 'been_close_to_frontier']
        self.strategy = strategy
        # kernels for dilation
        self.explore_kernel = torch.from_numpy(morph.disk(10)).unsqueeze(0).unsqueeze(0).float()
        self.border_kernel = torch.from_numpy(morph.disk(1)).unsqueeze(0).unsqueeze(0).float()

    @property
    def update_interval(self):
        return 1

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None):
        # labels unused here; call reach_then_explore instead
        raise NotImplementedError('Use reach_then_explore for goal selection')

    def reach_then_explore(self, features: torch.Tensor, category: torch.Tensor, visited_mask: torch.Tensor = None):
        """
        If category present in map features, set those cells as goal; otherwise flood frontier.
        """
        goal_map, found = self.reach_goal_if_present(features, category, visited_mask)
        goal_map = self.explore_if_missing(features, goal_map, found)
        return goal_map, found

    def reach_goal_if_present(self, features: torch.Tensor, category: torch.Tensor, visited_mask: torch.Tensor = None):
        B, C, H, W = features.shape
        device = features.device
        goal = torch.zeros((B, H, W), device=device)
        found = torch.zeros(B, dtype=torch.bool, device=device)
        for i in range(B):
            cat_feat = features[i, category[i] + 2 * MC.NON_SEM_CHANNELS]
            if visited_mask is not None:
                cat_feat = cat_feat * (1 - features[i, MC.BLACKLISTED_TARGETS_MAP])
            if (cat_feat == 1).any():
                goal[i] = (cat_feat == 1).float()
                found[i] = True
        return goal, found

    def get_frontier(self, features: torch.Tensor):
        """
        Compute weighted frontier map based on unexplored regions.
        """
        if self.strategy == 'seen_frontier':
            mask = (features[:, MC.EXPLORED_MAP] == 0).float()
        else:
            mask = (features[:, MC.BEEN_CLOSE_MAP] == 0).float()
        # dilate unexplored
        dig = 1 - binary_dilation(1 - mask, self.explore_kernel)
        border = binary_dilation(dig, self.border_kernel) - dig
        # unknown gain
        unk = (features[:, MC.EXPLORED_MAP] == 0).float()
        k = 15
        kernel = torch.ones((1,1,k,k), device=unk.device)
        gain = nn.functional.conv2d(unk.unsqueeze(1), kernel, padding=k//2).squeeze(1)
        weighted = border * gain
        return weighted

    def explore_if_missing(self, features: torch.Tensor, goal: torch.Tensor, found: torch.Tensor):
        B = features.shape[0]
        weighted = self.get_frontier(features)
        thr = 0.5
        mask = (weighted > thr).float()
        for i in range(B):
            if not found[i]:
                goal[i] = mask[i]
        return goal

    def cluster_filter(self, goal_map: torch.Tensor):
        """
        Keep only largest connected cluster of goal cells per batch.
        """
        B, H, W = goal_map.shape
        out = torch.zeros_like(goal_map)
        for i in range(B):
            gm = goal_map[i]
            if not gm.any():
                continue
            pts = np.array(gm.cpu().nonzero()).T
            labels = DBSCAN(eps=4, min_samples=1).fit_predict(pts)
            mode = stats.mode(labels).mode.item()
            keep = pts[labels == mode]
            om = torch.zeros((H,W), device=goal_map.device)
            om[keep[:,0], keep[:,1]] = 1
            out[i] = om
        return out
