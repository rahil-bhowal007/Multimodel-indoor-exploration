from typing import Any, Dict, List, Optional, Tuple

import clip
import numpy as np
import torch
from torchvision.transforms import ToPILImage

from mapping.semantic.constants import MapConstants as MC
from mapping.semantic.instance_tracking_modules import InstanceMemory

# Minimum size thresholds for valid crops
MIN_PIXELS = 1000
MIN_EDGE = 15


class GoatMatching:
    """
    Matches image- or text-specified goals against instance views
    in the current frame or memory, and localizes the best match.
    """

    def __init__(
        self,
        device: int,
        score_type: str,
        num_categories: int,
        cfg: Dict[str, Any],
        vis_dir: str,
        enable_logging: bool,
        instance_memory: InstanceMemory,
    ):
        # Validate score type
        assert score_type in ("confidence_sum", "match_count"), "Invalid score_type"
        self.device = device
        self.score_type = score_type
        self.num_categories = num_categories
        # Configuration for thresholding
        self.score_function = cfg.get("score_function", score_type)
        self.score_thresh = cfg.get("score_thresh", 0.0)
        self.use_past_pose = cfg.get("goto_past_pose", False)
        self.memory = instance_memory

        # Load CLIP for language-image matching
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)

    def _collect_candidates(
        self,
        from_memory: bool,
        step: int,
        image_goal: Optional[np.ndarray] = None,
        text_goal: Optional[str] = None,
        restrict_categories: Optional[List[int]] = None,
        use_full_frame: bool = False,
    ) -> Tuple[List[np.ndarray], List[int], List[int]]:
        """
        Gather image crops from current frame (new views) or full memory.
        Returns (views, counts, instance_ids).
        """
        views, counts, ids = [], [], []
        items = self.memory.instance_views[0] if from_memory else self.memory.unprocessed_views[0]
        for inst_id, inst in items.items():
            if restrict_categories and inst.category_id not in restrict_categories:
                continue
            crop_list = inst.instance_views if from_memory else [inst]
            valid_views = []
            for v in crop_list:
                h, w = v.cropped_image.shape[:2]
                if h * w < MIN_PIXELS or min(h, w) < MIN_EDGE:
                    continue
                img = (self.memory.images[0][v.timestep].cpu().numpy()
                       if use_full_frame else v.cropped_image)
                valid_views.append(img)
            if valid_views:
                views.extend(valid_views)
                counts.append(len(valid_views))
                ids.append(inst_id)
        return views, counts, ids

    def _compute_matches(
        self,
        views: List[np.ndarray],
        match_fn,
        step: int,
        image_goal: Optional[np.ndarray],
        text_goal: Optional[str],
        **kwargs
    ) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        Use SuperGlue or CLIP to match each view to the goal.
        Returns (keypoints, matches, confidences).
        """
        kp, matches, conf = [], [], []
        if image_goal is not None:
            kp0, m, c = match_fn(
                views,
                goal_image=image_goal,
                goal_image_keypoints=kwargs.get("goal_kp"),
                step=1000 * step,
            )
            kp.extend(kp0)
            matches = m
            conf = c
        elif text_goal is not None:
            m, c = match_fn(views, language_goal=text_goal)
            kp = [[] for _ in views]
            matches = m
            conf = c
        return kp, np.array(matches), np.array(conf)

    def superglue(
        self,
        goal_map: torch.Tensor,
        found_goal: torch.Tensor,
        local_map: torch.Tensor,
        matches: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Goal detection and localization via SuperGlue-like scoring.
        Updates goal_map and found_goal based on match scores.
        """
        score_func = self.score_function
        assert score_func in ("confidence_sum", "match_count"), "Invalid score_function"
        for e in range(confidence.shape[0]):
            # skip if no semantic channel present
            if not local_map[e, -1].any().item():
                continue
            if score_func == "confidence_sum":
                score = confidence[e][matches[e] != -1].sum()
            else:
                score = (matches[e] != -1).sum()
            if score < self.score_thresh:
                continue
            found_goal[e] = True
            # set the goal to the last semantic channel
            goal_map[e, 0] = local_map[e, -1]
        return goal_map, found_goal