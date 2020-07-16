import numpy as np
import cv2
import mmcv

import torch

from ..builder import BBOX_ASSIGNERS
from .base_assigner import BaseAssigner
from .assign_result import AssignResult


@BBOX_ASSIGNERS.register_module()
class PointCTAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each point.

    Each proposals will be assigned with `0`, or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    """
    def assign(self, points, gt_bboxes, gt_contours, sizes):
        """Assign gt to bboxes.

        This method assign a gt bbox to every point, each bbox
        will be assigned with  0, or a positive number.
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every points to 0
        2. for each gt box, we find the k most closest points to the
            box center and assign the gt bbox to those points, we also record
            the minimum distance from each point to the closest gt box. When we
            assign the bbox to the points, we check whether its distance to the
            points is closest.

        Args:
            points (Tensor): points to be assigned, shape(n, 3) while last
                dimension stands for (x, y, stride).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_points = gt_contours.shape[0], points.shape[0]

        if num_points == 0 or num_gts == 0:
            # stores the assigned gt heatmap of each point
            assigned_gt_ct = points.new_ones((num_points,), dtype=torch.long)
            # stores the assigned gt dist (to this point) of each point
            assigned_gt_offsets = points.new_zeros((num_points, 2), dtype=torch.float32)
            pos_inds = torch.nonzero(assigned_gt_ct == 0, as_tuple=False).squeeze(-1).unique()
            neg_inds = torch.nonzero(assigned_gt_ct > 0, as_tuple=False).squeeze(-1).unique()

            return assigned_gt_ct, assigned_gt_offsets, pos_inds, neg_inds

        points_range = torch.arange(num_points)
        points_xy = points[:, :2]
        points_stride = points[:, 2]
        points_lvl = torch.log2(points_stride).int()  # [3...,4...,5...,6...,7...]
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()

        # stores the assigned gt heatmap of each point
        assigned_gt_ct = points.new_ones((num_points,), dtype=torch.long)
        # stores the assigned gt dist (to this point) of each point
        assigned_gt_offsets = points.new_zeros((num_points, 2), dtype=torch.float32)

        lvls = torch.arange(lvl_min, lvl_max + 1, dtype=points_lvl.dtype, device=points_lvl.device)
        for gt_lvl in lvls:
            lvl_size = sizes[gt_lvl - 3]

            lvl_idx = gt_lvl == points_lvl
            points_index = points_range[lvl_idx]

            # generate contours
            downscale_factor = torch.pow(2, gt_lvl)

            f_lvl_contours_pts_x = (gt_contours[:, 0] / downscale_factor).clamp_(max=lvl_size[1] - 1)
            lvl_contours_pts_x = torch.round(f_lvl_contours_pts_x)
            f_lvl_contours_pts_y = (gt_contours[:, 1] / downscale_factor).clamp_(max=lvl_size[0] - 1)
            lvl_contours_pts_y = torch.round(f_lvl_contours_pts_y)
            lvl_indices = (lvl_contours_pts_x + lvl_contours_pts_y * lvl_size[1]).to(torch.long)

            points_index = points_index[lvl_indices]

            assigned_gt_offsets[points_index, 0] = f_lvl_contours_pts_x - lvl_contours_pts_x
            assigned_gt_offsets[points_index, 1] = f_lvl_contours_pts_y - lvl_contours_pts_y

            assigned_gt_ct[points_index] = 0

        pos_inds = torch.nonzero(assigned_gt_ct == 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assigned_gt_ct > 0, as_tuple=False).squeeze(-1).unique()

        return assigned_gt_ct, assigned_gt_offsets, pos_inds, neg_inds
