import mmcv
import numpy as np
import scipy.interpolate
import torch

from mmdet.core import bbox2result
from .single_stage import SingleStageDetector
from ..builder import DETECTORS


@DETECTORS.register_module()
class DenseRepPointsV2Detector(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(DenseRepPointsV2Detector, self).__init__(backbone, neck, bbox_head, train_cfg,
                                                     test_cfg, pretrained)
    @property
    def with_mask(self):
        return True

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_sem_map=None,
                      gt_contours=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x, test=False)
        loss_inputs = outs + (gt_bboxes, gt_masks, gt_sem_map, gt_contours, gt_labels, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x, test=True)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)

        det_bboxes, det_points, det_points_refine, det_pts_scores, det_pts_scores_refine, det_cls = bbox_list[0]

        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']

        bbox_results = bbox2result(det_bboxes, det_cls, self.bbox_head.num_classes)
        rle_results = self.get_seg_masks(det_pts_scores, det_points, det_bboxes, det_cls, det_pts_scores_refine, det_points_refine,
                                          self.test_cfg, ori_shape, scale_factor, rescale)
        # For visualization(rescale=False), we also return pts_results to show the points
        if not rescale:
            det_points_reshape = det_points.reshape(det_points.shape[0], -1, 2)
            det_pts_scores_reshape = det_pts_scores.reshape(det_pts_scores.shape[0], -1, 1)
            det_pts_score_cat = torch.cat([det_points_reshape, det_pts_scores_reshape], dim=-1) \
                .reshape(det_points.shape[0], -1)
            det_pts_score_cls_cat = torch.cat([det_pts_score_cat, det_points[:, [-1]]], dim=-1)
            pts_results = pts2result(det_pts_score_cls_cat, det_cls, self.bbox_head.num_classes)
            return (bbox_results, rle_results), pts_results
        else:
            return bbox_results, rle_results

    def get_seg_masks(self, pts_score, det_pts, det_bboxes, det_labels, det_scores_refine, det_pts_refine,
                      test_cfg, ori_shape, scale_factor, rescale=False):
        """
        Get segmentation masks from points and scores

        Args:
            pts_score (Tensor or ndarray): shape (n, num_pts)
            det_pts (Tensor): shape (n, num_pts*2)
            det_bboxes (Tensor): shape (n, 4)
            det_labels (Tensor): shape (n, 1)
            test_cfg (dict): rcnn testing config
            ori_shape: original image size
            scale_factor: scale factor for image
            rescale: whether rescale to original size
        Returns:
            list[list]: encoded masks
        """

        cls_segms = [[] for _ in range(self.bbox_head.num_classes)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy()

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
        scale_factor = 1.0

        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0], 1)
            h = max(bbox[3] - bbox[1], 1)

            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            im_pts = det_pts[i].clone()
            im_pts = im_pts.reshape(-1, 2)
            im_pts_score = pts_score[i]

            im_pts = im_pts[im_pts_score > 0, :]
            im_pts_score = im_pts_score[im_pts_score > 0]

            if det_pts_refine is not None:
                det_pts_refine_valid = det_pts_refine[i].reshape(-1, 2)
                det_pts_refine_valid = det_pts_refine_valid[det_scores_refine[i] > 0, :]
                det_scores_refine_valid = det_scores_refine[i][det_scores_refine[i] > 0]
                im_pts = torch.cat([im_pts, det_pts_refine_valid])
                im_pts_score = torch.cat([im_pts_score, det_scores_refine_valid])

            im_pts[:, 0] = (im_pts[:, 0] - bbox[0])
            im_pts[:, 1] = (im_pts[:, 1] - bbox[1])
            _h, _w = h, w
            corner_pts = im_pts.new_tensor([[0, 0], [_h - 1, 0], [0, _w - 1], [_w - 1, _h - 1]])
            corner_score = im_pts_score.new_tensor([0, 0, 0, 0])
            im_pts = torch.cat([im_pts, corner_pts], dim=0).cpu().numpy()
            im_pts_score = torch.cat([im_pts_score, corner_score], dim=0).cpu().numpy()
            # im_pts = im_pts.cpu().numpy()
            # im_pts_score = im_pts_score.cpu().numpy()
            # im_pts_score = (im_pts_score > 0.5).astype(np.float32)
            grids = tuple(np.mgrid[0:_w:1, 0:_h:1])
            bbox_mask = scipy.interpolate.griddata(im_pts, im_pts_score, grids)
            bbox_mask = bbox_mask.transpose(1, 0)
            bbox_mask = mmcv.imresize(bbox_mask, (w, h))

            bbox_mask = bbox_mask.astype(np.float32)
            bbox_mask[np.isnan(bbox_mask)] = 0
            bbox_mask = (bbox_mask > test_cfg.get('pts_score_thr', 0.5)).astype(np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            cls_segms[label].append(im_mask)
        return cls_segms


def pts2result(pts, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, pts_num)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if pts.shape[0] == 0:
        return [np.zeros((0, pts.shape[1]), dtype=np.float32) for i in range(num_classes)]
    else:
        pts = pts.cpu().numpy()
        labels = labels.cpu().numpy()
        return [pts[labels == i, :] for i in range(num_classes)]
