import numpy as np

import torch

from mmdet.core import bbox2result, bbox_mapping_back, multiclass_nms
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class RepPointsV2Detector(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RepPointsV2Detector, self).__init__(backbone, neck, bbox_head, train_cfg,
                                                test_cfg, pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_sem_map=None,
                      gt_sem_weights=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_sem_map, gt_sem_weights, gt_labels, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def merge_aug_results(self, aug_bboxes, aug_scores, img_metas):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).

        Returns:
            tuple: (bboxes, scores)
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                       flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores

    def aug_test_simple(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        # recompute feats to save memory
        feats = self.extract_feats(imgs)

        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.bbox_head(x)
            bbox_inputs = outs + (img_metas, self.test_cfg, False, False)
            det_bboxes, det_scores = self.bbox_head.get_bboxes(*bbox_inputs)[0]
            aug_bboxes.append(det_bboxes)
            aug_scores.append(det_scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = self.merge_aug_results(
            aug_bboxes, aug_scores, img_metas)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                self.test_cfg.score_thr,
                                                self.test_cfg.nms,
                                                self.test_cfg.max_per_img)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        return bbox_results

    def merge_aug_vote_results(self, aug_bboxes, aug_labels, img_metas):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).
            rcnn_test_cfg (dict): rcnn test config.

        Returns:
            tuple: (bboxes, scores)
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            bboxes[:, :4] = bbox_mapping_back(bboxes[:, :4], img_shape, scale_factor, flip)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_labels is None:
            return bboxes
        else:
            labels = torch.cat(aug_labels, dim=0)
            return bboxes, labels

    def remove_boxes(self, boxes, min_scale, max_scale):
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = torch.nonzero((areas >= min_scale * min_scale) & (areas <= max_scale * max_scale), as_tuple=False).squeeze(1)

        return keep

    def bboxes_vote(self, boxes, scores, vote_thresh=0.66):
        eps = 1e-6

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy().reshape(-1, 1)
        det = np.concatenate((boxes, scores), axis=1)
        if det.shape[0] <= 1:
            return np.zeros((0, 5)), np.zeros((0, 1))
        order = det[:, 4].ravel().argsort()[::-1]
        det = det[order, :]
        dets = []
        while det.shape[0] > 0:
            # IOU
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            xx1 = np.maximum(det[0, 0], det[:, 0])
            yy1 = np.maximum(det[0, 1], det[:, 1])
            xx2 = np.minimum(det[0, 2], det[:, 2])
            yy2 = np.minimum(det[0, 3], det[:, 3])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = area[0] + area[:] - inter
            union = np.maximum(union, eps)
            o = inter / union
            o[0] = 1

            # get needed merge det and delete these  det
            merge_index = np.where(o >= vote_thresh)[0]
            det_accu = det[merge_index, :]
            det_accu_iou = o[merge_index]
            det = np.delete(det, merge_index, 0)

            if merge_index.shape[0] <= 1:
                try:
                    dets = np.row_stack((dets, det_accu))
                except:
                    dets = det_accu
                continue
            else:
                soft_det_accu = det_accu.copy()
                soft_det_accu[:, 4] = soft_det_accu[:, 4] * (1 - det_accu_iou)
                soft_index = np.where(soft_det_accu[:, 4] >= 0.05)[0]
                soft_det_accu = soft_det_accu[soft_index, :]

                det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
                max_score = np.max(det_accu[:, 4])
                det_accu_sum = np.zeros((1, 5))
                det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
                det_accu_sum[:, 4] = max_score

                if soft_det_accu.shape[0] > 0:
                    det_accu_sum = np.row_stack((det_accu_sum, soft_det_accu))

                try:
                    dets = np.row_stack((dets, det_accu_sum))
                except:
                    dets = det_accu_sum

        order = dets[:, 4].ravel().argsort()[::-1]
        dets = dets[order, :]

        boxes = torch.from_numpy(dets[:, :4]).float().cuda()
        scores = torch.from_numpy(dets[:, 4]).float().cuda()

        return boxes, scores

    def aug_test_vote(self, imgs, img_metas, rescale=False):
        # recompute feats to save memory
        feats = self.extract_feats(imgs)

        aug_bboxes = []
        aug_labels = []
        for i, (x, img_meta) in enumerate(zip(feats, img_metas)):
            # only one image in the batch
            # TODO more flexible
            outs = self.bbox_head(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, False, True)
            det_bboxes, det_labels = self.bbox_head.get_bboxes(*bbox_inputs)[0]
            keeped = self.remove_boxes(det_bboxes, self.test_cfg.scale_ranges[i // 2][0], self.test_cfg.scale_ranges[i // 2][1])
            det_bboxes, det_labels = det_bboxes[keeped, :], det_labels[keeped]
            aug_bboxes.append(det_bboxes)
            aug_labels.append(det_labels)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_labels = self.merge_aug_vote_results(
            aug_bboxes, aug_labels, img_metas)

        det_bboxes = []
        det_labels = []
        for j in range(80):
            inds = (merged_labels == j).nonzero().squeeze(1)

            scores_j = merged_bboxes[inds, 4]
            bboxes_j = merged_bboxes[inds, :4].view(-1, 4)
            bboxes_j, scores_j = self.bboxes_vote(bboxes_j, scores_j)

            if len(bboxes_j) > 0:
                det_bboxes.append(torch.cat([bboxes_j, scores_j[:, None]], dim=1))
                det_labels.append(torch.full((bboxes_j.shape[0],), j, dtype=torch.int64, device=scores_j.device))

        if len(det_bboxes) > 0:
            det_bboxes = torch.cat(det_bboxes, dim=0)
            det_labels = torch.cat(det_labels)
        else:
            det_bboxes = merged_bboxes.new_zeros((0, 5))
            det_labels = merged_bboxes.new_zeros((0,), dtype=torch.long)

        if det_bboxes.shape[0] > 1000 > 0:
            cls_scores = det_bboxes[:, 4]
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(),
                det_bboxes.shape[0] - 1000 + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
            det_bboxes = det_bboxes[keep]
            det_labels = det_labels[keep]

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        if self.test_cfg.get("method", "simple") == "simple":
            return self.aug_test_simple(imgs, img_metas, rescale)
        else:
            return self.aug_test_vote(imgs, img_metas, rescale)