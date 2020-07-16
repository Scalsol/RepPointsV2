import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import mmcv
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from mmdet.core import (PointGenerator, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms_pts, unmap)
from mmdet.ops import DeformConv
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead


@HEADS.register_module()
class DenseRepPointsHead(AnchorFreeHead):
    """RepPoint head.

    Args:
        point_feat_channels (int): Number of channels of points features.
        gradient_mul (float): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Iterable): points strides.
        point_base_scale (int): bbox scale for assigning labels.
        loss_cls (dict): Config of classification loss.
        loss_bbox_init (dict): Config of initial points loss.
        loss_bbox_refine (dict): Config of points loss in refinement.
        use_grid_points (bool): If we use bounding box representation, the
        reppoints is represented as grid points on the bounding box.
        center_init (bool): Whether to use center point assignment.
        transform_method (str): The methods to transform RepPoints to bbox.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 point_feat_channels=256,
                 stacked_mask_convs=3,
                 num_group=9,
                 num_points=729,
                 num_score_group=121,
                 gradient_mul=0.1,
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=4,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_init=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_bbox_refine=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_pts_init=dict(type='ChamferLoss2D', use_cuda=True, loss_weight=0.5, eps=1e-12),
                 loss_pts_refine=dict(type='ChamferLoss2D', use_cuda=True, loss_weight=1.0, eps=1e-12),
                 loss_mask_score_init=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 transform_method='minmax',
                 sample_padding_mode='border',
                 fuse_mask_feat=False,
                 **kwargs):
        self.num_group = num_group
        self.num_points = num_points
        self.num_score_group = num_score_group
        self.point_feat_channels = point_feat_channels
        self.stacked_mask_convs = stacked_mask_convs
        self.fuse_mask_feat = fuse_mask_feat
        self.sample_padding_mode = sample_padding_mode

        # we use deformable conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.point_generators = [PointGenerator() for _ in self.point_strides]

        if self.train_cfg:
            self.init_assigner = build_assigner(self.train_cfg.init.assigner)
            self.refine_assigner = build_assigner(self.train_cfg.refine.assigner)
            # use PseudoSampler when sampling is False
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.transform_method = transform_method


        self.cls_out_channels = self.num_classes
        self.loss_bbox_init = build_loss(loss_bbox_init)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)
        self.loss_pts_init = build_loss(loss_pts_init)
        self.loss_pts_refine = build_loss(loss_pts_refine)
        self.loss_mask_score_init = build_loss(loss_mask_score_init)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.mask_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        for i in range(self.stacked_mask_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.mask_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        pts_out_dim = 2 * self.num_points
        self.reppoints_cls_conv = nn.Conv2d(self.feat_channels * self.num_group, self.point_feat_channels, 1, 1, 0)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels, self.cls_out_channels, 1, 1, 0)

        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels, pts_out_dim, 1, 1, 0)

        self.reppoints_pts_refine_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels, pts_out_dim, 1, 1, 0)

        self.reppoints_mask_init_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)
        self.reppoints_mask_init_out = nn.Conv2d(self.point_feat_channels, self.num_score_group, 1, 1, 0)

        if self.fuse_mask_feat:
            self.mask_fuse_conv = nn.Conv2d(self.feat_channels, self.feat_channels, 3, 1, 1)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.mask_convs:
            normal_init(m.conv, std=0.01)
        if self.fuse_mask_feat:
            normal_init(self.mask_fuse_conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.reppoints_cls_conv, std=0.01)
        normal_init(self.reppoints_cls_out, std=0.01, bias=bias_cls)
        normal_init(self.reppoints_pts_init_conv, std=0.01)
        normal_init(self.reppoints_pts_init_out, std=0.01)
        normal_init(self.reppoints_pts_refine_conv, std=0.01)
        normal_init(self.reppoints_pts_refine_out, std=0.01)
        normal_init(self.reppoints_mask_init_conv, std=0.01)
        normal_init(self.reppoints_mask_init_out, std=0.01)

    def points2bbox(self, pts):
        """Converting the points set into bounding box.

        :param pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        :param y_first: if y_fisrt=True, the point set is represented as
            [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
            represented as [x1, y1, x2, y2 ... xn, yn].
        :return: each points set is converting to a bbox [x1, y1, x2, y2].
        """
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_x = pts_reshape[:, :, 0, ...]
        pts_y = pts_reshape[:, :, 1, ...]
        if self.transform_method == 'minmax':
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom], dim=1)
        else:
            raise NotImplementedError
        return bbox

    def sample_offset(self, x, flow, padding_mode):
        """
        sample feature based on offset

            Args:
                x (Tensor): input feature, size (n, c, h, w)
                flow (Tensor): flow fields, size(n, 2, h', w')
                padding_mode (str): grid sample padding mode, 'zeros' or 'border'
            Returns:
                Tensor: warped feature map (n, c, h', w')
        """
        # assert x.size()[-2:] == flow.size()[-2:]
        n, _, h, w = flow.size()
        x_ = torch.arange(w).view(1, -1).expand(h, -1)
        y_ = torch.arange(h).view(-1, 1).expand(-1, w)
        grid = torch.stack([x_, y_], dim=0).float().cuda()
        grid = grid.unsqueeze(0).expand(n, -1, -1, -1)
        grid = grid + flow
        gx = 2 * grid[:, 0, :, :] / (w - 1) - 1
        gy = 2 * grid[:, 1, :, :] / (h - 1) - 1
        grid = torch.stack([gx, gy], dim=1)
        grid = grid.permute(0, 2, 3, 1)
        return F.grid_sample(x, grid, padding_mode=padding_mode, align_corners=True)

    def compute_offset_feature(self, x, offset, padding_mode):
        """
        sample feature based on offset

            Args:
                x (Tensor) : feature map, size (n, C, h, w),  x first
                offset (Tensor) : offset, size (n, sample_pts*2, h, w), x first
                padding_mode (str): 'zeros' or 'border' or 'relection'
            Returns:
                Tensor: the warped feature generated by the offset and the input feature map, size (n, sample_pts, C, h, w)
        """
        offset_reshape = offset.view(offset.shape[0], -1, 2, offset.shape[2], offset.shape[3])  # (n, sample_pts, 2, h, w)
        num_pts = offset_reshape.shape[1]
        offset_reshape = offset_reshape.contiguous().view(-1, 2, offset.shape[2],
                                                          offset.shape[3])  # (n*sample_pts, 2, h, w)
        x_repeat = x.unsqueeze(1).repeat(1, num_pts, 1, 1, 1)  # (n, sample_pts, C, h, w)
        x_repeat = x_repeat.view(-1, x_repeat.shape[2], x_repeat.shape[3], x_repeat.shape[4])  # (n*sample_pts, C, h, w)
        sampled_feat = self.sample_offset(x_repeat, offset_reshape, padding_mode)  # (n*sample_pts, C, h, w)
        sampled_feat = sampled_feat.view(-1, num_pts, sampled_feat.shape[1], sampled_feat.shape[2],
                                         sampled_feat.shape[3])  # (n, sample_pts, C, h, w)
        return sampled_feat

    def sample_offset_3d(self, x, flow, padding_mode):
        """
        sample feature based on 2D offset(x, y) + 1-D index(z)

            Args:
                x (Tensor): size (n, c, d', h', w')
                flow (Tensor): size(n, 3, d, h, w)
                padding_mode (str): 'zeros' or 'border'
            Returns:
                warped feature map generated by the offset and the input feature map, size(n, c, d, h, w)
        """
        n, _, d, h, w = flow.size()
        num_group = x.shape[2]
        device = flow.get_device()
        x_ = torch.arange(w, device=device).view(1, 1, -1).expand(d, h, -1).float()  # (d, h, w)
        y_ = torch.arange(h, device=device).view(1, -1, 1).expand(d, -1, w).float()  # (d, h, w)
        z_ = torch.zeros(d, h, w, device=device)  # (d, h, w)
        grid = torch.stack([x_, y_, z_], dim=0).float()  # (3, d, h, w)
        del x_, y_, z_
        grid = grid.unsqueeze(0).expand(n, -1, -1, -1, -1)  # (n, 3, d, h, w)
        grid = grid + flow  # (n, 3, d, h, w)
        gx = 2 * grid[:, 0, :, :, :] / (w - 1) - 1  # (n, d, h, w)
        gy = 2 * grid[:, 1, :, :, :] / (h - 1) - 1  # (n, d, h, w)
        gz = 2 * grid[:, 2, :, :, :] / (num_group - 1) - 1  # (n, d, h, w)
        grid = torch.stack([gx, gy, gz], dim=1)  # (n, 3, d, h, w)
        del gx, gy, gz
        grid = grid.permute(0, 2, 3, 4, 1)  # (n, d, h, w, 3)
        return F.grid_sample(x, grid, padding_mode=padding_mode, align_corners=True)

    def compute_offset_feature_5d(self, x, offset, padding_mode):
        """
        sample 5D feature based on offset

            Args:
                x (Tensor) : input feature, size (n, C, d', h', w'), x first
                offset (Tensor) : flow field, size (n, 3, sample_pts, h, w), x first
                padding_mode (str): 'zeros' or 'border'
            Returns:
                Tensor: offset_feature, size (n, sample_pts, C, h, w)
        """
        sampled_feat = self.sample_offset_3d(x, offset, padding_mode)  # (n, C, sample_pts, h, w)
        sampled_feat = sampled_feat.transpose(1, 2)  # (n, sample_pts, C, h, w)
        return sampled_feat

    def forward(self, feats, test=False):
        cls_out_list, pts_out_init_list, pts_out_refine_list = multi_apply(self.forward_pts_head_single, feats)
        if test:
            pts_out_list = pts_out_refine_list
        else:
            pts_out_list = [(1 - self.gradient_mul) * pts_out_init.detach()
                            + self.gradient_mul * pts_out_init for pts_out_init in pts_out_init_list]

        pts_score_out = self.forward_mask_head(feats, pts_out_list)
        return cls_out_list, pts_out_init_list, pts_out_refine_list, pts_score_out

    def forward_pts_head_single(self, x):
        b, _, h, w = x.shape
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        scale = self.point_base_scale / 2
        points_init = dcn_base_offset / dcn_base_offset.max() * scale

        cls_feat = x
        pts_feat = x

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)

        # generate points_init
        pts_out_init = self.reppoints_pts_init_out(self.relu(self.reppoints_pts_init_conv(pts_feat)))
        pts_out_init = pts_out_init + points_init  # (b, 2n, h, w)
        pts_out_init_detach = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init

        # classify dense reppoints based on group pooling
        cls_offset = pts_out_init_detach.view(b, self.num_group, -1, 2, h, w)
        cls_offset = cls_offset[:, :, 0, ...].reshape(b, -1, h, w)

        cls_pts_feature = self.compute_offset_feature(cls_feat, cls_offset, padding_mode=self.sample_padding_mode)
        cls_pts_feature = cls_pts_feature.contiguous().view(b, -1, h, w)

        cls_out = self.reppoints_cls_out(self.relu(self.reppoints_cls_conv(cls_pts_feature)))

        # generate offset field
        pts_refine_field = self.reppoints_pts_refine_out(self.relu(self.reppoints_pts_refine_conv(pts_feat)))  # (b, n*2, h, w)
        pts_refine_field = pts_refine_field.view(b * self.num_points, -1, h, w)  # (b*n, 2, h, w)
        pts_out_init_detach_reshape = pts_out_init_detach.view(b, -1, 2, h, w).view(-1, 2, h, w)  # (b*n, 2, h, w)
        pts_out_refine = self.compute_offset_feature(pts_refine_field, pts_out_init_detach_reshape,padding_mode=self.sample_padding_mode)  # (b*n, 2, h, w)
        pts_out_refine = pts_out_refine.view(b, -1, h, w)  # (b, n*2, h, w)
        # generate points_refine
        pts_out_refine = pts_out_refine + pts_out_init_detach

        return cls_out, pts_out_init, pts_out_refine

    def forward_mask_head(self, mask_feat_list, pts_out_list):
        for mask_conv in self.mask_convs:
            mask_feat_list = [mask_conv(mask_feat) for mask_feat in mask_feat_list]
        if self.fuse_mask_feat:
            mask_feat_high_res = mask_feat_list[0]
            H, W = mask_feat_high_res.shape[-2:]
            mask_feat_up_list = []
            for lvl, mask_feat in enumerate(mask_feat_list):
                mask_feat_up = mask_feat
                if lvl > 0:
                    mask_feat_up = F.interpolate(
                        mask_feat, size=(H, W), mode="bilinear", align_corners=False
                    )
                    del mask_feat
                mask_feat_up_list.append(
                    self.mask_fuse_conv(mask_feat_up + mask_feat_high_res)
                )
                del mask_feat_up
            del mask_feat_high_res
            del mask_feat_list
            mask_feat_list = mask_feat_up_list

        pts_score_out = multi_apply(self.forward_mask_head_single, pts_out_list, mask_feat_list)[0]
        return pts_score_out

    def forward_mask_head_single(self, pts, mask_feat):
        b, _, h, w = mask_feat.shape
        h_pts, w_pts = pts.shape[-2:]
        score_map = self.reppoints_mask_init_out(
            self.relu(self.reppoints_mask_init_conv(mask_feat)))  # (b, G*1, h, w)
        # position sensitive group partition based on grids
        pts_reshape_detach = pts.detach().view(b, -1, 2, h_pts, w_pts)  # (b, n, 2, h_pts, w_pts)
        group_inds = self.grid_position_sensitive_group_partition(
            pts_reshape_detach, self.num_score_group)  # (b, 1, n, h_pts, w_pts)
        del pts_reshape_detach
        score_map = score_map.unsqueeze(1)  # (b, 1, G, h, w)

        pts_reshape = pts.view(b, -1, 2, h_pts, w_pts).transpose(1, 2)  # (b, 2, n, h_pts, w_pts)
        pts_reshape = pts_reshape.detach()
        _pts_inds_cat = torch.cat([pts_reshape, group_inds], dim=1)  # (b, 3, n, h_pts, w_pts)
        del group_inds, pts_reshape
        # position sensitive sampling on score maps
        pts_score_out = self.compute_offset_feature_5d(
            score_map, _pts_inds_cat, padding_mode=self.sample_padding_mode)  # (b, n, 1, h_pts, w_pts)

        pts_score_out = pts_score_out.view(b, -1, h_pts, w_pts)  # (b, n, h_pts, w_pts)
        return pts_score_out, _

    @staticmethod
    def normalize_pts_within_bboxes(pts):
        """
        Normalize pts offset within bboxes(instance level)

            Args:
                pts(Tensor): input points, size (b, n, 2, h_pts, w_pts)

            Returns:
                Tensor: normalized_pts, size (b, n, 2, h_pts, w_pts)
        """
        b, _, _, h_pts, w_pts = pts.shape
        _pts_x = pts[:, :, 0, :, :]  # (b, n, h_pts, w_pts)
        _pts_y = pts[:, :, 1, :, :]  # (b, n, h_pts, w_pts)
        _bbox_left = torch.min(_pts_x, dim=1, keepdim=True)[0]  # (b, 1, h_pts, w_pts)
        _bbox_right = torch.max(_pts_x, dim=1, keepdim=True)[0]  # (b, 1, h_pts, w_pts)
        _bbox_bottom = torch.max(_pts_y, dim=1, keepdim=True)[0]  # (b, 1, h_pts, w_pts)
        _bbox_up = torch.min(_pts_y, dim=1, keepdim=True)[0]  # (b, 1, h_pts, w_pts)
        _bbox_w = _bbox_right - _bbox_left  # (b, 1, h_pts, w_pts)
        _bbox_h = _bbox_bottom - _bbox_up  # (b, 1, h_pts, w_pts)

        normalized_x = (_pts_x - _bbox_left) / (_bbox_w + 1e-6)  # (b, n, h_pts, w_pts)
        normalized_y = (_pts_y - _bbox_up) / (_bbox_h + 1e-6)  # (b, n, h_pts, w_pts)
        normalized_pts = torch.stack([normalized_x, normalized_y], dim=2)  # (b, n, 2, h_pts, w_pts)
        return normalized_pts

    def grid_position_sensitive_group_partition(self, pts, num_group):
        """
        Position-sensitive group partition based on grids.

            Args:
                pts(Tensor): input points, size (b, n, 2, h_pts, w_pts)
                num_group(int): the number of groups

            Returs:
                Tensor: group_inds, size (b, 1, n, h_pts, w_pts)
        """
        normalized_pts = self.normalize_pts_within_bboxes(pts)  # (b, n, 2, h_pts, w_pts)
        normalized_x = normalized_pts[:, :, 0, :, :]  # (b, n, h_pts, w_pts)
        normalized_y = normalized_pts[:, :, 1, :, :]  # (b, n, h_pts, w_pts)

        num_group_kernel = int(np.sqrt(num_group))
        grid_x_inds = (normalized_x * num_group_kernel).long()  # (b, n, h_pts, w_pts)
        grid_y_inds = (normalized_y * num_group_kernel).long()  # (b, n, h_pts, w_pts)
        group_inds = grid_y_inds * num_group_kernel + grid_x_inds  # (b, n, h_pts, w_pts)
        group_inds = group_inds.unsqueeze(1).float()  # (b, 1, n, h_pts, w_pts)
        return group_inds

    def get_points(self, featmap_sizes, img_metas):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(
                featmap_sizes[i], self.point_strides[i])
            multi_level_points.append(points)
        points_list = [[point.clone() for point in multi_level_points] for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                point_stride = self.point_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w = img_meta['pad_shape'][:2]
                valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

    def offset_to_pts(self, center_list, pred_list):
        """Change from point offset to point coordinate."""
        pts_list = []
        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                xy_pts_shift = pts_shift.permute(1, 2, 0).view( -1, 2 * self.num_points)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    # pts_to_img_lvl
    def offset_to_pts_img_lvl(self, center_list, pred_list):
        """
        Project points offset based on center point to image scale and organized in image-level order

            Args:
                center_list(list(Tensor)): Multi image center list with different level
                pred_list: Multi image pred points offset with different level
            Returns:
                list(Tensor): multi-image points in image scale with different level
        """
        pts_list = []
        for i_img, point in enumerate(center_list):
            pts_img = []
            for i_lvl in range(len(center_list[0])):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                xy_pts_shift = pts_shift.permute(1, 2, 0).view(-1, 2 * self.num_points)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_img.append(pts)
            pts_list.append(pts_img)
        return pts_list

    def _dense_point_target_single(self,
                                   flat_proposals,
                                   flat_proposals_pts,
                                   valid_flags,
                                   num_level_proposals,
                                   gt_bboxes,
                                   gt_bboxes_ignore,
                                   gt_masks,
                                   gt_labels,
                                   num_pts,
                                   label_channels=1,
                                   stage='init',
                                   unmap_outputs=True):
        inside_flags = valid_flags
        if not inside_flags.any():
            return (None, ) * 9
        # assign gt and sample proposals
        proposals = flat_proposals[inside_flags, :]
        proposals_pts = flat_proposals_pts[inside_flags, :]

        num_level_proposals_inside = self.get_num_level_proposals_inside(num_level_proposals, inside_flags)
        if stage == 'init':
            assigner = self.init_assigner
            assigner_type = self.train_cfg.init.assigner.type
            pos_weight = self.train_cfg.init.pos_weight
        else:
            assigner = self.refine_assigner
            assigner_type = self.train_cfg.refine.assigner.type
            pos_weight = self.train_cfg.refine.pos_weight
        if assigner_type != "ATSSAssigner":
            assign_result = assigner.assign(proposals, gt_bboxes, gt_bboxes_ignore, gt_labels)
        else:
            assign_result = assigner.assign(proposals, num_level_proposals_inside, gt_bboxes, gt_bboxes_ignore, gt_labels)
        sampling_result = self.sampler.sample(assign_result, proposals, gt_bboxes)

        gt_ind = sampling_result.pos_assigned_gt_inds.cpu().numpy()
        gt_pts_numpy = distance_sample_pts(gt_bboxes, gt_masks, self.train_cfg.get(stage), num_pts)

        pts_label_list = []
        proposals_pos_pts = proposals_pts[sampling_result.pos_inds, :].detach().cpu().numpy().round().astype(np.long)
        for i in range(len(gt_ind)):
            gt_mask = gt_masks.masks[gt_ind[i]]
            h, w = gt_mask.shape
            pts_long = proposals_pos_pts[i]
            _pts_label = gt_mask[pts_long[1::2].clip(0, h - 1), pts_long[0::2].clip(0, w - 1)]
            pts_label_list.append(_pts_label)
        del proposals_pos_pts

        if len(gt_ind) != 0:
            gt_pts = gt_bboxes.new_tensor(gt_pts_numpy)
            pos_gt_pts = gt_pts[gt_ind]
            pts_label = np.stack(pts_label_list, 0)
            pos_gt_pts_label = gt_bboxes.new_tensor(pts_label)
        else:
            pos_gt_pts = None
            pos_gt_pts_label = None

        num_valid_proposals = proposals.shape[0]
        bbox_gt = proposals.new_zeros([num_valid_proposals, 4])
        bbox_weights = proposals.new_zeros([num_valid_proposals, 4])
        mask_gt = proposals.new_zeros([0, num_pts * 2])
        mask_gt_label = proposals.new_zeros([0, int(flat_proposals_pts.shape[1] / 2)]).long()
        mask_gt_index = proposals.new_zeros([num_valid_proposals, ], dtype=torch.long)
        labels = proposals.new_full((num_valid_proposals, ), self.background_label, dtype=torch.long)
        label_weights = proposals.new_zeros(num_valid_proposals, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_gt_bboxes = sampling_result.pos_gt_bboxes
            bbox_gt[pos_inds, :] = pos_gt_bboxes
            bbox_weights[pos_inds, :] = 1.0
            if pos_gt_pts is not None:
                mask_gt = pos_gt_pts.type(bbox_gt.type())
                mask_gt_index[pos_inds] = torch.arange(len(pos_inds)).long().cuda() + 1
            if pos_gt_pts_label is not None:
                mask_gt_label = pos_gt_pts_label.long()
            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of proposals
        if unmap_outputs:
            num_total_proposals = flat_proposals.size(0)
            labels = unmap(labels, num_total_proposals, inside_flags)
            label_weights = unmap(label_weights, num_total_proposals, inside_flags)
            bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_proposals, inside_flags)
            mask_gt_index = unmap(mask_gt_index, num_total_proposals, inside_flags)

        return labels, label_weights, bbox_gt, bbox_weights, mask_gt_index, mask_gt, mask_gt_label, pos_inds, neg_inds

    def get_targets(self,
                    proposals_list,
                    proposals_pts_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    gt_masks_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    num_pts=729,
                    stage='init',
                    label_channels=1,
                    unmap_outputs=True):
        """Compute corresponding GT box and classification targets for
        proposals.

        Args:
            proposals_list (list[list]): Multi level points/bboxes of each
                image.
            valid_flag_list (list[list]): Multi level valid flags of each
                image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_bboxes_list (list[Tensor]): Ground truth labels of each box.
            stage (str): `init` or `refine`. Generate target for init stage or
                refine stage
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each level.  # noqa: E501
                - bbox_gt_list (list[Tensor]): Ground truth bbox of each level.
                - proposal_list (list[Tensor]): Proposals(points/bboxes) of each level.  # noqa: E501
                - proposal_weights_list (list[Tensor]): Proposal weights of each level.  # noqa: E501
                - num_total_pos (int): Number of positive samples in all images.  # noqa: E501
                - num_total_neg (int): Number of negative samples in all images.  # noqa: E501
        """
        assert stage in ['init', 'refine']
        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs

        # points number of multi levels
        num_level_proposals = [points.size(0) for points in proposals_list[0]]
        num_level_proposals_list = [num_level_proposals] * num_imgs

        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])
            proposals_pts_list[i] = torch.cat(proposals_pts_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_labels, all_label_weights, all_bbox_gt, all_bbox_weights,
         all_mask_gt_index, all_mask_gt, all_mask_gt_label,
         pos_inds_list, neg_inds_list) = multi_apply(
             self._dense_point_target_single,
             proposals_list,
             proposals_pts_list,
             valid_flag_list,
             num_level_proposals_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_masks_list,
             gt_labels_list,
             num_pts=num_pts,
             stage=stage,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid points
        if any([labels is None for labels in all_labels]):
            return None
        # sampled points of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        labels_list = images_to_levels(all_labels, num_level_proposals)
        label_weights_list = images_to_levels(all_label_weights, num_level_proposals)
        bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                                 num_level_proposals)
        mask_gt_index_list = images_to_levels(all_mask_gt_index, num_level_proposals)
        mask_gt_list = mask_to_levels(all_mask_gt, mask_gt_index_list)
        mask_gt_label_list = mask_to_levels(all_mask_gt_label, mask_gt_index_list)

        return (labels_list, label_weights_list, bbox_gt_list, bbox_weights_list,
                mask_gt_list, mask_gt_label_list,
                num_total_pos, num_total_neg)

    def loss_single(self, cls_score, pts_pred_init, pts_pred_refine, pts_score_pred_init,
                    labels, label_weights,
                    bbox_gt_init, pts_gt_init, bbox_weights_init,
                    bbox_gt_refine, pts_gt_refine, pts_score_gt_label, bbox_weights_refine,
                    stride, num_total_samples_init, num_total_samples_refine):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples_refine)

        # bbox loss
        bbox_gt_init = bbox_gt_init.reshape(-1, 4)
        bbox_weights_init = bbox_weights_init.reshape(-1, 4)
        bbox_pred_init = self.points2bbox(pts_pred_init.reshape(-1, 2 * self.num_points))
        bbox_gt_refine = bbox_gt_refine.reshape(-1, 4)
        bbox_weights_refine = bbox_weights_refine.reshape(-1, 4)
        bbox_pred_refine = self.points2bbox(pts_pred_refine.reshape(-1, 2 * self.num_points))
        normalize_term = self.point_base_scale * stride

        loss_bbox_init = self.loss_bbox_init(
            bbox_pred_init / normalize_term,
            bbox_gt_init / normalize_term,
            bbox_weights_init,
            avg_factor=num_total_samples_init)
        loss_bbox_refine = self.loss_bbox_refine(
            bbox_pred_refine / normalize_term,
            bbox_gt_refine / normalize_term,
            bbox_weights_refine,
            avg_factor=num_total_samples_refine)

        # pts_loss_init
        valid_pts_gt_init = torch.cat(pts_gt_init, 0)
        valid_pts_gt_init = valid_pts_gt_init.view(-1, self.num_points, 2)
        mask_pred_init = pts_pred_init.reshape(-1, 2 * self.num_points)
        valid_pts_pred_init = mask_pred_init[bbox_weights_init[:, 0] > 0]
        valid_pts_pred_init = valid_pts_pred_init.view(-1, self.num_points, 2)
        valid_pts = valid_pts_gt_init.sum(-1).sum(-1) > 0
        num_total_samples = max(num_total_samples_init, 1)
        loss_pts_init = self.loss_pts_init(
            valid_pts_gt_init[valid_pts] / normalize_term,
            valid_pts_pred_init[valid_pts] / normalize_term).sum() / num_total_samples
        # pts_loss_refine
        valid_pts_gt_refine = torch.cat(pts_gt_refine, 0)
        valid_pts_gt_refine = valid_pts_gt_refine.view(-1, self.num_points, 2)
        pts_pred_refine = pts_pred_refine.reshape(-1, 2 * self.num_points)
        valid_pts_pred_refine = pts_pred_refine[bbox_weights_refine[:, 0] > 0]
        valid_pts_pred_refine = valid_pts_pred_refine.view(-1, self.num_points, 2)
        valid_pts = valid_pts_gt_refine.sum(-1).sum(-1) > 0
        num_total_samples = max(num_total_samples_refine, 1)
        loss_pts_refine = self.loss_pts_refine(
            valid_pts_gt_refine[valid_pts] / normalize_term,
            valid_pts_pred_refine[valid_pts] / normalize_term).sum() / num_total_samples
        # mask score loss
        valid_pts_score_gt_label = torch.cat(pts_score_gt_label, 0)
        valid_pts_score_gt_label = valid_pts_score_gt_label.view(-1, self.num_points, 1)
        pts_score_pred_init = pts_score_pred_init.reshape(-1, self.num_points)
        valid_pts_score_pred_init = pts_score_pred_init[bbox_weights_refine[:, 0] > 0]
        valid_pts_score_pred_init = valid_pts_score_pred_init.view(-1, self.num_points, 1)
        valid_pts_score_inds = (valid_pts_score_gt_label.sum(-1).sum(-1) > 0)
        num_total_samples = max(num_total_samples_refine, 1)
        loss_mask_score_init = self.loss_mask_score_init(
            valid_pts_score_pred_init[valid_pts_score_inds],
            valid_pts_score_gt_label[valid_pts_score_inds],
            weight=bbox_weights_init.new_ones(*valid_pts_score_pred_init[valid_pts_score_inds].shape),
            avg_factor=num_total_samples
        ) / self.num_points

        return loss_cls, loss_bbox_init, loss_pts_init, loss_bbox_refine, loss_pts_refine, loss_mask_score_init

    def loss(self,
             cls_scores,
             pts_preds_init,
             pts_preds_refine,
             pts_preds_score_init,
             gt_bboxes,
             gt_masks,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.point_generators)
        label_channels = self.cls_out_channels

        # target for initial stage
        center_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)
        real_pts_preds_init = self.offset_to_pts(center_list, pts_preds_init)
        proposal_pts_list = self.offset_to_pts_img_lvl(center_list, pts_preds_init)
        real_pts_preds_score_init = []
        for lvl_pts_score in pts_preds_score_init:
            b = lvl_pts_score.shape[0]
            real_pts_preds_score_init.append(lvl_pts_score.permute(0, 2, 3, 1).view(b, -1, self.num_points))

        cls_reg_targets_init = self.get_targets(
            center_list,
            proposal_pts_list,
            valid_flag_list,
            gt_bboxes,
            gt_masks,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            num_pts=self.num_points,
            stage='init',
            label_channels=label_channels)
        (*_, bbox_gt_list_init, bbox_weights_list_init, pts_gt_list_init, _,
         num_total_pos_init, num_total_neg_init) = cls_reg_targets_init

        # target for refinement stage
        center_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)
        real_pts_preds_refine = self.offset_to_pts(center_list, pts_preds_refine)
        bbox_pts_list = self.offset_to_pts_img_lvl(center_list, pts_preds_init)

        bbox_list = []
        for i_img, center in enumerate(center_list):
            bbox = []
            for i_lvl in range(len(pts_preds_refine)):
                bbox_preds_init = self.points2bbox(pts_preds_init[i_lvl].detach())
                bbox_shift = bbox_preds_init * self.point_strides[i_lvl]
                bbox_center = torch.cat([center[i_lvl][:, :2], center[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift[i_img].permute(1, 2, 0).reshape(-1, 4))
            bbox_list.append(bbox)
        cls_reg_targets_refine = self.get_targets(
            bbox_list,
            bbox_pts_list,
            valid_flag_list,
            gt_bboxes,
            gt_masks,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            num_pts=self.num_points,
            stage='refine',
            label_channels=label_channels)
        (labels_list, label_weights_list,
         bbox_gt_list_refine, bbox_weights_list_refine, pts_gt_list_refine, pts_score_gt_label_list,
         num_total_pos_refine, num_total_neg_refine) = cls_reg_targets_refine

        # compute loss
        losses_cls, losses_bbox_init, losses_pts_init, losses_bbox_refine, losses_pts_refine, losses_mask_score_init = multi_apply(
            self.loss_single,
            cls_scores,
            real_pts_preds_init,
            real_pts_preds_refine,
            real_pts_preds_score_init,
            labels_list,
            label_weights_list,
            bbox_gt_list_init,
            pts_gt_list_init,
            bbox_weights_list_init,
            bbox_gt_list_refine,
            pts_gt_list_refine,
            pts_score_gt_label_list,
            bbox_weights_list_refine,
            self.point_strides,
            num_total_samples_init=num_total_pos_init,
            num_total_samples_refine=num_total_pos_refine)

        loss_dict_all = {'loss_cls': losses_cls,
                         'loss_bbox_init': losses_bbox_init,
                         'losses_pts_init': losses_pts_init,
                         'losses_bbox_refine': losses_bbox_refine,
                         'losses_pts_refine': losses_pts_refine,
                         'losses_mask_score_init': losses_mask_score_init,
                         }
        return loss_dict_all

    def get_bboxes(self,
                   cls_scores,
                   pts_preds_init,
                   pts_preds_refine,
                   pts_preds_score_refine,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   nms=True):
        assert len(cls_scores) == len(pts_preds_refine)
        bbox_preds_refine = [self.points2bbox(pts_pred_refine) for pts_pred_refine in pts_preds_refine]
        num_levels = len(cls_scores)
        mlvl_points = [
            self.point_generators[i].grid_points(cls_scores[i].size()[-2:],
                                                 self.point_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds_refine[i][img_id].detach() for i in range(num_levels)
            ]
            pts_pred_list = [
                pts_preds_refine[i][img_id].detach() for i in range(num_levels)
            ]
            mask_pred_list = [
                pts_preds_score_refine[i][img_id].sigmoid().detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list, pts_pred_list, mask_pred_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale,
                                                nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           pts_preds,
                           mask_preds,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           nms=True):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_pts = []
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_masks = []
        for i_lvl, (cls_score, bbox_pred, pts_pred, mask_pred, points) in enumerate(zip(cls_scores, bbox_preds, pts_preds, mask_preds, mlvl_points)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            pts_pred = pts_pred.permute(1, 2, 0).reshape(-1, 2 * self.num_points)
            mask_pred = mask_pred.permute(1, 2, 0).reshape(-1, self.num_points)

            # mask scoring
            mask_sum = (mask_pred > 0.5).sum(1).float()
            mask_score = ((mask_pred > 0.5).float() * mask_pred).sum(1) / (mask_sum + 1e-6)
            scores = scores * mask_score.unsqueeze(1)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                pts_pred = pts_pred[topk_inds, :]
                mask_pred = mask_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            pts_pos_center = points[:, :2].repeat(1, self.num_points)
            pts = pts_pred * self.point_strides[i_lvl] + pts_pos_center
            pts[:, 0::2] = pts[:, 0::2].clamp(min=0, max=img_shape[1] - 1)
            pts[:, 1::2] = pts[:, 1::2].clamp(min=0, max=img_shape[0] - 1)

            bbox_pos_center = torch.cat([points[:, :2], points[:, :2]], dim=1)
            bboxes = bbox_pred * self.point_strides[i_lvl] + bbox_pos_center
            x1 = bboxes[:, 0].clamp(min=0, max=img_shape[1])
            y1 = bboxes[:, 1].clamp(min=0, max=img_shape[0])
            x2 = bboxes[:, 2].clamp(min=0, max=img_shape[1])
            y2 = bboxes[:, 3].clamp(min=0, max=img_shape[0])
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)

            mlvl_pts.append(pts)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_masks.append(mask_pred)
        mlvl_pts = torch.cat(mlvl_pts)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_pts /= mlvl_pts.new_tensor(scale_factor[:2]).repeat(mlvl_pts.shape[1] // 2)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_masks = torch.cat(mlvl_masks)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        if nms:
            det_bboxes, det_pts, det_masks, det_labels = multiclass_nms_pts(
                mlvl_bboxes, mlvl_pts, mlvl_scores, mlvl_masks, cfg.score_thr, cfg.nms, cfg.max_per_img)
            return det_bboxes, det_pts, det_masks, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    def get_num_level_proposals_inside(self, num_level_proposals, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_proposals)
        num_level_proposals_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_proposals_inside


def mask_to_levels(target, mask_index_list):
    """
    Convert target by mask_index_list
    """
    target_gt_list = []
    for lvl in range(len(mask_index_list)):
        mask_gt_lvl_list = []
        for i in range(mask_index_list[lvl].shape[0]):
            index = mask_index_list[lvl][i]
            index = index[index > 0]
            mask_gt_lvl = target[i][index - 1]
            mask_gt_lvl_list.append(mask_gt_lvl)
        target_gt_list.append(mask_gt_lvl_list)
    return target_gt_list


def mask_to_poly(mask):
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:
            polygons.append(contour)
    return polygons


def distance_sample_pts(gt_bboxes, gt_masks, cfg, num_pts):
    """
    Sample pts based on distance transformation map.

    Args:
        gt_bboxes(list(Tensor)): groud-truth bounding box
        gt_masks(list(Mask)): ground-truth mask
        cfg(dict): sampling config
        num_pts(int): number of points

    Returns:
        numpy: the sampling points based on distance transform map
    """
    dist_sample_thr = cfg.get('dist_sample_thr', 2)
    pts_list = []
    pts_label_list = []
    for i in range(len(gt_bboxes)):
        x1, y1, x2, y2 = gt_bboxes[i].cpu().numpy().astype(np.int32)
        w = np.maximum(x2 - x1, 1)
        h = np.maximum(y2 - y1, 1)
        mask = mmcv.imresize(gt_masks.masks[i][y1:y1 + h, x1:x1 + w],
                             (cfg.get('mask_size', 56), cfg.get('mask_size', 56)))
        polygons = mask_to_poly(mask)
        distance_map = np.ones(mask.shape).astype(np.uint8)
        for poly in polygons:
            poly = np.array(poly).astype(np.int)
            for j in range(len(poly) // 2):
                x_0, y_0 = poly[2 * j:2 * j + 2]
                if j == len(poly) // 2 - 1:
                    x_1, y_1 = poly[0:2]
                else:
                    x_1, y_1 = poly[2 * j + 2:2 * j + 4]
                cv2.line(distance_map, (x_0, y_0), (x_1, y_1), 0, thickness=2)
        roi_dist_map = cv2.distanceTransform(distance_map, cv2.DIST_L2, 3)
        con_index = np.stack(np.nonzero(roi_dist_map == 0)[::-1], axis=-1)
        roi_dist_map[roi_dist_map == 0] = 1
        roi_dist_map[roi_dist_map > dist_sample_thr] = 0

        index_y, index_x = np.nonzero(roi_dist_map > 0)
        index = np.stack([index_x, index_y], axis=-1)
        _len = index.shape[0]
        if len(con_index) == 0:
            pts = np.zeros([2 * num_pts])
        else:
            repeat = num_pts // _len
            mod = num_pts % _len
            perm = np.random.choice(_len, mod, replace=False)
            draw = [index.copy() for i in range(repeat)]
            draw.append(index[perm])
            draw = np.concatenate(draw, 0)
            draw = np.random.permutation(draw)
            draw = draw + np.random.rand(*draw.shape)
            x_scale = float(w) / cfg.get('mask_size', 56)
            y_scale = float(h) / cfg.get('mask_size', 56)
            draw[:, 0] = draw[:, 0] * x_scale + x1
            draw[:, 1] = draw[:, 1] * y_scale + y1
            pts = draw.reshape(2 * num_pts)

        pts_list.append(pts)
        pts_long = pts.astype(np.long)
        pts_label = gt_masks.masks[i][pts_long[1::2], pts_long[0::2]]
        pts_label_list.append(pts_label)
    pts_list = np.stack(pts_list, 0)
    return pts_list