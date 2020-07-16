_base_ = [
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='DenseRepPointsDetector',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='DenseRepPointsHead',
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        stacked_mask_convs=7,
        fuse_mask_feat=True,
        num_group=9,
        num_score_group=121,
        num_points=729,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.5),
        loss_bbox_refine=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
        loss_pts_init=dict(type='ChamferLoss2D', use_cuda=True, loss_weight=0.5, eps=1e-12),
        loss_pts_refine=dict(type='ChamferLoss2D', use_cuda=True, loss_weight=1.0, eps=1e-12),
        loss_mask_score_init=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        transform_method='minmax'))
# training and testing settings
train_cfg = dict(
    init=dict(
        assigner=dict(type='PointAssignerV2', scale=4, pos_num=1),
        allowed_border=-1,
        pos_weight=-1,
        mask_size=56,
        dist_sample_thr=2,
        debug=False),
    refine=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        mask_size=56,
        dist_sample_thr=2,
        debug=False))
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.6),
    max_per_img=100)
optimizer = dict(lr=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2), _delete_=True)