_base_ = './reppoints_v2_r50_fpn_giou_mstrain_2x_coco.py'
model = dict(
    pretrained='https://cloudstor.aarnet.edu.au/plus/s/xtixKaxLWmbcyf7/download#mobilenet_v2-ecbe2b5.pth',
    backbone=dict(
        _delete_=True,
        type='MobileNetV2',
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN')),
    neck=dict(
        in_channels=[24, 32, 96, 320]))
