_base_ = './reppoints_minmax_r50_fpn_1x_coco.py'
model = dict(bbox_head=dict(transform_method='exact_minmax'))
