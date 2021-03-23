_base_ = './reppoints_moment_r50_fpn_gn-neck+head_1x_cuhk_reid.py'
model = dict(bbox_head=dict(transform_method='minmax'))
