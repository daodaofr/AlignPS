_base_ = './reppoints_moment_r50_fpn_1x_cuhk_reid.py'
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(neck=dict(norm_cfg=norm_cfg), bbox_head=dict(norm_cfg=norm_cfg))
optimizer = dict(lr=0.0001)
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=10,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
total_epochs = 12
