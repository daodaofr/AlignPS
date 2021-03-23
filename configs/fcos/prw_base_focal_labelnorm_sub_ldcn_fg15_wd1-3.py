_base_ = 'fcos_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid_1000.py'

model = dict(
    pretrained='open-mmlab://detectron2/resnet50_caffe',
    neck=dict(
        type='FPNDcnLconv3Dcn'),
    bbox_head=dict(
        type='FCOSReidHeadFocalOimSub',
        unlabel_weight=10,
        temperature=15,
        label_norm=True,
        num_person=483,
        queue_size=500,
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=True,
        center_sampling=True,
        conv_bias=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)))

# training and testing settings
test_cfg = dict(nms=dict(type='nms', iou_threshold=0.5))

# dataset settings
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', 
        img_scale=[(667, 400),(1000, 600), (1333, 800), (1500,900), (1666, 1000), (2000, 1200)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_ids']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        #img_scale=(1600, 900),
        img_scale=(1500, 900),
        #img_scale=(2000, 1000),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# change the path of the dataset
data_root = '/home/yy1/2021/data/prw/PRW-v16.04.20/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        ann_file=data_root + 'train_pid.json', # change the path of the annotation file
        img_prefix=data_root + 'frames/',
        pipeline=train_pipeline),
    val=dict(
        ann_file=data_root + 'test_pid.json',  # change the path of the annotation file
        img_prefix=data_root + 'frames/',
        pipeline=test_pipeline),
    test=dict(
        ann_file=data_root + 'test_pid.json',  # change the path of the annotation file
        img_prefix=data_root + 'frames/',
        proposal_file=data_root+'annotation/test/train_test/TestG50.mat',
        pipeline=test_pipeline)
)

optimizer_config = dict(_delete_=True, grad_clip=None)

# optimizer
optimizer = dict(
    lr=0.001, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.), weight_decay=0.001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22])
total_epochs = 24
