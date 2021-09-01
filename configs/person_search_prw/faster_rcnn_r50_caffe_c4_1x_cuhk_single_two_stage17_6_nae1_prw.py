_base_ = [
    "./faster_rcnn_r50_caffe_c4_single_two_stage_nae.py",
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type="SingleTwoStageDetector176PRW",
    roi_head=dict(
        type='PersonSearchRoIHead2Input1',
        bbox_head=dict(
            type="PersonSearchNormAwareNewoim2InputBNBBoxHeadPRW", 
            num_classes=1,
            loss_bbox=dict(type="L1Loss", loss_weight=10.0),
        )),
    neck=dict(
        type='FPNDcnLconv3Dcn'),
    bbox_head=dict(
        type='FCOSReidHeadFocalSubTriQueue3PRW',
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=True,
        center_sampling=True,
        conv_bias=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)))
# resume_from='work_dirs/faster_rcnn_r50_caffe_c4_1x_cuhk_single_two_stage/latest.pth'
# schedule settings
# optimizer
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
        img_scale=(1500, 900),
        #img_scale=(1333, 800),
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
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
optimizer_config = dict(_delete_=True, grad_clip=None)

# optimizer
optimizer = dict(type="SGD", lr=0.0015, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=10, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1141,
    warmup_ratio=1.0 / 200,
    step=[16, 22])
total_epochs = 24

# optimizer = dict(type="SGD", lr=0.000625, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# # learning policy
# # actual epoch = 3 * 3 = 9
# lr_config = dict(policy="step", step=[3])
# # runtime settings
# total_epochs = 4  # actual epoch = 4 * 3 = 12
