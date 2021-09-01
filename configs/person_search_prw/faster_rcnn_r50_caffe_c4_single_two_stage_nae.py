# model settings
_base_ = [
    '../_base_/datasets/cuhk_detection_1000.py',
]

norm_cfg = dict(type="BN", requires_grad=False)
model = dict(
    type="SingleTwoStageDetector",
    pretrained="open-mmlab://detectron2/resnet50_caffe",
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    rpn_head=dict(
        type="RPNHead",
        in_channels=1024,
        feat_channels=1024,
        anchor_generator=dict(
            type="AnchorGenerator", scales=[2, 4, 8, 16, 32], ratios=[0.5, 1.0, 2.0], strides=[16]
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
    ),
    roi_head=dict(
        type="StandardRoIHead",
        shared_head=dict(
            type="ResLayer",
            depth=50,
            stage=3,
            stride=2,
            dilation=1,
            style="caffe",
            norm_cfg=norm_cfg,
            norm_eval=True,
        ),
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", out_size=14, sample_num=0),
            out_channels=1024,
            featmap_strides=[16],
        ),
        bbox_head=dict(
            type="BBoxHead",
            with_avg_pool=True,
            roi_feat_size=7,
            in_channels=2048,
            num_classes=80,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=False,
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type="L1Loss", loss_weight=1.0),
        ),
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
)
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1,
        ),
        sampler=dict(
            type="RandomSampler",
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False,
        ),
        allowed_border=0,
        pos_weight=-1,
        debug=False,
    ),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=12000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0,
    ),
    rcnn=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.5,
            neg_iou_thr=0.1,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1,
        ),
        sampler=dict(
            type="RandomSampler",
            num=128,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=True,
        ),
        pos_weight=-1,
        debug=False,
    ),
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False
)
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=6000,
        nms_post=300,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0,
    ),
    rcnn=dict(score_thr=0.05, nms=dict(type="nms", iou_thr=0.5), max_per_img=100),
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=100
)

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
        ann_file=data_root + 'test_pid.json', # change the path of the annotation file
        img_prefix=data_root + 'frames/',
        pipeline=test_pipeline),
    test=dict(
        ann_file=data_root + 'test_pid.json', # change the path of the annotation file
        img_prefix=data_root + 'frames/',
        proposal_file=data_root+'annotation/test/train_test/TestG50.mat',
        pipeline=test_pipeline)
)
