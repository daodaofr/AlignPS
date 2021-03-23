_base_ = './fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_mot.py'
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w18',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(18, 36)),
            stage3=dict(num_channels=(18, 36, 72)),
            stage4=dict(num_channels=(18, 36, 72, 144)))),
    neck=dict(type='HRFPN', in_channels=[18, 36, 72, 144], out_channels=256))
#load_from = "/raid/yy1/mmdetection/checkpoints/fcos_hrnetv2p_w18_coco_pretrained_weights_classes_1.pth"
#resume_from = '/raid/yy1/mmdetection/work_dirs/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_mot/epoch_19.pth'
