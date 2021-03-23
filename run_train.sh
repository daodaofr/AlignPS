# CUHK-SYSU, AlignPS
python tools/train.py configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_cuhk_reid_1500_stage1_fpncat_dcn_epoch24_multiscale_focal_x4_bg-2_lconv3dcn_sub_triqueue_dcn0.py --gpu-ids 0 --no-validate

# CUHK-SYSU, AlignPS+
# python tools/train.py configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_cuhk_reid_1500_stage1_fpncat_dcn_epoch24_multiscale_focal_x4_bg-2_lconv3dcn_sub_triqueue.py --gpu-ids 0 --no-validate

# PRW, AlignPS
# python tools/train.py configs/fcos/prw_base_focal_labelnorm_sub_ldcn_fg15_wd1-3.py --gpu-ids 0 --no-validate

# PRW, AlignPS+
# python tools/train.py configs/fcos/prw_dcn_base_focal_labelnorm_sub_ldcn_fg15_wd7-4.py --gpu-ids 0 --no-validate