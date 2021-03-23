#./tools/dist_test.sh configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_1333.py \
#        work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_1333/epoch_12.pth \
#        4 --out work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_1333/results.pkl \
#        --eval bbox \
#./tools/dist_test.sh configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk.py \
#        work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk/epoch_12.pth \
#        4 --out work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk/results.pkl \
#        --eval bbox 
#./tools/dist_test.sh configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_1000.py \
#        work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_1000/epoch_12.pth \
#        2 --out work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_1000/results_1000.pkl \
#        --eval bbox 
#./tools/dist_test.sh configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid.py \
#        work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid/epoch_12.pth \
#        2 --out work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid/results.pkl \
#        --eval bbox 
#./tools/dist_test.sh configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid_1000.py \
#        work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid_1000/epoch_12.pth \
#        2 --out work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid_1000/results_1000.pkl \
#        --eval bbox 
#./tools/dist_test.sh configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid_1000_tri.py \
#work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid_1000_tri/epoch_24.pth \
#2 --out work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid_1000_tri/results_1000.pkl \
#--eval bbox 
#./tools/dist_test.sh configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid_1000_softmax.py \
#work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid_1000_softmax/epoch_12.pth \
#2 --out work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid_1000_softmax/results_1000.pkl \
#--eval bbox
#./tools/dist_test.sh configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid_1000_fpncat.py \
#work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid_1000_fpncat/epoch_12.pth \
#2 --out work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid_1000_fpncat/results_1000.pkl \
#--eval bbox
#./tools/dist_test.sh configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_cuhk_reid_1500_stage1_fpncat_dcn_epoch24_multiscale_lr0008_stage12.py \
#work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_cuhk_reid_1500_stage1_fpncat_dcn_epoch24_multiscale_lr0008_stage12/epoch_24.pth \
#8 --out work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_cuhk_reid_1500_stage1_fpncat_dcn_epoch24_multiscale_lr0008_stage12/results_1000.pkl \
#--eval bbox
#./tools/dist_test.sh configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid_normaware_1000.py \
#        work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid_normaware_1000/epoch_12.pth \
#        2 --out work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid_normaware_1000/results_na.pkl \
#        --eval bbox 
#./tools/dist_test.sh configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_mot_reid_coco.py \
#        work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_mot_reid_coco/epoch_30.pth \
#        2 --out work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_mot_reid_coco/results_mot_coco_1.pkl \
#        --eval bbox 
#./tools/dist_test.sh configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_mot.py \
#        work_dirs/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_mot/epoch_12.pth \
#        2 --out work_dirs/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_mot/results_x101_mot.pkl \
#        --eval bbox 
#./tools/dist_test.sh configs/hrnet/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_mot.py \
#        work_dirs/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_mot/epoch_19.pth \
#        2 --out work_dirs/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_mot/results_hrnet_mot.pkl \
#        --eval bbox 
#./tools/dist_test.sh configs/hrnet/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_mot_detection.py \
#        work_dirs/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_mot_detection/epoch_24.pth \
#        2 --out work_dirs/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_mot_detection/results_hrnet_mot_detection.pkl \
#        --eval bbox 
#./tools/dist_test.sh configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_mot_reid_coco_softmax.py \
#work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_mot_reid_coco_softmax/epoch_30.pth \
#2 --out work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_mot_reid_coco_softmax/results_mot_coco_softmax.pkl \
#--eval bbox 
#./tools/dist_test.sh configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid_roi_1000.py \
#        work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid_roi_1000/epoch_12.pth \
#        2 --out work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid_roi_1000/results_roi_query.pkl \
#        --eval bbox 
#python tools/test.py configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_1333.py \
#        work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_1333/epoch_12.pth \
#        --show-dir work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_1333/results
#python tools/test.py configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk.py \
#        work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk/epoch_12.pth \
#        --show-dir work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk/results
#python tools/test.py configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid.py \
#        work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid/epoch_12.pth \
#        --eval bbox
#python tools/test.py configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk.py \
#        work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk/epoch_12.pth \
#        --show-dir work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk/results
python tools/test.py configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_cuhk_reid_1500_stage1_fpncat_dcn_epoch24_multiscale_focal_x4_bg-2_lconv3dcn_sub_triqueue_dcn0.py \
work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_cuhk_reid_1500_stage1_fpncat_dcn_epoch24_multiscale_focal_x4_bg-2_lconv3dcn_sub_triqueue_dcn0/epoch_24.pth \
--out work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_cuhk_reid_1500_stage1_fpncat_dcn_epoch24_multiscale_focal_x4_bg-2_lconv3dcn_sub_triqueue_dcn0/results_1000.pkl \
--eval bbox