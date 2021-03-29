
# CUHK-SUSU AlignPS
TESTPATH='fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_cuhk_reid_1500_stage1_fpncat_dcn_epoch24_multiscale_focal_x4_bg-2_lconv3dcn_sub_triqueue_dcn0'
TESTNAME='cuhk_alignps.pth'

# CUHK-SUSU AlignPS+
# TESTPATH='fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_cuhk_reid_1500_stage1_fpncat_dcn_epoch24_multiscale_focal_x4_bg-2_lconv3dcn_sub_triqueue'
# TESTNAME='cuhk_alignps_plus.pth'

# Make sure the model path is work_dirs/TESTPATH/TESTNAME, tesults_1000.pkl will be saved in work_dirs/TESTPATH
./tools/dist_test.sh ./configs/fcos/${TESTPATH}.py work_dirs/${TESTPATH}/${TESTNAME} 1 --out work_dirs/${TESTPATH}/results_1000.pkl
echo '------------------------'
python ./tools/test_results.py ${TESTPATH}
echo $TESTPATH
