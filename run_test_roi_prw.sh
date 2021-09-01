# PRW ROI-AlignPS
TESTPATH='faster_rcnn_r50_caffe_c4_1x_cuhk_single_two_stage17_6_nae1_prw'
TESTNAME='prw_roi_alignps.pth'

# Make sure the model path is work_dirs/TESTPATH/TESTNAME, tesults_1000.pkl will be saved in work_dirs/TESTPATH
./tools/dist_test_d.sh ./configs/person_search_prw/${TESTPATH}.py work_dirs/${TESTPATH}/epoch_${i}.pth 1 --out work_dirs/${TESTPATH}/results_1000.pkl
echo '------------------------'
python ./tools/test_results_psd_prw.py ${TESTPATH}
