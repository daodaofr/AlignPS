#!/bin/bash
#SBATCH -n 40
#SBATCH --gres=gpu:v100:4
#SBATCH --time=48:00:00

END=24
TESTPATH='fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_cuhk_reid_1500_stage1_fpncat_dcn_epoch24_multiscale_focal_x4_bg-2_lconv3dcn_sub_triqueue_dcn0'
for ((i=20;i<=END;i+=1)); do
    echo $i
    ./tools/dist_test.sh ./configs/fcos/${TESTPATH}.py work_dirs/${TESTPATH}/epoch_${i}.pth 4 --out work_dirs/${TESTPATH}/results_1000.pkl
    echo '------------------------'
    python ./tools/test_results.py ${TESTPATH}
    echo $TESTPATH
done
