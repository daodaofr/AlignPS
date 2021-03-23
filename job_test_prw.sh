#!/bin/bash
#SBATCH -n 40
#SBATCH --gres=gpu:v100:4
#SBATCH --time=48:00:00

END=24
# TESTPATH='prw_base_focal_labelnorm_sub_ldcn_fg15_wd1-3'
TESTPATH='prw_dcn_base_focal_labelnorm_sub_ldcn_fg15_wd7-4'
for ((i=20;i<=END;i+=1)); do
    echo $i
    ./tools/dist_test.sh ./configs/fcos/${TESTPATH}.py work_dirs/${TESTPATH}/epoch_${i}.pth 4 --out work_dirs/${TESTPATH}/results_1000.pkl
    echo '------------------------'
    python ./tools/test_results_prw.py ${TESTPATH}
    echo $i
done
