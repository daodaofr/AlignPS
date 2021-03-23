#!/bin/bash

# PRW AlignPS
# TESTPATH='prw_base_focal_labelnorm_sub_ldcn_fg15_wd1-3'

#PRW AlignPS+
TESTPATH='prw_dcn_base_focal_labelnorm_sub_ldcn_fg15_wd7-4'

i=24
./tools/dist_test.sh ./configs/fcos/${TESTPATH}.py work_dirs/${TESTPATH}/epoch_${i}.pth 1 --out work_dirs/${TESTPATH}/results_1000.pkl
echo '------------------------'
python ./tools/test_results_prw.py ${TESTPATH}


