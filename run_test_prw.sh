#!/bin/bash

# PRW AlignPS
# TESTPATH='prw_base_focal_labelnorm_sub_ldcn_fg15_wd1-3'
# TESTNAME='prw_alignps.pth'

#PRW AlignPS+
TESTPATH='prw_dcn_base_focal_labelnorm_sub_ldcn_fg15_wd7-4'
TESTNAME='prw_alignps_plus.pth'

i=24
./tools/dist_test.sh ./configs/fcos/${TESTPATH}.py work_dirs/${TESTPATH}/${TESTNAME} 1 --out work_dirs/${TESTPATH}/results_1000.pkl
echo '------------------------'
python ./tools/test_results_prw.py ${TESTPATH}


