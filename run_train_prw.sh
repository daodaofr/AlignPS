nohup python -u tools/train.py configs/fcos/prw_dcn_base_focal.py --gpu-ids 3 >work_dirs/prw_focal.txt 2>&1 &
#nohup python -u tools/train.py configs/fcos/prw_dcn_newoim_focal.py --gpu-ids 4 >work_dirs/prw_newoim_focal.txt 2>&1 &
#nohup python -u tools/train.py configs/fcos/prw_dcn_newoimsmr_focal.py --gpu-ids 3 >work_dirs/prw_newoimsmr_focal.txt 2>&1 &
#nohup python -u tools/train.py configs/fcos/prw_dcn_newoimsmr_focal_epoch12.py --gpu-ids 1 >work_dirs/prw_newoimsmr_focal_epoch12.txt 2>&1 &
#nohup python -u tools/train.py configs/fcos/prw_dcn_newoim_focal_epoch12.py --gpu-ids 0 >work_dirs/prw_newoim_focal_epoch12.txt 2>&1 &


