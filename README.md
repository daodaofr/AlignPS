
## Introduction

This is the implementationn for [Anchor-Free Person Search](https://arxiv.org/abs/2103.11617) in CVPR2021

![demo image](demo/arch.jpg)


## License

This project is released under the [Apache 2.0 license](LICENSE).


## Installation

Please refer to [install.md](docs/install.md) to install MMdetection.


## Dataset

Download [CUHK-SYSU](https://github.com/ShuangLI59/person_search) and [PRW](https://github.com/liangzheng06/PRW-baseline).

We provide coco-style annotation in [demo/anno](demo/anno).

For CUHK-SYSU, change the path of your dataset and the annotaion file in the [config file](configs/_base_/datasets/cuhk_detection_1000.py)

For PRW, change these config files: [config1](configs/fcos/prw_base_focal_labelnorm_sub_ldcn_fg15_wd1-3.py) [config2](configs/fcos/prw_dcn_base_focal_labelnorm_sub_ldcn_fg15_wd7-4.py)



## Experiments
  1. Train

   ```bash
   sh run_train.sh
   ```
  2. Test CUHK-SYSU

  Change the paths in L59 and L72 in [test_results.py](tools/test_results.py)

   ```bash
   sh run_test.sh
   ```
   Test PRW

   Change the paths in L127 and L128 in [test_results_prw.py](tools/test_results_prw.py)

   ```bash
   sh run_test.sh
   ```


## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@inproceedings{yan2021alignps,
  title={Anchor-Free Person Search},
  author={Yichao Yan, Jingpeng Li, Jie Qin, Song Bai, Shengcai Liao, Li Liu, Fan Zhu, Ling Shao},
  booktitle={CVPR},
  year={2021}
}
```

