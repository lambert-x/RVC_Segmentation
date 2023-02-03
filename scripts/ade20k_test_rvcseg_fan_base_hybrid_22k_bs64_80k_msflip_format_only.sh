#!/user/bin/env bash
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FAN/blob/main/LICENSE

cd /saccadenet/jfxiao/codes/FAN-OOD/segmentation

source /opt/conda/etc/profile.d/conda.sh
conda activate /saccadenet/jfxiao/conda_envs/fan_seg

GPU_NUM=8
DATASET_ABBR='ade20k'
DATASET_NAME='ade20k-151'
set -e
set -x


bash tools/dist_test.sh \
"local_configs/fan/rvc_seg_test/fan_hybrid_base_22k.1080.rvcseg_test_${DATASET_ABBR}.40k.py" \
"./work_dirs/exp10_rvcseg_balanced_except_kitti_add_bdd_idd_fan_base_hybrid_22k_bs64_80k_resume/latest.pth" \
${GPU_NUM} \
--format-only \
--options data.test.img_dir="data/rvc_seg_test/${DATASET_NAME}/" \
data.test.rvc_seg_mapping_tsv="/saccadenet/jfxiao/codes/rvc_devkit/segmentation/ss_mapping_uint8_new.csv" \
--eval-options "imgfile_prefix=./work_dirs/rvc_seg_test_ALL-KITTI+BDD+IDD_80k/uint8_results/${DATASET_NAME}" \
--aug-test


