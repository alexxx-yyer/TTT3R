#!/bin/bash

set -e

# 切换到项目根目录
cd "$(dirname "$0")/../.."
workdir='.'
model_names=('ttt3r') # ttt3r cut3r
ckpt_name='cut3r_512_dpt_4_64'
model_weights="${workdir}/src/${ckpt_name}.pth"

# 数据集路径配置
export DATA_ROOT_7SCENES="/home/alex/dataset/7-scenes"
export DATA_ROOT_NRGBD="/home/alex/dataset/neural_rgbd_data"

# PyTorch 显存管理优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for model_name in "${model_names[@]}"; do

# for max_frames in 50 100 150 200 250 300 350 400
for max_frames in 100

do
    output_dir="${workdir}/eval_results/video_recon/7scenes_${max_frames}/${model_name}"
    echo "$output_dir"
    NCCL_TIMEOUT=360000 accelerate launch --num_processes 1 --main_process_port 29502 eval/mv_recon/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --model_name "$model_name" \
        --model_update_type "$model_name" \
        --max_frames $max_frames \

done
done
