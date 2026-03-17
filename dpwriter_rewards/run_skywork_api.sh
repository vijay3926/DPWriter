#!/bin/bash

LOG_DIR="./logs"
mkdir -p $LOG_DIR

GPUS=(0 1 2 3)  # 修改这里来指定具体的GPU卡号
NUM_GPUS=${#GPUS[@]}

NUM_GPUS=4
unset http_proxy https_proxy all_proxy
for (( i=0; i<NUM_GPUS; i++ )); do
    gpu_id=${GPUS[i]}
    port=$((18000+i))
    echo "Starting server on port $port with GPU: $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id python -m sglang.launch_server \
        --model-path /YOUR_PATH/Skywork-Reward-V2-Llama-3.1-8B \
        --mem-fraction-static 0.8 \
        --tp 1 \
        --host 0.0.0.0 \
        --port $port \
        --context-length 16384 \
        --is-embedding \
        &
done

