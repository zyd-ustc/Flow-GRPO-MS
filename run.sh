#!/bin/sh
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

SD3_PATH="/home/zyd/Reward/Flow-GRPO-MS/models/stable-diffusion-3.5-medium"


# export QWEN_VL_PATH="/path/to/Qwen2.5-VL-7B-Instruct"
# export QWEN_VL_OCR_PATH="/path/to/Qwen2.5-VL-7B-Instruct"
# export CLIP_PATH="/path/to/clip-vit-large-patch14"
# export PICKSCORE_PATH="/path/to/PickScore_v1"
# export UNIFIED_REWARD_PATH="/path/to/UnifiedReward-qwen-7b"


# export QWEN_VL_VLLM_URL="http://0.0.0.0:9529/v1"
# export QWEN_VL_OCR_VLLM_URL="http://0.0.0.0:9529/v1"
# export UNIFIED_REWARD_VLLM_URL="http://0.0.0.0:9529/v1"

export TOKENIZERS_PARALLELISM=False

msrun --worker_num 8 --local_worker_num 8 --master_port 9527 --join True "${ROOT_DIR}/scripts/train_sd3.py" \
    --reward diffusion-rm-sd3 \
    --reward-weights 1.0 \
    --model "${SD3_PATH}" \
    --dataset "${ROOT_DIR}/dataset/ocr" \
    --guidance-scale 1.0 \
    --train-batch-size 1 \
    --num-image-per-prompt 1 \
    --test-batch-size 1 \
    --beta 0.0 \
    --diffusion-rm-checkpoint-path "/home/zyd/Reward/Flow-GRPO-MS/models/epoch_001" \
    --diffusion-rm-config-path "/home/zyd/Reward/Flow-GRPO-MS/models/epoch_001/config.json" \
    --diffusion-rm-u 0.9 \
