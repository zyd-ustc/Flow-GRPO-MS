#!/bin/sh
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ====== 可选：按你的环境覆盖这些变量 ======
# SD3 模型（可以是本地路径，也可以是 HuggingFace repo id）
SD3_PATH="${SD3_PATH:-stabilityai/stable-diffusion-3.5-medium}"

# （可选）若你本地 mindone 不在环境里，可用 PYTHONPATH 指向它
# export PYTHONPATH="/path/to/mindone:${PYTHONPATH}"

# 其它 scorer 可能用到的模型路径（按需设置）
# export QWEN_VL_PATH="/path/to/Qwen2.5-VL-7B-Instruct"
# export QWEN_VL_OCR_PATH="/path/to/Qwen2.5-VL-7B-Instruct"
# export CLIP_PATH="/path/to/clip-vit-large-patch14"
# export PICKSCORE_PATH="/path/to/PickScore_v1"
# export UNIFIED_REWARD_PATH="/path/to/UnifiedReward-qwen-7b"

# vLLM URLs
# export QWEN_VL_VLLM_URL="http://0.0.0.0:9529/v1"
# export QWEN_VL_OCR_VLLM_URL="http://0.0.0.0:9529/v1"
# export UNIFIED_REWARD_VLLM_URL="http://0.0.0.0:9529/v1"

export TOKENIZERS_PARALLELISM=False

# ====== 单卡训练（默认：jpeg 可压缩性 reward）======
msrun --worker_num 1 --local_worker_num 1 --master_port 9527 --join True "${ROOT_DIR}/scripts/train_sd3.py" \
    --reward jpeg-compressibility \
    --reward-weights 1.0 \
    --model "${SD3_PATH}" \
    --dataset "${ROOT_DIR}/dataset/ocr" \

# ====== 示例：使用 Diffusion-RM (SD3) ======
# 1) 先把 Diffusion-RM 的 PyTorch 权重转为 MindSpore ckpt（仅需做一次）
# python "${ROOT_DIR}/scripts/convert_diffusion_rm_weights.py" --checkpoint_dir "/path/to/Diffusion-RM/.../checkpoints/step_15000"
#
# 2) 开始训练（注意：checkpoint 目录里应包含 rm_head.ckpt / (可选) backbone_lora/adapter_model.ckpt）
# msrun --worker_num 1 --local_worker_num 1 --master_port 9527 --join True "${ROOT_DIR}/scripts/train_sd3.py" \
#     --reward diffusion-rm-sd3 \
#     --reward-weights 1.0 \
#     --model "${SD3_PATH}" \
#     --dataset "${ROOT_DIR}/dataset/ocr" \
#     --diffusion-rm-checkpoint-path "/path/to/Diffusion-RM/.../checkpoints/step_15000" \
#     --diffusion-rm-config-path "/path/to/Diffusion-RM/.../config.json" \
#     --diffusion-rm-u 0.9 \
#     --diffusion-rm-use-training-pipeline True \
