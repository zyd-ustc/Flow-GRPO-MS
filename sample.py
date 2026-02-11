import os

import numpy as np
import mindspore as ms
from mindone.diffusers import StableDiffusion3Pipeline
from accelerate import Accelerator
from mindone.peft import PeftModel

# --- 配置 ---
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
# LORA_PATH = None
LORA_PATH = "/Users/zyd/Documents/2025/HUAWEI/RewardModel_MS/Flow-GRPO-MS/models/checkpoints/step_15000/backbone_lora"  # 填入 LoRA 路径或保持 None
PROMPT_FILE = "dataset/ocr/test.txt"
OUTPUT_DIR = "outputs"
# OUTPUT_DIR = "00_outputs/final/pickscore/base"
SEED = 42


def main():
    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    device = accelerator

    # 1. 加载模型
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID, 
        ms_dtype=ms.float16
    )

    if LORA_PATH and os.path.exists(LORA_PATH):
        print(f"Rank {rank} 加载 LoRA 权重: {LORA_PATH}")

        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, LORA_PATH)

        pipe.transformer.merge_and_unload()
        print(f"Rank {rank} LoRA 权重加载完成并合并。")
    else:
        print(f"Rank {rank} 未提供有效的 LoRA 路径，使用基础模型。")


    # 2. 读取所有提示词
    if not os.path.exists(PROMPT_FILE):
        return
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        all_prompts = [line.strip() for line in f.readlines() if line.strip()]

    # 3. 确保输出目录存在
    if rank == 0 and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    accelerator.wait_for_everyone()

    # 4. 根据 Rank 分配任务
    for idx, prompt in enumerate(all_prompts):
        # 核心逻辑：只处理属于当前 Rank 的索引
        if idx % world_size == rank:
            current_seed = SEED + idx
            generator = np.random.default_rng(current_seed)

            print(f"Rank {rank} 正在处理序号 {idx}: {prompt[:30]}...")
            
            image = pipe(
                prompt=prompt,
                num_inference_steps=40,
                guidance_scale=4.5,
                width=512,
                height=512,
                generator=generator
            ).images[0]

            # 保持原始序号作为文件名
            save_path = os.path.join(OUTPUT_DIR, f"{idx:06d}.png")
            image.save(save_path)

    accelerator.wait_for_everyone()
    if rank == 0:
        print("所有采样任务已完成。")

if __name__ == "__main__":
    main()