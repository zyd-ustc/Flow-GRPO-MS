# Flow-GRPO-MS

**Diffusion-RM** 作为主奖励模型，面向 SD3/FLUX 场景。

## 项目定位

- 使用 GRPO 思路对扩散模型策略进行强化学习微调。
- 在采样图像后，通过 Diffusion-RM 打分构造 reward/advantage。
- 对策略网络（通常是 transformer 或 LoRA 参数）进行迭代更新。

## 目录结构

```text
Flow-GRPO-MS/
├── flow_grpo/
│   ├── trainer/                     # GRPO 训练核心（pipeline/scheduler/loss）
│   ├── scorer/
│   │   ├── diffusion_rm.py          # Diffusion-RM scorer 入口
│   │   ├── diffusion_rm_impl/       # Diffusion-RM 推理实现
│   │   ├── multi.py                 # 多 scorer 聚合（当前仅保留 diffusion-rm-*）
│   │   └── scorer.py                # scorer 基类
│   ├── optim/                       # 优化器
│   ├── dataset.py                   # prompt 数据集与采样器
│   └── utils.py
├── scripts/
│   ├── train_sd3.py                 # 主训练入口
│   ├── train_diffusion_rm_sd3.sh    # 训练启动脚本（推荐）
│   ├── infer_sd3.py                 # 推理脚本（LoRA 可选）
│   └── convert_diffusion_rm_weights.py
├── config/
│   └── diffusion_rm_example.yaml    # Diffusion-RM 配置示例
├── dataset/
│   └── ocr/
│       ├── train.txt
│       └── test.txt
└── requirements.txt
```

## Diffusion-RM 原理（简版）

Diffusion-RM 的核心思想是：  
给定图像 latent 和文本条件，在某个噪声水平 `u` 对 latent 加噪后，由 reward model 预测质量分数，分数越高代表图文质量越好。

在本项目中：

1. 策略模型采样生成图像；
2. 将图像编码为 latent，并结合 prompt 编码；
3. 调用 Diffusion-RM 计算 reward；
4. 将 reward 转为 advantage；
5. 使用 GRPO/PPO-style 目标更新策略。

## 训练流程（Diffusion-RM-SD3）

### 1) 环境准备

```bash
python -m pip install -r requirements.txt
```
- `SD3_MODEL_PATH`：基础 SD3 模型路径
- `DIFFUSION_RM_CHECKPOINT_PATH`：Diffusion-RM checkpoint 目录
- `DIFFUSION_RM_CONFIG_PATH`：Diffusion-RM 配置文件路径

### 3) 启动训练

```bash
chmod +x scripts/train_diffusion_rm_sd3.sh

export SD3_MODEL_PATH=/path/to/stable-diffusion-3.5-medium
export DIFFUSION_RM_CHECKPOINT_PATH=/path/to/diffusion-rm/checkpoints/step_xxx
export DIFFUSION_RM_CONFIG_PATH=/path/to/diffusion-rm/config.json

scripts/train_diffusion_rm_sd3.sh
```

可选变量：

- `WORKER_NUM`（默认 `8`）
- `MASTER_PORT`（默认 `9527`）
- `DATASET_DIR`（默认 `dataset/ocr`）
- `OUTPUT_NAME`（默认 `diffusion-rm-sd3`）

## 推理流程（加载基础模型 + 可选 LoRA）

```bash
python scripts/infer_sd3.py \
  --model-id /path/to/stable-diffusion-3.5-medium \
  --lora-path /path/to/output/checkpoints/step_xxx/backbone_lora \
  --prompt-file dataset/ocr/test.txt \
  --output-dir outputs/infer \
  --num-inference-steps 40 \
  --guidance-scale 4.5 \
  --dtype fp16
```

如果不使用 LoRA，可省略 `--lora-path`。

## 关键训练参数说明

- `--reward diffusion-rm-sd3`：启用 SD3 版 Diffusion-RM scorer
- `--diffusion-rm-checkpoint-path`：reward model 权重目录
- `--diffusion-rm-config-path`：reward model 配置
- `--diffusion-rm-u`：打分噪声强度（常用 `0.9`）
- `--beta`：GRPO 中 KL 项系数（`0.0` 表示关闭 KL）

