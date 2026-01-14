# Diffusion-RM 推理接口集成

本文档说明如何在 Flow-GRPO-MS 中使用 Diffusion-RM 的预训练权重进行推理评分。

## 概述

本实现提供了一个轻量级的解决方案，通过在 MindSpore 环境中调用 PyTorch 实现的 Diffusion-RM，实现推理接口的集成。

**主要特点**:
- ✅ 仅实现推理接口，不涉及训练逻辑
- ✅ 支持加载 Diffusion-RM 的预训练 checkpoint
- ✅ 集成到 Flow-GRPO-MS 的现有 scorer 框架
- ✅ 支持 FLUX 和 SD3 两种模型
- ✅ 可配置噪声水平 u

## 文件结构

```
Flow-GRPO-MS/
├── flow_grpo/scorer/
│   ├── diffusion_rm.py              # Diffusion-RM scorer 实现 (新增)
│   ├── diffusion_rm_impl/           # Diffusion-RM 推理最小实现 (vendor, 新增)
│   │   ├── inferencer.py
│   │   └── models/
│   └── multi.py                    # MultiScorer (已更新)
├── config/
│   └── diffusion_rm_example.yaml    # 配置示例 (新增)
├── scripts/
│   └── test_diffusion_rm_scorer.py # 测试脚本 (新增)
└── INFERENCE_ONLY_PLAN.md          # 详细方案文档
```

## 快速开始

### 1. 安装依赖

```bash
# 安装 Flow-GRPO-MS
cd Flow-GRPO-MS
pip install -e .
```

说明：`Flow-GRPO-MS` 已内置（vendor）Diffusion-RM 推理所需的最小实现，不再依赖外部 `Diffusion-RM` Python 包；你只需要提供 Diffusion-RM 训练产物的 `checkpoint_path` 与 `config_path`。

### 2. 准备 Checkpoint

确保 Diffusion-RM 的 checkpoint 文件结构正确：

```
Diffusion-RM/outputs/
└── [NODE-00 HPDv3]-.../
    ├── step_15000/
    │   ├── backbone_lora/
    │   │   ├── adapter_config.json
    │   │   └── adapter_model.safetensors
    │   └── rm_head.pt
    └── config.json
```

### 3. 使用 DiffusionRMScorer

#### 单独使用

```python
from flow_grpo.scorer.diffusion_rm import DiffusionRMFluxScorer
from PIL import Image

# 初始化 scorer
scorer = DiffusionRMFluxScorer(
    checkpoint_path="../Diffusion-RM/outputs/.../checkpoints/step_15000",
    config_path="../Diffusion-RM/outputs/.../config.json",
    pipeline_path="black-forest-labs/FLUX.1-dev",
    device="cuda",
    u=0.9,
)

# 评分图像
images = [Image.open("image1.jpg"), Image.open("image2.jpg")]
prompts = ["A beautiful landscape", "A portrait"]

scores = scorer(images, prompts)
print(scores)  # [0.85, 0.72]
```

#### 在 MultiScorer 中使用

```python
from flow_grpo.scorer.multi import MultiScorer
from flow_grpo.scorer.diffusion_rm import DiffusionRMFluxScorer

# 创建 DiffusionRMScorer 实例
diffusion_rm_scorer = DiffusionRMFluxScorer(
    checkpoint_path="../Diffusion-RM/outputs/.../checkpoints/step_15000",
    config_path="../Diffusion-RM/outputs/.../config.json",
    pipeline_path="black-forest-labs/FLUX.1-dev",
    device="cuda",
    u=0.9,
)

# 配置 MultiScorer
scorers = {
    "diffusion-rm-flux": 0.6,    # Diffusion-RM 权重 60%
    "aesthetic": 0.2,             # Aesthetic 20%
    "pickscore": 0.2,             # PickScore 20%
}

scorer_configs = {
    "diffusion-rm-flux": diffusion_rm_scorer,
}

multi_scorer = MultiScorer(scorers, scorer_configs)

# 评分
score_details = multi_scorer(images, prompts)
print(score_details)
# {
#     "diffusion-rm-flux": [0.85, 0.72],
#     "aesthetic": [0.78, 0.65],
#     "pickscore": [0.82, 0.70],
#     "avg": [0.822, 0.702]
# }
```

#### 直接对 Latents 评分

```python
import mindspore as ms
from flow_grpo.scorer.diffusion_rm import DiffusionRMFluxScorer

scorer = DiffusionRMFluxScorer(
    checkpoint_path="../Diffusion-RM/outputs/.../checkpoints/step_15000",
    config_path="../Diffusion-RM/outputs/.../config.json",
    pipeline_path="black-forest-labs/FLUX.1-dev",
    device="cuda",
    u=0.9,
)

# 准备 latents (例如从 VAE 编码得到)
latents = ms.randn(2, 16, 64, 64, dtype=ms.float32)

# 准备 text conditions
text_conds = {
    'encoder_hidden_states': prompt_embeds,      # [B, seq_len, hidden_dim]
    'pooled_projections': pooled_prompt_embeds,  # [B, hidden_dim]
}

# 评分
rewards = scorer.reward(latents, text_conds, u=0.9)
print(rewards)  # [[0.85], [0.72]]
```

### 4. 在 Flow-GRPO 训练中使用

```python
from flow_grpo.trainer.trainer import Trainer
from flow_grpo.scorer.multi import MultiScorer
from flow_grpo.scorer.diffusion_rm import DiffusionRMFluxScorer

# 创建 DiffusionRMScorer
diffusion_rm_scorer = DiffusionRMFluxScorer(
    checkpoint_path="../Diffusion-RM/outputs/.../checkpoints/step_15000",
    config_path="../Diffusion-RM/outputs/.../config.json",
    pipeline_path="black-forest-labs/FLUX.1-dev",
    device="cuda",
    u=0.9,
)

# 创建 MultiScorer
scorers = {
    "diffusion-rm-flux": 1.0,  # 只使用 Diffusion-RM
}
scorer_configs = {
    "diffusion-rm-flux": diffusion_rm_scorer,
}
multi_scorer = MultiScorer(scorers, scorer_configs)

# 训练
trainer = Trainer(
    model=model,
    scorer=multi_scorer,
    # ... 其他参数
)

trainer.train()
```

## 配置参数

### DiffusionRMFluxScorer / DiffusionRMSD3Scorer

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| checkpoint_path | str | 必填 | Diffusion-RM checkpoint 目录路径 |
| config_path | str | 必填 | Diffusion-RM 配置文件路径 |
| pipeline | object | None | Diffusers Pipeline (如果不提供会自动加载) |
| pipeline_path | str | "black-forest-labs/FLUX.1-dev" | Pipeline 模型路径 |
| device | str | "cuda" | 计算设备 ("cuda" 或 "cpu") |
| u | float | 0.9 | 噪声水平 (0.0-1.0) |

### MultiScorer

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| scorers | Dict[str, float] | 必填 | Scorer 名称和权重 |
| scorer_configs | Dict[str, Dict] | {} | Scorer 配置 (用于 DiffusionRMScorer) |

## 可用的 Scorer 类型

- `aesthetic`: AestheticScorer
- `jpeg-compressibility`: JpegCompressibilityScorer
- `jpeg-imcompressibility`: JpegImcompressibilityScorer
- `pickscore`: PickScoreScorer
- `qwenvl`: QwenVLScorer
- `qwenvl-ocr-vllm`: QwenVLOCRVLLMScorer
- `qwenvl-vllm`: QwenVLVLLMScorer
- `unified-reward-vllm`: UnifiedRewardVLLMScorer
- `diffusion-rm-flux`: DiffusionRMFluxScorer (新增)
- `diffusion-rm-sd3`: DiffusionRMSD3Scorer (新增)

## 运行测试

```bash
cd Flow-GRPO-MS

# 运行测试脚本（需要修改脚本中的 checkpoint 路径）
python scripts/test_diffusion_rm_scorer.py
```

## 注意事项

### 1. 内存占用

由于需要加载完整的 pipeline（包括 VAE 和 text encoders），内存占用较大：

- FLUX Pipeline: 约 24GB (FP16)
- SD3 Pipeline: 约 16GB (FP16)
- Diffusion-RM: 约 2GB

**建议**: 在 GPU 内存不足时，可以：
- 使用 `device="cpu"` 进行评分（速度较慢）
- 使用模型量化
- 使用多 GPU 分摊负载

### 2. 框架兼容性

本实现同时使用 PyTorch 和 MindSpore：

- Diffusion-RM 模型使用 PyTorch
- Flow-GRPO-MS 使用 MindSpore
- 数据通过 numpy 进行转换

这会带来轻微的性能开销，但确保了与原始 Diffusion-RM 的完全兼容。

### 3. Checkpoint 路径

确保 checkpoint 路径正确，且包含以下文件：
- `rm_head.pt` (必需)
- `backbone_lora/` (如果使用 LoRA)
- `full_model.pt` (如果未使用 LoRA 且未冻结 backbone)

### 4. Pipeline 路径

确保 pipeline 模型路径正确：
- FLUX: `"black-forest-labs/FLUX.1-dev"` 或其他 FLUX 模型
- SD3: `"stabilityai/stable-diffusion-3.5-medium"` 或其他 SD3 模型

## 性能优化建议

### 1. 预加载 Pipeline

如果在多个 scorer 中使用相同的 pipeline，可以预加载：

```python
from diffusers import FluxPipeline

# 预加载 pipeline
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
pipeline.to("cuda")

# 在多个 scorer 中复用
scorer1 = DiffusionRMFluxScorer(
    checkpoint_path=checkpoint_path1,
    config_path=config_path1,
    pipeline=pipeline,  # 复用
)
scorer2 = DiffusionRMFluxScorer(
    checkpoint_path=checkpoint_path2,
    config_path=config_path2,
    pipeline=pipeline,  # 复用
)
```

### 2. 批处理评分

尽可能使用批处理以提高效率：

```python
# 推荐: 批处理
scores = scorer(images_batch, prompts_batch)

# 不推荐: 单张处理
for img, prompt in zip(images, prompts):
    score = scorer([img], [prompt])
```

### 3. 缓存 Text Embeddings

如果多次使用相同的 prompts，可以缓存 text embeddings：

```python
# 缓存 text embeddings
text_conds = scorer._encode_prompts(prompts)

# 对多个 latents 评分
for latents in latents_list:
    rewards = scorer.reward(latents, text_conds)
```

## 常见问题

### Q1: 如何选择噪声水平 u？

**A**: u 是噪声水平，通常在 0.5-0.99 之间：

- `u=0.99`: 接近原图，噪声很少
- `u=0.9`: 常用值，平衡质量和噪声
- `u=0.5`: 高噪声，测试模型的鲁棒性

### Q2: 能否使用 CPU 进行评分？

**A**: 可以，但速度会很慢：

```python
scorer = DiffusionRMFluxScorer(
    checkpoint_path=checkpoint_path,
    config_path=config_path,
    pipeline_path="black-forest-labs/FLUX.1-dev",
    device="cpu",  # 使用 CPU
)
```

### Q3: 如何处理不同的 checkpoint 格式？

**A**: Diffusion-RM 支持三种模式：

1. LoRA + frozen backbone: 需要 `backbone_lora/` 和 `rm_head.pt`
2. 全参数训练: 需要 `full_model.pt`
3. 只训练 reward head: 需要 `rm_head.pt`

### Q4: 如何调试评分结果？

**A**: 可以单独测试每个 scorer：

```python
from flow_grpo.scorer.diffusion_rm import DiffusionRMFluxScorer

scorer = DiffusionRMFluxScorer(
    checkpoint_path=checkpoint_path,
    config_path=config_path,
    pipeline_path="black-forest-labs/FLUX.1-dev",
    device="cuda",
)

# 测试单张图像
scores = scorer([image], [prompt])
print(f"Score: {scores[0]}")
```

## 参考文献

- [Diffusion-RM 原始实现](../Diffusion-RM/)
- [Flow-GRPO-MS 原始实现](../Flow-GRPO-MS/)
- [Flow-GRPO 原始实现](../flow-grpo-rm/)
- [详细方案文档](../INFERENCE_ONLY_PLAN.md)

## 联系方式

如有问题或建议，请提 issue 或联系维护者。
