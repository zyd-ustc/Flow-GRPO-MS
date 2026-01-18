import os
from typing import Optional

import mindspore as ms
from mindspore import mint
from .config_utils import load_config
from .models.sd3_rm import SD3RewardModel
from .models.flux_rm import FLUXRewardModel


def get_timesteps_from_u(noise_scheduler, u: ms.Tensor, n_dim: int = 4):
    # Keep consistent with upstream implementation: index = u * num_train_timesteps
    indices = (u * noise_scheduler.config.num_train_timesteps).long()
    timesteps = noise_scheduler.timesteps[indices]

    sigmas = noise_scheduler.sigmas[indices]
    while len(sigmas.shape) < n_dim:
        sigmas = sigmas[:, None]
    return sigmas, timesteps


class DRMInferencer:
    """Self-contained inference engine for Diffusion-RM checkpoints."""

    def __init__(
        self,
        pipeline,
        config_path: str,
        model_path: str,
        model_dtype: ms.dtype = ms.bfloat16,
    ):
        self.config = load_config(config_path)
        self.model_dtype = model_dtype


        backbone_id = ""
        try:
            backbone_id = str(self.config.model.backbone_model_id).lower()
        except Exception:
            backbone_id = ""
        pipe_cls = type(pipeline).__name__.lower()
        is_flux = ("flux" in backbone_id) or ("flux" in pipe_cls)

        if is_flux:
            vae_scale_factor = getattr(pipeline, "vae_scale_factor", None)
            if vae_scale_factor is None:
                # best-effort fallback
                vae_scale_factor = getattr(getattr(pipeline, "vae", None), "vae_scale_factor", 8)
            self.model = FLUXRewardModel(
                pipeline=pipeline,
                config_model=self.config.model,
                dtype=model_dtype,
                vae_scale_factor=vae_scale_factor,
            )
        else:
            self.model = SD3RewardModel(
                pipeline=pipeline,
                config_model=self.config.model,
                dtype=model_dtype,
            )

        # copy scheduler config
        scheduler_cls = type(pipeline.scheduler)
        self.noise_scheduler = scheduler_cls.from_config(pipeline.scheduler.config)

        self.add_noise = True
        try:
            if "training" in self.config and "add_noise" in self.config.training:
                self.add_noise = bool(self.config.training.add_noise)
        except Exception:
            self.add_noise = True

        self.load_checkpoint(model_path)

    def reward(self, text_conds, latents: ms.Tensor, u: float = 0.9) -> ms.Tensor:
        #self.model.eval()

        # switch to rm_lora if available
        ori_adapter = getattr(self.model.backbone, "active_adapter", None)
        if hasattr(self.model.backbone, "set_adapter"):
            self.model.backbone.set_adapter("rm_lora")

        bsz = latents.shape[0]

        u_tensor = ms.tensor([u] * bsz)
        sigmas, timesteps = get_timesteps_from_u(self.noise_scheduler, u_tensor, n_dim=len(latents.shape))

        noise = mint.randn_like(latents)
        if self.add_noise:
            noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
        else:
            noisy_model_input = latents

        noisy_model_input = noisy_model_input.to(self.model_dtype)

        score = self.model(
            latents=noisy_model_input,
            timesteps=timesteps,
            **text_conds,
        )

        # restore adapter
        if ori_adapter is not None and hasattr(self.model.backbone, "set_adapter"):
            self.model.backbone.set_adapter(ori_adapter)

        return score

    def load_checkpoint(self, checkpoint_path: str):
        # NOTE:
        # Diffusion-RM 原始训练产物通常是 PyTorch 的 .pt/.safetensors。
        # 这里推理侧 **只加载 MindSpore ckpt**（需要提前用脚本转换一次）。
        #
        # 转换脚本会尽量保持参数 key 不变，推理侧按 name 匹配加载；
        # 未匹配到的 key 会被忽略（通常是训练侧冗余信息或不同 backbone 结构导致）。

        def _load_ms_ckpt_into_net(net: ms.nn.Cell, ckpt_path: str):
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"MindSpore checkpoint not found: {ckpt_path}. "
                    f"Please convert Diffusion-RM weights to .ckpt first."
                )

            ckpt = ms.load_checkpoint(ckpt_path)
            net_params = net.parameters_dict()
            filtered = {k: v for k, v in ckpt.items() if k in net_params}
            not_loaded = [k for k in ckpt.keys() if k not in net_params]
            if not_loaded:
                # best-effort: ignore unmatched keys
                print(
                    f"[Diffusion-RM] Warning: {len(not_loaded)} keys in {os.path.basename(ckpt_path)} "
                    "do not match current network and were ignored."
                )

            ms.load_param_into_net(net, filtered, strict_load=False)
            print(
                f"[Diffusion-RM] load {os.path.basename(ckpt_path)}: "
                f"ckpt={len(ckpt)} params, net={len(net_params)} params, matched={len(filtered)}"
            )
            if len(filtered) == 0:
                print(f"[Diffusion-RM] matched=0, ckpt key sample: {list(ckpt.keys())[:20]}")
                print(f"[Diffusion-RM] net  key sample: {list(net_params.keys())[:20]}")
            return ckpt, net_params, filtered, not_loaded

        if getattr(self.config.model, "use_lora", False):
            # 1) backbone LoRA weights
            lora_ckpt = os.path.join(checkpoint_path, "backbone_lora", "adapter_model.ckpt")
            if os.path.exists(lora_ckpt):
                _load_ms_ckpt_into_net(self.model.backbone, lora_ckpt)
            else:
                # allow running without LoRA weights if user didn't provide them
                lora_dir = os.path.join(checkpoint_path, "backbone_lora")
                if os.path.exists(lora_dir):
                    raise FileNotFoundError(
                        f"Found LoRA dir {lora_dir} but missing MindSpore file {lora_ckpt}. "
                        "Please convert adapter_model.safetensors to adapter_model.ckpt."
                    )

            # 2) reward head
            rm_head_ckpt = os.path.join(checkpoint_path, "rm_head.ckpt")
            ckpt, _, filtered, _ = _load_ms_ckpt_into_net(self.model.reward_head, rm_head_ckpt)
            if len(filtered) == 0:
                raise RuntimeError(
                    "[Diffusion-RM] rm_head.ckpt does not match current reward head. "
                    "This repository only supports Diffusion-RM v3 (QFormer-style) reward head. "
                    f"ckpt key sample: {list(ckpt.keys())[:20]}"
                )

        elif not getattr(self.config.model, "freeze_backbone", False):
            full_model_ckpt = os.path.join(checkpoint_path, "full_model.ckpt")
            _load_ms_ckpt_into_net(self.model, full_model_ckpt)

        else:
            rm_head_ckpt = os.path.join(checkpoint_path, "rm_head.ckpt")
            ckpt, _, filtered, _ = _load_ms_ckpt_into_net(self.model.reward_head, rm_head_ckpt)
            if len(filtered) == 0:
                raise RuntimeError(
                    "[Diffusion-RM] rm_head.ckpt does not match current reward head. "
                    "This repository only supports Diffusion-RM v3 (QFormer-style) reward head. "
                    f"ckpt key sample: {list(ckpt.keys())[:20]}"
                )
