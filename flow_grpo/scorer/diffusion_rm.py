"""
Diffusion-RM 推理接口

加载 Diffusion-RM 的预训练权重，在推理时计算 reward。
不涉及训练逻辑，仅提供 scorer 接口。
"""

import os
from typing import Dict, List, Optional, Union

import mindspore as ms
import numpy as np
import torch
from PIL import Image

from .scorer import Scorer


class DiffusionRMFluxScorer(Scorer):
    """
    FLUX-based Diffusion Reward Model Scorer

    使用 PyTorch 实现的 Diffusion-RM 进行推理评分。

    Args:
        checkpoint_path: Diffusion-RM checkpoint 目录路径
        config_path: Diffusion-RM 配置文件路径
        pipeline: Diffusers FluxPipeline (可选，如果不提供会自动加载)
        pipeline_path: FluxPipeline 模型路径 (如 "black-forest-labs/FLUX.1-dev")
        device: 计算设备 ("cuda" 或 "cpu")
        u: 噪声水平，默认 0.9
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        pipeline=None,
        pipeline_path: str = "black-forest-labs/FLUX.1-dev",
        device: str = "cuda",
        u: float = 0.9,
    ):
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.u = u

        if pipeline is None:
            print(f"Loading FLUX pipeline from {pipeline_path}...")
            from diffusers import FluxPipeline

            self.pipeline = FluxPipeline.from_pretrained(
                pipeline_path,
                torch_dtype=torch.bfloat16,
            )
            self.pipeline.to(self.device)
        else:
            self.pipeline = pipeline

        print(f"Loading Diffusion-RM from {checkpoint_path}...")
        from diffusion_rm import DRMInferencer

        self.drm = DRMInferencer(
            pipeline=self.pipeline,
            config_path=config_path,
            model_path=checkpoint_path,
            device=str(self.device),
            model_dtype=torch.bfloat16,
        )

        print("Diffusion-RM loaded successfully!")

    def __call__(
        self,
        images: Union[List[Image.Image], np.ndarray, ms.Tensor],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        """
        评分图像

        Args:
            images: 图像 (PIL Image, numpy array, 或 MindSpore tensor)
            prompts: 提示词列表

        Returns:
            scores: 评分列表
        """
        pil_images = self._to_pil_images(images)
        latents = self._encode_images_to_latents(pil_images)

        if prompts is None:
            prompts = [""] * len(pil_images)
        text_conds = self._encode_prompts(prompts)

        rewards = self.drm.reward(
            text_conds=text_conds,
            latents=latents,
            u=self.u,
        )

        scores = (rewards.float() / 5.0).cpu().numpy().tolist()
        return scores

    def reward(
        self,
        latents: ms.Tensor,
        text_conds: Dict[str, ms.Tensor],
        u: Optional[float] = None,
    ) -> ms.Tensor:
        """
        直接对 latents 评分

        Args:
            latents: 潜空间表示 [B, C, H, W] (MindSpore tensor)
            text_conds: 文本条件，包含 encoder_hidden_states, pooled_projections
            u: 噪声水平 (可选，默认使用初始化时的 u)

        Returns:
            rewards: reward 分数 [B, 1] (MindSpore tensor)
        """
        if u is None:
            u = self.u

        latents_pt = torch.from_numpy(latents.asnumpy()).to(self.device)
        text_conds_pt = {
            k: torch.from_numpy(v.asnumpy()).to(self.device)
            for k, v in text_conds.items()
        }

        rewards_pt = self.drm.reward(
            text_conds=text_conds_pt,
            latents=latents_pt,
            u=u,
        )

        return ms.Tensor(rewards_pt.cpu().numpy())

    def _to_pil_images(self, images: Union[List[Image.Image], np.ndarray, ms.Tensor]) -> List[Image.Image]:
        if isinstance(images, list) and all(isinstance(img, Image.Image) for img in images):
            return images
        elif isinstance(images, np.ndarray):
            if images.ndim == 3:
                images = [Image.fromarray(images)]
            else:
                images = [Image.fromarray(img) for img in images]
        elif isinstance(images, ms.Tensor):
            np_images = images.asnumpy()
            if np_images.ndim == 3:
                np_images = np_images.transpose(1, 2, 0)
                images = [Image.fromarray(np_images)]
            else:
                np_images = np_images.transpose(0, 2, 3, 1)
                images = [Image.fromarray(img) for img in np_images]
        else:
            raise ValueError(f"Unsupported image type: {type(images)}")

        return images

    def _encode_images_to_latents(self, images: List[Image.Image]) -> torch.Tensor:
        self.pipeline.vae.eval()

        latents_list = []
        with torch.no_grad():
            for img in images:
                img = img.convert("RGB")
                pixel_values = self.pipeline.image_processor.preprocess(img).to(
                    self.device, dtype=torch.bfloat16
                )

                latents = self.pipeline.vae.encode(pixel_values).latent_dist.sample()
                latents_list.append(latents)

        latents = torch.cat(latents_list, dim=0)
        return latents

    def _encode_prompts(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        from diffusion_rm.models.sd3_rm import encode_prompt

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders=[
                    self.pipeline.text_encoder,
                    self.pipeline.text_encoder_2,
                    self.pipeline.text_encoder_3,
                ],
                tokenizers=[
                    self.pipeline.tokenizer,
                    self.pipeline.tokenizer_2,
                    self.pipeline.tokenizer_3,
                ],
                prompts=prompts,
                max_sequence_length=128,
            )

            prompt_embeds = prompt_embeds.to(self.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(self.device)

        return {
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
        }


class DiffusionRMSD3Scorer(Scorer):
    """
    SD3-based Diffusion Reward Model Scorer

    与 DiffusionRMFluxScorer 类似，但使用 SD3 pipeline。
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        pipeline=None,
        pipeline_path: str = "stabilityai/stable-diffusion-3.5-medium",
        device: str = "cuda",
        u: float = 0.9,
    ):
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.u = u

        if pipeline is None:
            print(f"Loading SD3 pipeline from {pipeline_path}...")
            from diffusers import StableDiffusion3Pipeline

            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                pipeline_path,
                torch_dtype=torch.bfloat16,
            )
            self.pipeline.to(self.device)
        else:
            self.pipeline = pipeline

        print(f"Loading Diffusion-RM from {checkpoint_path}...")
        from diffusion_rm import DRMInferencer

        self.drm = DRMInferencer(
            pipeline=self.pipeline,
            config_path=config_path,
            model_path=checkpoint_path,
            device=str(self.device),
            model_dtype=torch.bfloat16,
        )

        print("Diffusion-RM loaded successfully!")

    def __call__(
        self,
        images: Union[List[Image.Image], np.ndarray, ms.Tensor],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        pil_images = self._to_pil_images(images)
        latents = self._encode_images_to_latents(pil_images)

        if prompts is None:
            prompts = [""] * len(pil_images)
        text_conds = self._encode_prompts(prompts)

        rewards = self.drm.reward(
            text_conds=text_conds,
            latents=latents,
            u=self.u,
        )

        scores = (rewards.float() / 5.0).cpu().numpy().tolist()
        return scores

    def reward(
        self,
        latents: ms.Tensor,
        text_conds: Dict[str, ms.Tensor],
        u: Optional[float] = None,
    ) -> ms.Tensor:
        if u is None:
            u = self.u

        latents_pt = torch.from_numpy(latents.asnumpy()).to(self.device)
        text_conds_pt = {
            k: torch.from_numpy(v.asnumpy()).to(self.device)
            for k, v in text_conds.items()
        }

        rewards_pt = self.drm.reward(
            text_conds=text_conds_pt,
            latents=latents_pt,
            u=u,
        )

        return ms.Tensor(rewards_pt.cpu().numpy())

    def _to_pil_images(self, images: Union[List[Image.Image], np.ndarray, ms.Tensor]) -> List[Image.Image]:
        if isinstance(images, list) and all(isinstance(img, Image.Image) for img in images):
            return images
        elif isinstance(images, np.ndarray):
            if images.ndim == 3:
                images = [Image.fromarray(images)]
            else:
                images = [Image.fromarray(img) for img in images]
        elif isinstance(images, ms.Tensor):
            np_images = images.asnumpy()
            if np_images.ndim == 3:
                np_images = np_images.transpose(1, 2, 0)
                images = [Image.fromarray(np_images)]
            else:
                np_images = np_images.transpose(0, 2, 3, 1)
                images = [Image.fromarray(img) for img in np_images]
        else:
            raise ValueError(f"Unsupported image type: {type(images)}")

        return images

    def _encode_images_to_latents(self, images: List[Image.Image]) -> torch.Tensor:
        self.pipeline.vae.eval()

        latents_list = []
        with torch.no_grad():
            for img in images:
                img = img.convert("RGB")
                pixel_values = self.pipeline.image_processor.preprocess(img).to(
                    self.device, dtype=torch.bfloat16
                )

                latents = self.pipeline.vae.encode(pixel_values).latent_dist.sample()
                latents_list.append(latents)

        latents = torch.cat(latents_list, dim=0)
        return latents

    def _encode_prompts(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        from diffusion_rm.models.sd3_rm import encode_prompt

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders=[
                    self.pipeline.text_encoder,
                    self.pipeline.text_encoder_2,
                    self.pipeline.text_encoder_3,
                ],
                tokenizers=[
                    self.pipeline.tokenizer,
                    self.pipeline.tokenizer_2,
                    self.pipeline.tokenizer_3,
                ],
                prompts=prompts,
                max_sequence_length=128,
            )

            prompt_embeds = prompt_embeds.to(self.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(self.device)

        return {
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
        }
