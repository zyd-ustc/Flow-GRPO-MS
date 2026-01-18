"""
将 Diffusion-RM（PyTorch）训练产物转换为 MindSpore ckpt，供 Flow-GRPO-MS 推理侧直接加载。

支持的输入结构（checkpoint_dir 为 step_xxx 目录）：

- LoRA 模式：
  checkpoint_dir/
    backbone_lora/adapter_model.safetensors
    rm_head.pt

- 全量训练（未冻结 backbone）：
  checkpoint_dir/
    full_model.pt

- 只训练 reward head（冻结 backbone）：
  checkpoint_dir/
    rm_head.pt

输出（同目录）：
  rm_head.ckpt / full_model.ckpt / backbone_lora/adapter_model.ckpt

用法示例：
  python scripts/convert_diffusion_rm_weights.py --checkpoint_dir "../Diffusion-RM/outputs/.../step_15000"
  python scripts/convert_diffusion_rm_weights.py --checkpoint_dir "../Diffusion-RM/outputs/.../step_15000" --strip-prefix reward_head.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Any

import mindspore as ms


def _unwrap_state_dict(obj: Any) -> Dict[str, Any]:
    """
    torch.load() 可能返回：
    - 纯 state_dict: Dict[str, Tensor]
    - 包一层: {"state_dict": {...}} / {"model": {...}} 等
    """
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "module", "net"):
            v = obj.get(k, None)
            if isinstance(v, dict) and v:
                return v
        # already a state_dict
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise ValueError("Unsupported checkpoint object format; cannot unwrap to state_dict.")


def _torch_tensor_to_ms(t, bf16_to_fp16: bool = True) -> ms.Tensor:
    # 延迟 import torch，避免推理侧引入依赖
    import torch  # pylint: disable=import-error

    if isinstance(t, ms.Tensor):
        return t
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(t)}")

    if bf16_to_fp16 and t.dtype == torch.bfloat16:
        t = t.to(torch.float16)
    return ms.Tensor(t.detach().cpu().numpy())


def _strip_prefixes(name: str, prefixes: list[str]) -> str:
    for p in prefixes:
        if name.startswith(p):
            name = name[len(p) :]
    return name


def _rewrite_state_dict_keys(state_dict: Dict[str, Any], strip_prefixes: list[str]) -> Dict[str, Any]:
    if not strip_prefixes:
        return state_dict
    out: Dict[str, Any] = {}
    collisions = []
    for k, v in state_dict.items():
        nk = _strip_prefixes(k, strip_prefixes)
        if nk in out and nk != k:
            collisions.append((k, nk))
            # keep the first occurrence
            continue
        out[nk] = v
    if collisions:
        print(f"[warn] key collisions after stripping prefixes: {len(collisions)} (kept first occurrence)")
        for old, new in collisions[:20]:
            print(f"  - {old} -> {new}")
    return out


def convert_pt_to_ckpt(
    pt_path: str,
    ckpt_path: str,
    overwrite: bool = False,
    bf16_to_fp16: bool = True,
    strip_prefixes: list[str] | None = None,
):
    import torch  # pylint: disable=import-error

    if os.path.exists(ckpt_path) and not overwrite:
        print(f"[skip] {ckpt_path} already exists")
        return

    obj = torch.load(pt_path, map_location="cpu")
    state_dict = _unwrap_state_dict(obj)
    state_dict = _rewrite_state_dict_keys(state_dict, strip_prefixes or [])

    ms_items = []
    bf16_seen = False
    for name, tensor in state_dict.items():
        if hasattr(tensor, "dtype"):
            try:
                import torch as _torch  # pylint: disable=import-error

                if tensor.dtype == _torch.bfloat16:
                    bf16_seen = True
            except Exception:
                pass
        ms_items.append({"name": name, "data": _torch_tensor_to_ms(tensor, bf16_to_fp16=bf16_to_fp16)})

    if bf16_seen and bf16_to_fp16:
        print("[warn] bf16 detected; converted to fp16 for MindSpore ckpt")

    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    ms.save_checkpoint(ms_items, ckpt_path)
    print(f"[ok] {pt_path} -> {ckpt_path} ({len(ms_items)} tensors)")


def convert_safetensors_to_ckpt(
    st_path: str,
    ckpt_path: str,
    overwrite: bool = False,
    bf16_to_fp16: bool = True,
    strip_prefixes: list[str] | None = None,
):
    if os.path.exists(ckpt_path) and not overwrite:
        print(f"[skip] {ckpt_path} already exists")
        return

    from safetensors.torch import load_file  # pylint: disable=import-error

    state_dict = load_file(st_path)
    state_dict = _rewrite_state_dict_keys(state_dict, strip_prefixes or [])
    ms_items = []
    bf16_seen = False
    for name, tensor in state_dict.items():
        try:
            import torch as _torch  # pylint: disable=import-error

            if tensor.dtype == _torch.bfloat16:
                bf16_seen = True
        except Exception:
            pass
        ms_items.append({"name": name, "data": _torch_tensor_to_ms(tensor, bf16_to_fp16=bf16_to_fp16)})

    if bf16_seen and bf16_to_fp16:
        print("[warn] bf16 detected; converted to fp16 for MindSpore ckpt")

    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    ms.save_checkpoint(ms_items, ckpt_path)
    print(f"[ok] {st_path} -> {ckpt_path} ({len(ms_items)} tensors)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Diffusion-RM 的 step_xxx checkpoint 目录")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 .ckpt")
    parser.add_argument("--keep_bf16", action="store_true", help="不将 bf16 转 fp16（若环境支持 bf16）")
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=[],
        help="Strip prefix from state_dict keys before saving ckpt. "
        "Can be specified multiple times. Example: --strip-prefix reward_head.",
    )
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir
    bf16_to_fp16 = not args.keep_bf16
    strip_prefixes = list(args.strip_prefix)

    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"checkpoint_dir not found: {ckpt_dir}")

    # rm_head
    rm_head_pt = os.path.join(ckpt_dir, "rm_head.pt")
    if os.path.exists(rm_head_pt):
        convert_pt_to_ckpt(
            rm_head_pt,
            os.path.join(ckpt_dir, "rm_head.ckpt"),
            overwrite=args.overwrite,
            bf16_to_fp16=bf16_to_fp16,
            strip_prefixes=strip_prefixes,
        )

    # full_model
    full_model_pt = os.path.join(ckpt_dir, "full_model.pt")
    if os.path.exists(full_model_pt):
        convert_pt_to_ckpt(
            full_model_pt,
            os.path.join(ckpt_dir, "full_model.ckpt"),
            overwrite=args.overwrite,
            bf16_to_fp16=bf16_to_fp16,
            strip_prefixes=strip_prefixes,
        )

    # backbone_lora adapter
    adapter_st = os.path.join(ckpt_dir, "backbone_lora", "adapter_model.safetensors")
    if os.path.exists(adapter_st):
        convert_safetensors_to_ckpt(
            adapter_st,
            os.path.join(ckpt_dir, "backbone_lora", "adapter_model.ckpt"),
            overwrite=args.overwrite,
            bf16_to_fp16=bf16_to_fp16,
            strip_prefixes=strip_prefixes,
        )

    print("[done] conversion finished")


if __name__ == "__main__":
    main()

