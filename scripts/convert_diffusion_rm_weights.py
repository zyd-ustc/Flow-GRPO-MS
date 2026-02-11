"""
将 Diffusion-RM（PyTorch）训练产物转换为 MindSpore ckpt，供 Flow-GRPO-MS 推理侧直接加载。

支持的输入结构（checkpoint_dir 为 step_xxx 目录，或直接为 backbone_lora 目录）：

- LoRA 模式：
  checkpoint_dir/
    backbone_lora/adapter_model.safetensors
    rm_head.pt
  或者（某些保存方式）：
  checkpoint_dir/
    backbone_lora/pytorch_lora_weights.safetensors
    rm_head.pt
  或者直接传入 LoRA 目录：
  checkpoint_dir(=backbone_lora)/
    adapter_model.safetensors | adapter_model.bin | pytorch_lora_weights.safetensors | pytorch_lora_weights.bin

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
  python scripts/convert_diffusion_rm_weights.py --checkpoint_dir "../Diffusion-RM/outputs/.../step_15000" --rm-head-add-prefix reward_head.

说明：
- LoRA adapter 的文件名/路径在不同保存方式下可能不同（见上方）。
- LoRA state_dict 的 key 形态在不同 PEFT 版本下也可能不同。
  本脚本默认 **不强制改写 key**，仅在你显式指定时才会：
  - 给 key 增加前缀（如 `backbone.`）
  - 将 `.lora_A.weight` 改写为 `.lora_A.<adapter>.weight`（同理 lora_B）
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Any, Optional, Tuple

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


def _to_ms_dtype(name: str) -> Optional[ms.dtype]:
    name = (name or "").lower()
    if name in ("", "keep", "none"):
        return None
    if name in ("fp16", "float16"):
        return ms.float16
    if name in ("fp32", "float32"):
        return ms.float32
    if name in ("bf16", "bfloat16"):
        return ms.bfloat16
    raise ValueError(f"Unsupported dtype name: {name}")


def _torch_tensor_to_ms(
    t,
    bf16_to_fp16: bool = True,
    cast_fp16: bool = False,
    target_ms_dtype: Optional[ms.dtype] = None,
) -> ms.Tensor:
    # 延迟 import torch，避免推理侧引入依赖
    import torch  # pylint: disable=import-error

    if isinstance(t, ms.Tensor):
        return t
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(t)}")

    if cast_fp16 and t.is_floating_point():
        # Force all floating tensors to fp16 for MindSpore ckpt.
        t = t.to(torch.float16)
    elif bf16_to_fp16 and t.dtype == torch.bfloat16:
        # torch bf16 -> numpy is not reliable; go through fp16 then cast on MindSpore side if needed
        t = t.to(torch.float16)

    out = ms.Tensor(t.detach().cpu().numpy())
    if target_ms_dtype is not None and out.dtype != target_ms_dtype and out.dtype in (
        ms.float16,
        ms.float32,
        ms.bfloat16,
    ):
        out = out.astype(target_ms_dtype)
    return out


def _strip_prefixes(name: str, prefixes: list[str]) -> str:
    for p in prefixes:
        if name.startswith(p):
            name = name[len(p) :]
    return name


def _add_prefixes(name: str, prefixes: list[str]) -> str:
    # Apply in order; avoid double-adding if already present.
    for p in prefixes:
        if p and not name.startswith(p):
            name = p + name
    return name


def _rewrite_state_dict_keys(
    state_dict: Dict[str, Any],
    strip_prefixes: list[str],
    add_prefixes: list[str],
) -> Dict[str, Any]:
    if not strip_prefixes and not add_prefixes:
        return state_dict
    out: Dict[str, Any] = {}
    collisions = []
    for k, v in state_dict.items():
        nk = _strip_prefixes(k, strip_prefixes)
        nk = _add_prefixes(nk, add_prefixes)
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


def _rewrite_lora_adapter_keys(
    state_dict: Dict[str, Any],
    backbone_prefix: str = "",
    adapter_name: str = "",
) -> Dict[str, Any]:
    """
    Best-effort rewrite LoRA adapter keys:
    - If `backbone_prefix` is non-empty, add it if missing.
    - If `adapter_name` is non-empty, convert `.lora_A.weight` -> `.lora_A.<adapter>.weight`
      (same for lora_B). This is useful for some multi-adapter PEFT variants.
    """
    out: Dict[str, Any] = {}
    collisions = []
    for k, v in state_dict.items():
        nk = str(k)

        if backbone_prefix and not nk.startswith(backbone_prefix):
            nk = backbone_prefix + nk

        # Insert adapter name segment if requested: lora_A.weight -> lora_A.<adapter>.weight
        if adapter_name:
            for lora_tag in ("lora_A", "lora_B"):
                needle = f".{lora_tag}.weight"
                if needle in nk and f".{lora_tag}.{adapter_name}.weight" not in nk:
                    nk = nk.replace(needle, f".{lora_tag}.{adapter_name}.weight")

        if nk in out and nk != k:
            collisions.append((k, nk))
            continue
        out[nk] = v

    if collisions:
        print(f"[warn] LoRA key collisions after rewrite: {len(collisions)} (kept first occurrence)")
        for old, new in collisions[:20]:
            print(f"  - {old} -> {new}")
    return out


def convert_pt_to_ckpt(
    pt_path: str,
    ckpt_path: str,
    overwrite: bool = False,
    bf16_to_fp16: bool = True,
    strip_prefixes: list[str] | None = None,
    add_prefixes: list[str] | None = None,
    cast_fp16: bool = False,
    target_dtype: str = "keep",
):
    import torch  # pylint: disable=import-error

    if os.path.exists(ckpt_path) and not overwrite:
        print(f"[skip] {ckpt_path} already exists")
        return

    obj = torch.load(pt_path, map_location="cpu")
    state_dict = _unwrap_state_dict(obj)
    state_dict = _rewrite_state_dict_keys(state_dict, strip_prefixes or [], add_prefixes or [])

    ms_items = []
    bf16_seen = False
    target_ms_dtype = _to_ms_dtype(target_dtype)
    for name, tensor in state_dict.items():
        if hasattr(tensor, "dtype"):
            try:
                import torch as _torch  # pylint: disable=import-error

                if tensor.dtype == _torch.bfloat16:
                    bf16_seen = True
            except Exception:
                pass
        ms_items.append(
            {
                "name": name,
                "data": _torch_tensor_to_ms(
                    tensor,
                    bf16_to_fp16=bf16_to_fp16,
                    cast_fp16=cast_fp16,
                    target_ms_dtype=target_ms_dtype,
                ),
            }
        )

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
    add_prefixes: list[str] | None = None,
    lora_adapter: bool = False,
    lora_backbone_prefix: str = "backbone.",
    lora_adapter_name: str = "default",
    cast_fp16: bool = False,
    default_target_dtype: str = "keep",
    lora_target_dtype: str = "keep",
):
    if os.path.exists(ckpt_path) and not overwrite:
        print(f"[skip] {ckpt_path} already exists")
        return

    from safetensors.torch import load_file  # pylint: disable=import-error

    state_dict = load_file(st_path)
    state_dict = _rewrite_state_dict_keys(state_dict, strip_prefixes or [], add_prefixes or [])
    if lora_adapter:
        state_dict = _rewrite_lora_adapter_keys(
            state_dict,
            backbone_prefix=lora_backbone_prefix,
            adapter_name=lora_adapter_name,
        )
    ms_items = []
    bf16_seen = False
    default_ms_dtype = _to_ms_dtype(default_target_dtype)
    lora_ms_dtype = _to_ms_dtype(lora_target_dtype)
    for name, tensor in state_dict.items():
        try:
            import torch as _torch  # pylint: disable=import-error

            if tensor.dtype == _torch.bfloat16:
                bf16_seen = True
        except Exception:
            pass
        ms_items.append(
            {
                "name": name,
                "data": _torch_tensor_to_ms(
                    tensor,
                    bf16_to_fp16=bf16_to_fp16,
                    cast_fp16=cast_fp16,
                    target_ms_dtype=(lora_ms_dtype if (lora_adapter and ".lora_" in str(name)) else default_ms_dtype),
                ),
            }
        )

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
        "--cast-fp16",
        action="store_true",
        help="将所有浮点权重统一转换为 float16 再保存为 MindSpore ckpt（会覆盖 fp32/bf16 等）。",
    )
    parser.add_argument(
        "--rm-head-dtype",
        type=str,
        default="fp32",
        choices=["keep", "fp16", "fp32", "bf16"],
        help="Target dtype for rm_head.ckpt tensors. Default: fp32 (matches common MindSpore reward_head params).",
    )
    parser.add_argument(
        "--adapter-dtype",
        type=str,
        default="keep",
        choices=["keep", "fp16", "fp32", "bf16"],
        help="Default target dtype for LoRA adapter ckpt tensors. Default: keep (preserve original).",
    )
    parser.add_argument(
        "--adapter-lora-dtype",
        type=str,
        default="keep",
        choices=["keep", "fp16", "fp32", "bf16"],
        help="Target dtype only for LoRA tensors (keys containing '.lora_') in adapter ckpt. Default: keep.",
    )
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=[],
        help="Strip prefix from state_dict keys before saving ckpt. "
        "Can be specified multiple times. Example: --strip-prefix reward_head.",
    )
    parser.add_argument(
        "--add-prefix",
        action="append",
        default=[],
        help="Add prefix to state_dict keys before saving ckpt. "
        "Can be specified multiple times. Example: --add-prefix reward_head.",
    )
    parser.add_argument(
        "--rm-head-add-prefix",
        type=str,
        default="",
        help="Only add prefix to rm_head.pt when converting (e.g. reward_head.). Overrides --add-prefix for rm_head.",
    )
    parser.add_argument(
        "--lora-dir",
        type=str,
        default="backbone_lora",
        help="LoRA 目录名或路径。默认按 checkpoint_dir/backbone_lora 查找；若传入绝对/相对路径且存在，则直接使用该目录。",
    )
    parser.add_argument(
        "--lora-backbone-prefix",
        type=str,
        default="",
        help="可选：为 LoRA adapter 的参数名增加前缀（例如 'backbone.'）。默认不加。",
    )
    parser.add_argument(
        "--lora-adapter-name",
        type=str,
        default="",
        help="可选：将 '.lora_A.weight' 改写为 '.lora_A.<adapter>.weight' 的 <adapter> 名称。默认不改写。",
    )
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir
    bf16_to_fp16 = not args.keep_bf16
    strip_prefixes = list(args.strip_prefix)
    add_prefixes = list(args.add_prefix)
    rm_head_add_prefix = str(args.rm_head_add_prefix or "")
    cast_fp16 = bool(args.cast_fp16)
    rm_head_dtype = str(args.rm_head_dtype)
    adapter_dtype = str(args.adapter_dtype)
    adapter_lora_dtype = str(args.adapter_lora_dtype)
    lora_dir_arg = str(args.lora_dir or "backbone_lora")
    lora_backbone_prefix = str(args.lora_backbone_prefix or "")
    lora_adapter_name = str(args.lora_adapter_name or "")

    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"checkpoint_dir not found: {ckpt_dir}")

    # rm_head
    rm_head_pt = os.path.join(ckpt_dir, "rm_head.pt")
    if os.path.exists(rm_head_pt):
        rm_add = [rm_head_add_prefix] if rm_head_add_prefix else add_prefixes
        convert_pt_to_ckpt(
            rm_head_pt,
            os.path.join(ckpt_dir, "rm_head.ckpt"),
            overwrite=args.overwrite,
            bf16_to_fp16=bf16_to_fp16,
            strip_prefixes=strip_prefixes,
            add_prefixes=rm_add,
            cast_fp16=cast_fp16,
            target_dtype=rm_head_dtype,
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
            add_prefixes=add_prefixes,
            cast_fp16=cast_fp16,
            target_dtype="keep",
        )

    # backbone_lora adapter
    def _resolve_lora_dir(checkpoint_dir: str, lora_dir: str) -> str:
        # If user passes an existing path, use it directly.
        if lora_dir and os.path.isdir(lora_dir):
            return lora_dir
        # Otherwise treat it as a subdir name under checkpoint_dir.
        cand = os.path.join(checkpoint_dir, lora_dir or "backbone_lora")
        return cand

    def _find_lora_weight_file(lora_dir: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Return (path, kind) where kind in {"safetensors","pt"}.
        Supports common filenames produced by peft.save_pretrained / diffusers lora utils.
        """
        if not os.path.isdir(lora_dir):
            return None, None
        candidates = [
            ("adapter_model.safetensors", "safetensors"),
            ("pytorch_lora_weights.safetensors", "safetensors"),
            ("adapter_model.bin", "pt"),
            ("pytorch_lora_weights.bin", "pt"),
            ("adapter_model.pt", "pt"),
            ("pytorch_lora_weights.pt", "pt"),
        ]
        for fn, kind in candidates:
            p = os.path.join(lora_dir, fn)
            if os.path.exists(p):
                return p, kind
        # fallback: pick the first safetensors/bin in the directory
        for fn in os.listdir(lora_dir):
            low = fn.lower()
            if low.endswith(".safetensors"):
                return os.path.join(lora_dir, fn), "safetensors"
        for fn in os.listdir(lora_dir):
            low = fn.lower()
            if low.endswith(".bin") or low.endswith(".pt") or low.endswith(".pth"):
                return os.path.join(lora_dir, fn), "pt"
        return None, None

    lora_dir = _resolve_lora_dir(ckpt_dir, lora_dir_arg)
    lora_weight_path, lora_kind = _find_lora_weight_file(lora_dir)
    if lora_weight_path:
        out_ckpt = os.path.join(lora_dir, "adapter_model.ckpt")
        if lora_kind == "safetensors":
            convert_safetensors_to_ckpt(
                lora_weight_path,
                out_ckpt,
                overwrite=args.overwrite,
                bf16_to_fp16=bf16_to_fp16,
                strip_prefixes=strip_prefixes,
                add_prefixes=add_prefixes,
                lora_adapter=True,
                lora_backbone_prefix=lora_backbone_prefix,
                lora_adapter_name=lora_adapter_name,
                cast_fp16=cast_fp16,
                default_target_dtype=adapter_dtype,
                lora_target_dtype=adapter_lora_dtype,
            )
        else:
            # .bin/.pt: use torch.load
            # NOTE: For LoRA adapters, we typically want to keep fp32.
            convert_pt_to_ckpt(
                lora_weight_path,
                out_ckpt,
                overwrite=args.overwrite,
                bf16_to_fp16=bf16_to_fp16,
                strip_prefixes=strip_prefixes,
                add_prefixes=add_prefixes,
                cast_fp16=cast_fp16,
                target_dtype="keep",
            )
            # Optional key rewrite for pt-based adapter: apply and re-save if requested
            if lora_backbone_prefix or lora_adapter_name:
                # Reload ckpt dict and rewrite is not supported without torch; so we instead re-run conversion with rewritten keys.
                # (We keep behavior consistent by loading the pt again.)
                import torch  # pylint: disable=import-error

                obj = torch.load(lora_weight_path, map_location="cpu")
                state_dict = _unwrap_state_dict(obj)
                state_dict = _rewrite_state_dict_keys(state_dict, strip_prefixes or [], add_prefixes or [])
                state_dict = _rewrite_lora_adapter_keys(
                    state_dict,
                    backbone_prefix=lora_backbone_prefix,
                    adapter_name=lora_adapter_name,
                )
                ms_items = []
                target_ms_dtype = _to_ms_dtype(adapter_dtype)
                for name, tensor in state_dict.items():
                    ms_items.append(
                        {
                            "name": name,
                            "data": _torch_tensor_to_ms(
                                tensor,
                                bf16_to_fp16=bf16_to_fp16,
                                cast_fp16=cast_fp16,
                                target_ms_dtype=target_ms_dtype,
                            ),
                        }
                    )
                ms.save_checkpoint(ms_items, out_ckpt)
                print(f"[ok] rewritten LoRA keys saved to {out_ckpt}")

    print("[done] conversion finished")


if __name__ == "__main__":
    main()

