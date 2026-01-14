from typing import List, Optional, Tuple, Dict, Any
import math
import mindspore as ms

from mindspore import mint, nn

class AdaLayerNormZero(nn.Cell):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_in_dim: int, embedding_out_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = mint.nn.Linear(embedding_in_dim, 6 * embedding_out_dim, bias=True)

        self.norm = mint.nn.LayerNorm(embedding_out_dim, elementwise_affine=False, eps=1e-6)

    def construct(
        self,
        x: ms.Tensor,
        emb: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class TransformerBlock(nn.Cell):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = mint.nn.LayerNorm(d_model)
        self.attn = ms.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = mint.nn.Dropout(dropout)
        self.norm2 = mint.nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.SequentialCell(
            mint.nn.Linear(d_model, hidden),
            nn.SiLU(),
            mint.nn.Dropout(dropout),
            mint.nn.Linear(hidden, d_model),
        )

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        # x: (B, N, C)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(h)
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + self.dropout(h)
        return x
    
class Conv1DHead(nn.Cell):
    """
    tokens: (B, N, C)
    text_embed: (B, D_text)
    return: logits: (B, N)
    """
    def __init__(self, in_c: int, width: int = 256, out_c: int = 256):
        super().__init__()
        
        self.conv1 = nn.SequentialCell(
            nn.Conv1d(in_c, width, kernel_size=1),
            nn.GroupNorm(num_groups=min(32, width), num_channels=width),
            nn.SiLU(),
        )
        
        self.conv2 = nn.SequentialCell(
            nn.Conv1d(width, width, kernel_size=9, padding=4),
            nn.GroupNorm(num_groups=min(32, width), num_channels=width),
            nn.SiLU(),
        )
        
        self.conv_out = nn.Conv1d(width, out_c, kernel_size=1)
        

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = x.permute(0, 2, 1).contiguous()  # (B, C, N)
        
        x = self.conv1(x)  # (B, width, N)
        x = self.conv2(x)  # (B, width, N)
        
        x = self.conv_out(x)  # (B, out_c, N)
        
        x = x.permute(0, 2, 1).contiguous()  # (B, N, out_c)
        
        return x

class Conv2DHead(nn.Cell):
    """
    tokens: (B, C, H, W)
    text_embed: (B, D_text)
    return: logits: (B, N)
    """
    def __init__(self, in_c: int, width: int = 256, out_c: int = 256, t_embed_dim: int = 512, use_t_embed: bool = True):
        super().__init__()

        if use_t_embed:
            self.norm1 = AdaLayerNormZero(t_embed_dim, in_c)
        else:
            self.norm1 = None

        self.conv1 = nn.SequentialCell(
            nn.Conv2d(in_c, width, kernel_size=1),
            nn.GroupNorm(num_groups=min(32, width), num_channels=width),
            nn.SiLU(),
        )

        if self.norm1 is not None:
            self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        else:
            self.norm2 = None

        self.conv2 = nn.SequentialCell(
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(32, width), num_channels=width),
            nn.SiLU(),
        )
        
        self.conv_out = nn.Conv2d(width, out_c, kernel_size=1)
        

    def construct(self, x: ms.Tensor, t_embed: ms.Tensor) -> ms.Tensor:
        if self.norm1 is not None and t_embed is not None:
            # rearrange x to (B, H*W, C)
            b, c, h, w = x.shape
            residual = x
            x_flat = x.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()
            x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(x_flat, emb=t_embed)
            x = x_norm.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        x = self.conv1(x)  # (B, width, H, W)

        if self.norm2 is not None:
            x = residual + gate_msa[:, :, None, None] * x
            residual = x
            x_flat = x.permute(0, 2, 3, 1).reshape(b, h * w, -1).contiguous()
            x_norm = self.norm2(x_flat)
            x = x_norm * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        x = self.conv2(x)  # (B, width, H, W)

        if self.norm1 is not None and t_embed is not None:
            x = residual + gate_mlp[:, :, None, None] * x

        x = self.conv_out(x)  # (B, out_c, H, W)
        
        return x
    

class TransformerHead(nn.Cell):
    """
    tokens: (B, C, H, W)
    text_embed: (B, D_text)
    return: logits: (B, N)
    """
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ms.nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.SequentialCell(
            mint.nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            mint.nn.Linear(hidden, d_model),
        )
        

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.norm1(x)
        x = x + self.attn(x, x, x, need_weights=False)[0]
        
        x = self.mlp(self.norm2(x))

        return x


class RewardHead(nn.Cell):
    def __init__(
        self,
        token_dim: int,
        conv_type: str = '1d',
        width: int = -1,
        out_dim: int = 1,
        n_visual_heads: int = 1,
        n_text_heads: int = 1,
        patch_size: int = 16,
        t_embed_dim: int = 512,
        use_t_embed: bool = True,
    ):
        super().__init__()
        assert conv_type in ['1d', '2d', 'transformer'], f'Unsupported conv_type: {conv_type}'
        self.conv_type = conv_type
        
        if width == -1:
            width = token_dim
            
        if conv_type == '1d':
            self.visual_heads = nn.CellList([
                Conv1DHead(token_dim, width, width) for _ in range(n_visual_heads)
            ])
        elif conv_type == '2d':
            self.visual_heads = nn.CellList([
                Conv2DHead(token_dim // (patch_size ** 2), width // (patch_size ** 2), width // (patch_size ** 2), t_embed_dim=t_embed_dim, use_t_embed=use_t_embed) for _ in range(n_visual_heads)
            ])
        elif conv_type == 'transformer':
            self.visual_heads = nn.CellList([
                TransformerHead(token_dim, n_heads=width // 64) for _ in range(n_visual_heads)
            ])
        else:
            raise ValueError(f'Unsupported conv_type: {conv_type}')
        
        if n_text_heads > 0:
            self.text_heads = nn.CellList([
                TransformerHead(token_dim, n_heads=width // 64) for _ in range(n_text_heads)
            ])
            self.use_text_features = True
        else:
            self.text_heads = None
            
        
        self.out = mint.nn.Linear((n_visual_heads + n_text_heads) * token_dim, out_dim)

        self.patch_size = patch_size  
        
    def unpatchify(self, x, hw=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        p = self.patch_size
        c = x.shape[2] // (p ** 2)

        if hw is None:
            h = w = int(x.shape[1] ** 0.5)
        else:
            h, w = hw
            h = h // p
            w = w // p

        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c)).contiguous()
        x = mint.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p)).contiguous()
        return imgs
    
    def patchify(self, imgs):
        p = self.patch_size
        c = imgs.shape[1]
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p)).contiguous()
        x = mint.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c)).contiguous()

        return x
    
        
    def construct(
        self,
        visual_features,
        text_features,
        t_embed: Optional[ms.Tensor] = None,
        hw=None) -> ms.Tensor:

        out_features = []
        if self.conv_type in ['1d', 'transformer']:
            # visual_features: (B, N, C)
            for head, visual_feature in zip(self.visual_heads, visual_features):
                visual_feature = head(visual_feature)
                visual_feature = visual_feature.mean(dim=1)
                out_features.append(visual_feature)
                
        elif self.conv_type == '2d':
            # visual_features: (B, C, H, W)
            for head, visual_feature in zip(self.visual_heads, visual_features):
                visual_feature = self.unpatchify(visual_feature, hw=hw)    # [B, h * w, p * p * C] -> [B, C, H, W]
                visual_feature = head(visual_feature, t_embed)   # (B, C, H, W) -> (B, out_c, H, W)
                visual_feature = self.patchify(visual_feature)
                visual_feature = visual_feature.mean(dim=1)
                out_features.append(visual_feature)
                
        if self.use_text_features and text_features is not None:
            # text_features: (B, N_text, C)
            for head, text_feature in zip(self.text_heads, text_features):
                text_feature = head(text_feature)
                text_feature = text_feature.mean(dim=1)
                out_features.append(text_feature)
                
        out_features = mint.cat(out_features, dim=-1)
        out = self.out(out_features)
        
        return out
        

