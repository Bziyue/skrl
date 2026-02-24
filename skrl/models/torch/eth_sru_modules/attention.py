"""Cross-attention fusion modules.

Adapted from ETH Zurich RSL modules for local skrl integration.
"""

from __future__ import annotations

import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_positional_encoding_3d(
    channels: int, D: int, H: int, W: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Compute 3D positional encoding tensor in (1, C, D, H, W)."""
    org_channels = channels
    channels = int(math.ceil(channels / 6) * 2)
    if channels % 2:
        channels += 1
    inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=device).float() / channels))

    def get_emb(sin_inp: torch.Tensor) -> torch.Tensor:
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    pos_x = torch.arange(D, device=device, dtype=inv_freq.dtype)
    pos_y = torch.arange(H, device=device, dtype=inv_freq.dtype)
    pos_z = torch.arange(W, device=device, dtype=inv_freq.dtype)
    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
    sin_inp_z = torch.einsum("i,j->ij", pos_z, inv_freq)

    emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
    emb_y = get_emb(sin_inp_y).unsqueeze(1)
    emb_z = get_emb(sin_inp_z)

    emb = torch.zeros((D, H, W, channels * 3), device=device, dtype=dtype)
    emb[:, :, :, :channels] = emb_x
    emb[:, :, :, channels : 2 * channels] = emb_y
    emb[:, :, :, 2 * channels :] = emb_z

    enc = emb[None, :, :, :, :org_channels]
    enc = enc.permute(0, 4, 1, 2, 3)
    return enc


class CrossAttentionFuseModule(nn.Module):
    """Self-attention + cross-attention fusion for multi-view visual tokens."""

    def __init__(
        self,
        image_dim: int,
        info_dim: int,
        num_heads: int,
        spatial_dims: tuple[int, int, int],
    ) -> None:
        super().__init__()
        if image_dim % num_heads != 0:
            raise ValueError("image_dim must be divisible by num_heads")

        expand_dim = image_dim * 2
        self.image_dim = image_dim

        self.info_proj = nn.Sequential(
            nn.Linear(info_dim, expand_dim),
            nn.ELU(inplace=True),
            nn.Linear(expand_dim, image_dim),
            nn.ELU(inplace=True),
        )

        D, H, W = spatial_dims
        pos_enc = _compute_positional_encoding_3d(image_dim, D, H, W, torch.device("cpu"), torch.float32)
        self.register_buffer("pos_encoding", pos_enc, persistent=True)

        self.norm1 = nn.LayerNorm(image_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=image_dim, num_heads=num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(image_dim)
        self.ffn = nn.Sequential(
            nn.Linear(image_dim, expand_dim),
            nn.ELU(inplace=True),
            nn.Linear(expand_dim, image_dim),
            nn.ELU(inplace=True),
        )

        self.cross_attn = nn.MultiheadAttention(embed_dim=image_dim, num_heads=num_heads, batch_first=True)

    def forward(self, img: Union[torch.Tensor, List[torch.Tensor]], info: torch.Tensor) -> torch.Tensor:
        mask: torch.Tensor | None = None

        if isinstance(img, list):
            views = img
            B, _ = views[0].shape[:2]
            H_max = max(v.shape[2] for v in views)
            W_max = max(v.shape[3] for v in views)
            padded, masks = [], []

            for v in views:
                _, _, h, w = v.shape
                pad = (0, W_max - w, 0, H_max - h)
                padded.append(F.pad(v, pad))
                m = torch.zeros((B, h, w), dtype=torch.bool, device=v.device)
                masks.append(F.pad(m, pad, value=True))

            feats = torch.stack(padded, dim=2)
            mask = torch.stack(masks, dim=1)
        elif img.dim() == 5:
            feats = img
        else:
            feats = img.unsqueeze(2)

        B, C, D, H, W = feats.shape
        feats = feats + self.pos_encoding.to(feats.device, feats.dtype)
        x = feats.view(B, C, D * H * W).permute(0, 2, 1)
        key_mask = mask.view(B, D * H * W) if mask is not None else None

        x_norm = self.norm1(x)
        sa, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=key_mask, need_weights=False)
        x = x + sa
        x = x + self.ffn(self.norm2(x))

        q = self.info_proj(info).unsqueeze(1)
        ca, _ = self.cross_attn(q, x, x, key_padding_mask=key_mask, need_weights=False)
        return ca.squeeze(1)
