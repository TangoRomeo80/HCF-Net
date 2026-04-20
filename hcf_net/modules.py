from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .utils import l2_normalize_last_dim, masked_mean


class TimeDistributedProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, l2_norm_input: bool = False):
        super().__init__()
        self.l2_norm_input = l2_norm_input
        self.proj = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.l2_norm_input:
            x = l2_normalize_last_dim(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class PreNormSelfAttentionBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float, ffn_dim: Optional[int] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = None
        if ffn_dim is not None:
            self.norm2 = nn.LayerNorm(model_dim)
            self.ffn = nn.Sequential(
                nn.Linear(model_dim, ffn_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, model_dim),
            )
            self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.norm1(x)
        attn_out, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout1(attn_out)

        if self.ffn is not None:
            y = self.norm2(x)
            x = x + self.dropout2(self.ffn(y))
        return x


class HierarchicalGraphEncoder(nn.Module):
    """
    modality-specific two-tier encoder:
    - local self-attention inside each fixed-length segment
    - mean pooling to segment nodes
    - global self-attention across segment nodes
    - mean pooling to utterance-level embedding

    Input shape:  [B, S, L, D]
    Token mask:   [B, S, L] (1 for valid token, 0 for padded token)
    Output shape: [B, D]
    """

    def __init__(self, model_dim: int, local_heads: int, global_heads: int, dropout: float):
        super().__init__()
        self.local_block = PreNormSelfAttentionBlock(
            model_dim=model_dim,
            num_heads=local_heads,
            dropout=dropout,
            ffn_dim=None,
        )
        self.local_ln = nn.LayerNorm(model_dim)
        self.global_block = PreNormSelfAttentionBlock(
            model_dim=model_dim,
            num_heads=global_heads,
            dropout=dropout,
            ffn_dim=None,
        )
        self.global_ln = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, s, l, d = x.shape

        x_flat = x.reshape(b * s, l, d)
        mask_flat = token_mask.reshape(b * s, l)
        local_key_padding_mask = ~mask_flat.bool()

        h_local = self.local_block(x_flat, key_padding_mask=local_key_padding_mask)
        h_local = self.local_ln(h_local)
        h_local = h_local.reshape(b, s, l, d)

        segment_nodes = masked_mean(h_local, token_mask, dim=2)  # [B, S, D]
        segment_valid = token_mask.any(dim=2)  # [B, S]

        global_key_padding_mask = ~segment_valid.bool()
        h_global = self.global_block(segment_nodes, key_padding_mask=global_key_padding_mask)
        h_global = self.global_ln(h_global)

        utterance_embedding = masked_mean(h_global, segment_valid, dim=1)

        aux = {
            "local_tokens": h_local,
            "segment_nodes": segment_nodes,
            "global_nodes": h_global,
            "segment_valid": segment_valid,
        }
        return utterance_embedding, aux


class CrossModalTransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float, ffn_dim: int):
        super().__init__()
        self.block = PreNormSelfAttentionBlock(
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            ffn_dim=ffn_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, 3, D]. There is no padding because all three modalities are always present.
        return self.block(x, key_padding_mask=None)


class ConflictAwareFusion(nn.Module):
    """
    dual-branch fusion:
      H in R^{B x 3 x D}
      alignment branch -> a_bar in R^{B x D}
      residual branch  -> r_bar in R^{B x D}
      gate g in R^{B x 1}
      z = g * a_bar + (1 - g) * r_bar
      LCD = | g - ||r|| / (||a|| + ||r||) |
    """

    def __init__(self, model_dim: int, num_blocks: int, num_heads: int, ffn_dim: int, low_rank_k: int, dropout: float):
        super().__init__()
        self.alignment_blocks = nn.ModuleList(
            [CrossModalTransformerBlock(model_dim, num_heads, dropout, ffn_dim) for _ in range(num_blocks)]
        )
        self.low_rank_basis = nn.Parameter(torch.empty(model_dim, low_rank_k))
        nn.init.xavier_uniform_(self.low_rank_basis)
        self.gate = nn.Linear(model_dim * 2, 1)

    def forward(self, g_text: torch.Tensor, g_audio: torch.Tensor, g_visual: torch.Tensor) -> dict[str, torch.Tensor]:
        h = torch.stack([g_text, g_audio, g_visual], dim=1)  # [B, 3, D]

        a = h
        for block in self.alignment_blocks:
            a = block(a)
        a_bar = a.mean(dim=1)

        p = self.low_rank_basis
        # N = P P^T H, implemented as H @ P @ P^T
        n = (h @ p) @ p.transpose(0, 1)
        r = h - n
        r_bar = r.mean(dim=1)

        gate_logits = self.gate(torch.cat([a_bar, r_bar], dim=-1))
        g = torch.sigmoid(gate_logits)
        z = g * a_bar + (1.0 - g) * r_bar

        a_norm = a_bar.norm(p=2, dim=-1, keepdim=True)
        r_norm = r_bar.norm(p=2, dim=-1, keepdim=True)
        residual_ratio = r_norm / (a_norm + r_norm + 1e-8)
        cd_loss = torch.abs(g - residual_ratio).mean()

        return {
            "H": h,
            "A": a,
            "N": n,
            "R": r,
            "a_bar": a_bar,
            "r_bar": r_bar,
            "gate": g,
            "z": z,
            "cd_loss": cd_loss,
        }


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float, final_activation: Optional[str] = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
        )
        self.final_activation = final_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        if self.final_activation == "sigmoid":
            return torch.sigmoid(x)
        return x


class ModalityDrop(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p <= 0.0:
            return x
        b = x.shape[0]
        keep = (torch.rand(b, 1, 1, 1, device=x.device) > self.p).to(dtype=x.dtype)
        return x * keep
