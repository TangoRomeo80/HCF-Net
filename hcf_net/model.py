from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .config import HCFNetConfig
from .modules import (
    ConflictAwareFusion,
    HierarchicalGraphEncoder,
    MLPHead,
    ModalityDrop,
    TimeDistributedProjection,
)


class HCFNet(nn.Module):
    """
    Expected inputs:
      text   : [B, S, L, 768]
      audio  : [B, S, L, 1024]
      visual : [B, S, L, 2048]
      masks  : [B, S, L] for each modality

    These are assumed to be *already aligned and segmented* in preprocessing pipeline. Because modality features are cached as tensors before training.
    """

    def __init__(self, config: HCFNetConfig):
        super().__init__()
        self.config = config

        self.modality_drop = ModalityDrop(config.modality_drop_p)

        self.text_proj = TimeDistributedProjection(
            in_dim=config.text_input_dim,
            out_dim=config.model_dim,
            dropout=config.dropout,
            l2_norm_input=True,
        )
        self.audio_proj = TimeDistributedProjection(
            in_dim=config.audio_input_dim,
            out_dim=config.model_dim,
            dropout=config.dropout,
            l2_norm_input=False,
        )
        self.visual_proj = TimeDistributedProjection(
            in_dim=config.visual_input_dim,
            out_dim=config.model_dim,
            dropout=config.dropout,
            l2_norm_input=True,
        )

        self.text_encoder = HierarchicalGraphEncoder(
            model_dim=config.model_dim,
            local_heads=config.local_num_heads,
            global_heads=config.global_num_heads,
            dropout=config.dropout,
        )
        self.audio_encoder = HierarchicalGraphEncoder(
            model_dim=config.model_dim,
            local_heads=config.local_num_heads,
            global_heads=config.global_num_heads,
            dropout=config.dropout,
        )
        self.visual_encoder = HierarchicalGraphEncoder(
            model_dim=config.model_dim,
            local_heads=config.local_num_heads,
            global_heads=config.global_num_heads,
            dropout=config.dropout,
        )

        self.fusion = ConflictAwareFusion(
            model_dim=config.model_dim,
            num_blocks=config.fusion_num_blocks,
            num_heads=config.fusion_num_heads,
            ffn_dim=config.fusion_ffn_dim,
            low_rank_k=config.low_rank_k,
            dropout=config.dropout,
        )

        self.class7_head = None
        if config.enable_class7_head:
            self.class7_head = MLPHead(
                in_dim=config.model_dim,
                hidden_dim=config.classifier_hidden_dim,
                out_dim=config.num_classes,
                dropout=config.dropout,
            )

        self.regression_head = None
        if config.enable_regression_head:
            self.regression_head = MLPHead(
                in_dim=config.model_dim,
                hidden_dim=config.regressor_hidden_dim,
                out_dim=1,
                dropout=config.dropout,
            )

        self.binary_head = None
        if config.enable_binary_head:
            self.binary_head = MLPHead(
                in_dim=config.model_dim,
                hidden_dim=config.binary_hidden_dim,
                out_dim=1,
                dropout=config.dropout,
            )

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> dict[str, Any]:
        # Modality-drop is applied at input-stream level.
        text = self.modality_drop(text)
        audio = self.modality_drop(audio)
        visual = self.modality_drop(visual)

        text = self.text_proj(text)
        audio = self.audio_proj(audio)
        visual = self.visual_proj(visual)

        g_text, text_aux = self.text_encoder(text, text_mask)
        g_audio, audio_aux = self.audio_encoder(audio, audio_mask)
        g_visual, visual_aux = self.visual_encoder(visual, visual_mask)

        fused = self.fusion(g_text, g_audio, g_visual)
        z = fused["z"]

        out: dict[str, Any] = {
            "g_text": g_text,
            "g_audio": g_audio,
            "g_visual": g_visual,
            "fusion": fused,
            "z": z,
            "text_aux": text_aux,
            "audio_aux": audio_aux,
            "visual_aux": visual_aux,
        }

        if self.class7_head is not None:
            out["class7_logits"] = self.class7_head(z)
        if self.regression_head is not None:
            out["regression"] = self.regression_head(z).squeeze(-1)
        if self.binary_head is not None:
            out["binary_logit"] = self.binary_head(z).squeeze(-1)

        return out
