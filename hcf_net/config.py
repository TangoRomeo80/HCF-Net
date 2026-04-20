from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HCFNetConfig:
    # Input feature sizes.
    text_input_dim: int = 768
    audio_input_dim: int = 1024
    visual_input_dim: int = 2048

    model_dim: int = 256
    segment_length: int = 32  # we use 32 by default.

    local_num_heads: int = 4
    local_head_dim: int = 64  # 4 * 64 = 256

    global_num_heads: int = 8
    global_head_dim: int = 32  # 8 * 32 = 256

    fusion_num_blocks: int = 2
    fusion_num_heads: int = 8
    fusion_head_dim: int = 32  # 8 * 32 = 256
    fusion_ffn_dim: int = 512  # conservative lightweight choice.

    low_rank_k: int = 32
    dropout: float = 0.10
    modality_drop_p: float = 0.20

    classifier_hidden_dim: int = 512
    regressor_hidden_dim: int = 512
    binary_hidden_dim: int = 512

    num_classes: int = 7
    alpha_cd: float = 0.10

    enable_class7_head: bool = True
    enable_regression_head: bool = True
    # We expose this as optional to cover binary head.
    enable_binary_head: bool = True
