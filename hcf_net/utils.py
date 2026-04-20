from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2_normalize_last_dim(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps))


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    x: tensor
    mask: boolean/binary mask broadcastable to x except on `dim`
    """
    mask = mask.to(dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    masked_x = x * mask
    denom = mask.sum(dim=dim).clamp_min(1.0)
    return masked_x.sum(dim=dim) / denom


def pad_to_multiple(length: int, multiple: int) -> int:
    if length % multiple == 0:
        return length
    return ((length // multiple) + 1) * multiple


def chunk_sequence(
    x: np.ndarray,
    segment_length: int,
    pad_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert [T, D] into [S, L, D] and segment-token mask [S, L].
    """
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array [T, D], got shape={x.shape}")

    t, d = x.shape
    padded_t = pad_to_multiple(t, segment_length)
    s = padded_t // segment_length

    out = np.full((padded_t, d), pad_value, dtype=x.dtype)
    out[:t] = x
    mask = np.zeros((padded_t,), dtype=np.float32)
    mask[:t] = 1.0

    out = out.reshape(s, segment_length, d)
    mask = mask.reshape(s, segment_length)
    return out, mask


def build_remainder_oversampled_indices(
    labels: Sequence[int],
    target_per_class: int,
) -> list[int]:
    """
    Implements the idea of expanding each class until it has at least
    `target_per_class` items. This is controlled oversampling.
    """
    class_to_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        class_to_indices.setdefault(int(label), []).append(idx)

    final_indices: list[int] = []
    for cls, indices in class_to_indices.items():
        if not indices:
            continue
        repeats = target_per_class // len(indices)
        remainder = target_per_class % len(indices)
        final_indices.extend(indices * repeats)
        final_indices.extend(indices[:remainder])

    random.shuffle(final_indices)
    return final_indices


def collapse_to_acc2_targets(class7_targets: torch.Tensor) -> torch.Tensor:
    """
    CMU-MOSI convention used: negative vs non-negative.
    Original class ordering is assumed to be [-3,-2,-1,0,+1,+2,+3] => indices [0..6].
    Therefore negative = {0,1,2}, non-negative = {3,4,5,6}.
    """
    return (class7_targets >= 3).long()


def score_to_class7(score: float) -> int:
    """
    Convert a continuous MOSI-style score to one of 7 discrete bins.
    This helper is only for fallback data preparation.
    """
    clipped = max(-3.0, min(3.0, float(score)))
    # Round to nearest integer and shift to [0, 6]
    return int(round(clipped)) + 3
