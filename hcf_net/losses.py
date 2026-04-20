from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .utils import collapse_to_acc2_targets


@dataclass
class LossOutput:
    total: torch.Tensor
    ce7: torch.Tensor
    mse: torch.Tensor
    bce: torch.Tensor
    cd: torch.Tensor


class HCFNetLoss:
    def __init__(
        self,
        alpha_cd: float = 0.1,
        class7_weight: float = 1.0,
        regression_weight: float = 1.0,
        binary_weight: float = 1.0,
    ):
        self.alpha_cd = alpha_cd
        self.class7_weight = class7_weight
        self.regression_weight = regression_weight
        self.binary_weight = binary_weight

    def __call__(
        self,
        outputs: dict,
        class7_target: Optional[torch.Tensor] = None,
        regression_target: Optional[torch.Tensor] = None,
        binary_target: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        device = outputs["z"].device
        zero = torch.tensor(0.0, device=device)

        ce7 = zero
        if class7_target is not None and "class7_logits" in outputs:
            ce7 = F.cross_entropy(outputs["class7_logits"], class7_target)

        mse = zero
        if regression_target is not None and "regression" in outputs:
            mse = F.mse_loss(outputs["regression"], regression_target.float())

        bce = zero
        if "binary_logit" in outputs:
            if binary_target is None:
                if class7_target is None:
                    raise ValueError("Need either binary_target or class7_target for binary supervision.")
                binary_target = collapse_to_acc2_targets(class7_target)
            bce = F.binary_cross_entropy_with_logits(outputs["binary_logit"], binary_target.float())

        cd = outputs["fusion"]["cd_loss"]

        total = (
            self.class7_weight * ce7
            + self.regression_weight * mse
            + self.binary_weight * bce
            + self.alpha_cd * cd
        )
        return LossOutput(total=total, ce7=ce7, mse=mse, bce=bce, cd=cd)
