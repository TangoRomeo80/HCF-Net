from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .utils import collapse_to_acc2_targets


@dataclass
class MetricBundle:
    acc2: Optional[float] = None
    acc7: Optional[float] = None
    macro_f1: Optional[float] = None
    weighted_f1: Optional[float] = None
    binary_f1: Optional[float] = None
    auroc: Optional[float] = None
    mae: Optional[float] = None
    mse: Optional[float] = None
    pearson_r: Optional[float] = None


@torch.no_grad()
def compute_metrics(outputs: dict, class7_target: torch.Tensor, regression_target: Optional[torch.Tensor] = None) -> MetricBundle:
    out = MetricBundle()

    y_true_7 = class7_target.detach().cpu().numpy()

    if "class7_logits" in outputs:
        y_pred_7 = outputs["class7_logits"].argmax(dim=-1).detach().cpu().numpy()
        out.acc7 = float(accuracy_score(y_true_7, y_pred_7))
        out.macro_f1 = float(f1_score(y_true_7, y_pred_7, average="macro", zero_division=0))
        out.weighted_f1 = float(f1_score(y_true_7, y_pred_7, average="weighted", zero_division=0))
        y_true_2 = collapse_to_acc2_targets(class7_target).detach().cpu().numpy()
        y_pred_2 = (y_pred_7 >= 3).astype(np.int64)
        out.acc2 = float(accuracy_score(y_true_2, y_pred_2))

    if "binary_logit" in outputs:
        y_true_2 = collapse_to_acc2_targets(class7_target).detach().cpu().numpy()
        logits = outputs["binary_logit"].detach().cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        pred = (probs >= 0.5).astype(np.int64)
        out.binary_f1 = float(f1_score(y_true_2, pred, zero_division=0))
        try:
            out.auroc = float(roc_auc_score(y_true_2, probs))
        except ValueError:
            out.auroc = None

    if regression_target is not None and "regression" in outputs:
        y_true = regression_target.detach().cpu().numpy().astype(np.float64)
        y_pred = outputs["regression"].detach().cpu().numpy().astype(np.float64)
        out.mae = float(np.mean(np.abs(y_true - y_pred)))
        out.mse = float(np.mean((y_true - y_pred) ** 2))
        if np.std(y_true) > 0 and np.std(y_pred) > 0:
            out.pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
        else:
            out.pearson_r = None

    return out
