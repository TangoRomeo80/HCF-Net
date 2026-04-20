"""

Tests robustness by dropping each modality at test time and measuring
performance degradation. Validates the claim that modality-drop
regularization promotes robustness.

Reports a table of metrics under each condition:
  - All modalities (baseline)
  - Drop Text (audio + visual only)
  - Drop Audio (text + visual only)
  - Drop Visual (text + audio only)

Usage:
    python scripts/missing_modality_test.py \
        --checkpoint ./outputs/mosi/best.pt \
        --manifest ./data/features/mosi/manifest.csv \
        --output-dir ./outputs/mosi/missing_modality
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from hcf_net.config import HCFNetConfig
from hcf_net.data import HCFNetFeatureDataset, collate_hcfnet, read_manifest
from hcf_net.metrics import compute_metrics
from hcf_net.model import HCFNet
from hcf_net.utils import set_seed


def move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate_with_condition(model, loader, device, drop_modality=None):
    """Run inference, optionally zeroing out one modality."""
    model.eval()
    all_outputs = []
    all_class7 = []
    all_scores = []

    for batch in loader:
        batch = move_batch(batch, device)

        text = batch["text"]
        audio = batch["audio"]
        visual = batch["visual"]

        # Zero out the dropped modality
        if drop_modality == "text":
            text = torch.zeros_like(text)
        elif drop_modality == "audio":
            audio = torch.zeros_like(audio)
        elif drop_modality == "visual":
            visual = torch.zeros_like(visual)

        outputs = model(
            text=text, audio=audio, visual=visual,
            text_mask=batch["text_mask"],
            audio_mask=batch["audio_mask"],
            visual_mask=batch["visual_mask"],
        )

        all_outputs.append({k: v.detach().cpu() for k, v in outputs.items() if torch.is_tensor(v)})
        all_class7.append(batch["class7"].detach().cpu())
        all_scores.append(batch["score"].detach().cpu())

    merged = {}
    for key in all_outputs[0].keys():
        merged[key] = torch.cat([o[key] for o in all_outputs], dim=0)

    class7 = torch.cat(all_class7, dim=0)
    scores = torch.cat(all_scores, dim=0)
    return compute_metrics(merged, class7, scores)


def main():
    parser = argparse.ArgumentParser(description="Test-time missing modality experiment.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="./missing_modality")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = HCFNetConfig(**ckpt["config"])
    model = HCFNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    # Build test loader
    records = read_manifest(args.manifest)
    dataset = HCFNetFeatureDataset(records, cfg)
    test_idx = [i for i, r in enumerate(records) if r.split.lower() == "test"]
    test_ds = Subset(dataset, test_idx)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_hcfnet)

    print(f"Test set: {len(test_idx)} utterances")
    print(f"Device: {device}\n")

    # ---- Run each condition ----
    conditions = [
        ("All Modalities", None),
        ("Drop Text", "text"),
        ("Drop Audio", "audio"),
        ("Drop Visual", "visual"),
    ]

    results = {}
    print("=" * 75)
    print(f"{'Condition':<20s} {'Acc-2':>7s}  {'Acc-7':>7s}  {'F1-W':>7s}  {'MAE':>7s}  {'Pearson':>8s}")
    print("-" * 75)

    def safe(v, default=0.0):
        """Replace None/NaN with default."""
        if v is None:
            return default
        try:
            import math
            if math.isnan(v):
                return default
        except (TypeError, ValueError):
            pass
        return v

    for name, drop in conditions:
        metrics = evaluate_with_condition(model, loader, device, drop_modality=drop)
        row = {
            "acc2": safe(metrics.acc2),
            "acc7": safe(metrics.acc7),
            "weighted_f1": safe(metrics.weighted_f1),
            "mae": safe(metrics.mae),
            "pearson_r": safe(metrics.pearson_r),
            "macro_f1": safe(metrics.macro_f1),
            "mse": safe(metrics.mse),
        }
        results[name] = row

        print(f"  {name:<18s} {row['acc2']:>7.3f}  {row['acc7']:>7.3f}  "
              f"{row['weighted_f1']:>7.3f}  {row['mae']:>7.3f}  {row['pearson_r']:>8.3f}")

    print("=" * 75)

    # Compute degradation relative to full model
    baseline = results["All Modalities"]

    print("\nDegradation (Δ from full model):")
    print("-" * 75)
    for name in ["Drop Text", "Drop Audio", "Drop Visual"]:
        r = results[name]
        d_acc2 = r["acc2"] - baseline["acc2"]
        d_acc7 = r["acc7"] - baseline["acc7"]
        d_mae = r["mae"] - baseline["mae"]
        print(f"  {name:<18s}  Δacc2={d_acc2:+.3f}  Δacc7={d_acc7:+.3f}  Δmae={d_mae:+.3f}")

        results[name]["delta_acc2"] = d_acc2
        results[name]["delta_acc7"] = d_acc7
        results[name]["delta_mae"] = d_mae

    print("-" * 75)

    with open(out / "missing_modality_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out / 'missing_modality_results.json'}")


if __name__ == "__main__":
    main()
