"""
Stratified Conflict Analysis — Reviewer #1 Comment 6

Partitions test utterances by gate value into tertiles:
  - LOW  gate (g < 0.33)  = High conflict
  - MID  gate (0.33-0.66) = Moderate conflict
  - HIGH gate (g > 0.66)  = Low conflict (agreement)

Reports Acc-7 and MAE per tertile for:
  - Full HCF-Net model
  - Alignment-only ablation (requires separate checkpoint)

Also exports data for gate distribution histogram (Figure for manuscript).

Usage:
    python scripts/stratified_conflict_analysis.py \
        --checkpoint ./outputs/mosi/best.pt \
        --manifest ./data/features/mosi/manifest.csv \
        --output-dir ./outputs/mosi/conflict_analysis

    # With alignment-only ablation comparison:
    python scripts/stratified_conflict_analysis.py \
        --checkpoint ./outputs/mosi/best.pt \
        --ablation-checkpoint ./outputs/mosi_no_conflict/best.pt \
        --manifest ./data/features/mosi/manifest.csv \
        --output-dir ./outputs/mosi/conflict_analysis
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
from hcf_net.model import HCFNet
from hcf_net.utils import set_seed


def move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def extract_predictions_and_gates(model, loader, device):
    """Run inference and return per-utterance predictions + gate values."""
    model.eval()
    results = []

    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(
            text=batch["text"], audio=batch["audio"], visual=batch["visual"],
            text_mask=batch["text_mask"], audio_mask=batch["audio_mask"],
            visual_mask=batch["visual_mask"],
        )

        pred_7 = outputs["class7_logits"].argmax(dim=-1).cpu().numpy()
        pred_reg = outputs["regression"].cpu().numpy().flatten()
        true_7 = batch["class7"].cpu().numpy()
        true_score = batch["score"].cpu().numpy()
        ids = batch["utterance_id"]

        # Gate values
        gates = None
        if "fusion" in outputs and "gate" in outputs["fusion"]:
            gates = outputs["fusion"]["gate"].cpu().numpy().flatten()

        for i in range(len(ids)):
            row = {
                "id": ids[i],
                "true_class7": int(true_7[i]),
                "pred_class7": int(pred_7[i]),
                "true_score": float(true_score[i]),
                "pred_score": float(pred_reg[i]),
                "gate": float(gates[i]) if gates is not None else None,
            }
            results.append(row)

    return results


def compute_tertile_metrics(results, tertile_edges=(0.33, 0.66)):
    """Compute Acc-7 and MAE per gate value tertile."""
    gates = np.array([r["gate"] for r in results], dtype=np.float64)
    true_7 = np.array([r["true_class7"] for r in results])
    pred_7 = np.array([r["pred_class7"] for r in results])
    true_s = np.array([r["true_score"] for r in results], dtype=np.float64)
    pred_s = np.array([r["pred_score"] for r in results], dtype=np.float64)

    # Filter out NaN gate values
    valid_mask = ~np.isnan(gates)
    n_valid = valid_mask.sum()
    n_nan = (~valid_mask).sum()
    if n_nan > 0:
        print(f"  Note: {n_nan}/{len(gates)} gate values are NaN (filtered out)")

    if n_valid < 3:
        print("  ERROR: Too few valid gate values for tertile analysis.")
        return {"error": "insufficient_valid_gates", "n_valid": int(n_valid)}

    # Use only valid entries for tertile computation
    v_gates = gates[valid_mask]
    v_true_7 = true_7[valid_mask]
    v_pred_7 = pred_7[valid_mask]
    v_true_s = true_s[valid_mask]
    v_pred_s = pred_s[valid_mask]

    # Use actual tertiles from data
    t1, t2 = np.percentile(v_gates, [33.3, 66.6])

    tertiles = {
        f"LOW (g<{t1:.2f}) = High Conflict": v_gates < t1,
        f"MID ({t1:.2f}≤g<{t2:.2f}) = Moderate": (v_gates >= t1) & (v_gates < t2),
        f"HIGH (g≥{t2:.2f}) = Agreement": v_gates >= t2,
    }

    metrics = {}
    print("\n" + "=" * 70)
    print("STRATIFIED CONFLICT ANALYSIS")
    print("=" * 70)
    print(f"{'Tertile':<40s} {'N':>5s}  {'Acc-7':>6s}  {'MAE':>6s}  {'Acc-2':>6s}")
    print("-" * 70)

    for name, mask in tertiles.items():
        n = mask.sum()
        if n == 0:
            continue

        acc7 = (v_pred_7[mask] == v_true_7[mask]).mean()
        mae_val = np.abs(v_pred_s[mask] - v_true_s[mask]).mean()
        true_bin = (v_true_7[mask] >= 3).astype(int)
        pred_bin = (v_pred_7[mask] >= 3).astype(int)
        acc2 = (pred_bin == true_bin).mean()

        # Handle NaN in mae
        mae_val = float(mae_val) if not np.isnan(mae_val) else 0.0

        metrics[name] = {"n": int(n), "acc7": float(acc7), "mae": mae_val, "acc2": float(acc2)}
        print(f"  {name:<38s} {n:>5d}  {acc7:>6.3f}  {mae_val:>6.3f}  {acc2:>6.3f}")

    # Overall (using all valid entries)
    acc7_all = (v_pred_7 == v_true_7).mean()
    mae_all = np.abs(v_pred_s - v_true_s).mean()
    mae_all = float(mae_all) if not np.isnan(mae_all) else 0.0
    true_bin_all = (v_true_7 >= 3).astype(int)
    pred_bin_all = (v_pred_7 >= 3).astype(int)
    acc2_all = (pred_bin_all == true_bin_all).mean()
    print("-" * 70)
    print(f"  {'OVERALL':<38s} {int(n_valid):>5d}  {acc7_all:>6.3f}  {mae_all:>6.3f}  {acc2_all:>6.3f}")
    print("=" * 70)

    metrics["overall"] = {
        "n": int(n_valid), "acc7": float(acc7_all),
        "mae": mae_all, "acc2": float(acc2_all)
    }
    metrics["tertile_thresholds"] = {"t1": float(t1), "t2": float(t2)}

    return metrics


def export_gate_histogram_data(results, output_dir):
    """Export gate values for histogram plotting in manuscript."""
    gates = np.array([r["gate"] for r in results], dtype=np.float64)
    correct = np.array([r["pred_class7"] == r["true_class7"] for r in results])

    out = Path(output_dir)

    # Filter NaN for stats
    valid = ~np.isnan(gates)
    v_gates = gates[valid]
    v_correct = correct[valid]

    np.save(out / "gate_values.npy", gates)
    np.save(out / "correct_mask.npy", correct)

    if len(v_gates) == 0:
        print("\n  WARNING: All gate values are NaN.")
        summary = {"error": "all_nan", "n_total": len(gates)}
    else:
        summary = {
            "mean": float(v_gates.mean()),
            "std": float(v_gates.std()),
            "median": float(np.median(v_gates)),
            "min": float(v_gates.min()),
            "max": float(v_gates.max()),
            "n_valid": int(len(v_gates)),
            "n_nan": int((~valid).sum()),
            "below_0.3": float((v_gates < 0.3).mean()),
            "between_0.3_0.7": float(((v_gates >= 0.3) & (v_gates < 0.7)).mean()),
            "above_0.7": float((v_gates >= 0.7).mean()),
            "accuracy_low_gate": float(v_correct[v_gates < np.percentile(v_gates, 33.3)].mean()) if len(v_gates) > 2 else 0.0,
            "accuracy_high_gate": float(v_correct[v_gates >= np.percentile(v_gates, 66.6)].mean()) if len(v_gates) > 2 else 0.0,
        }

    with open(out / "gate_distribution.json", "w") as f:
        json.dump(summary, f, indent=2)

    if "mean" in summary:
        print(f"\nGate distribution ({summary.get('n_valid', 0)} valid / {len(gates)} total):")
        print(f"  Mean: {summary['mean']:.3f} ± {summary['std']:.3f}")
        print(f"  Low (<0.3): {summary['below_0.3']:.1%}")
        print(f"  Mid (0.3-0.7): {summary['between_0.3_0.7']:.1%}")
        print(f"  High (>0.7): {summary['above_0.7']:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Stratified conflict analysis for HCF-Net.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--manifest", required=True, help="Path to manifest.csv")
    parser.add_argument("--ablation-checkpoint", default=None,
                        help="Optional alignment-only ablation checkpoint for comparison")
    parser.add_argument("--output-dir", default="./conflict_analysis")
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

    print(f"Running inference on {len(test_idx)} test utterances ...")
    results = extract_predictions_and_gates(model, loader, device)

    if results[0]["gate"] is None:
        print("ERROR: Model did not return gate values. Check model architecture.")
        return

    # Stratified analysis
    print("\n--- Full HCF-Net Model ---")
    metrics = compute_tertile_metrics(results)

    # Export histogram data
    export_gate_histogram_data(results, out)

    # Save full results
    with open(out / "stratified_metrics.json", "w") as f:
        json.dump({"full_model": metrics}, f, indent=2)

    # Ablation comparison (if checkpoint provided)
    if args.ablation_checkpoint:
        print("\n\n--- Alignment-Only Ablation ---")
        abl_ckpt = torch.load(args.ablation_checkpoint, map_location=device)
        abl_cfg = HCFNetConfig(**abl_ckpt["config"])
        abl_model = HCFNet(abl_cfg).to(device)
        abl_model.load_state_dict(abl_ckpt["model_state"])

        abl_results = extract_predictions_and_gates(abl_model, loader, device)
        if abl_results[0]["gate"] is not None:
            abl_metrics = compute_tertile_metrics(abl_results)
            # Save combined
            combined = {"full_model": metrics, "ablation_alignment_only": abl_metrics}
            with open(out / "stratified_metrics.json", "w") as f:
                json.dump(combined, f, indent=2)

    # Save predictions for further analysis
    with open(out / "test_predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll results saved to {out}/")
    print("Files: stratified_metrics.json, gate_distribution.json, gate_values.npy, test_predictions.json")


if __name__ == "__main__":
    main()
