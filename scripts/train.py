from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

from hcf_net.config import HCFNetConfig
from hcf_net.data import HCFNetFeatureDataset, collate_hcfnet, read_manifest
from hcf_net.losses import HCFNetLoss
from hcf_net.metrics import compute_metrics
from hcf_net.model import HCFNet
from hcf_net.utils import build_remainder_oversampled_indices, set_seed


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def run_epoch(model, loader, optimizer, loss_fn, device, train: bool):
    model.train(train)
    total_loss = 0.0
    all_outputs = []
    all_class7 = []
    all_scores = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        with torch.set_grad_enabled(train):
            outputs = model(
                text=batch["text"],
                audio=batch["audio"],
                visual=batch["visual"],
                text_mask=batch["text_mask"],
                audio_mask=batch["audio_mask"],
                visual_mask=batch["visual_mask"],
            )
            losses = loss_fn(
                outputs,
                class7_target=batch["class7"],
                regression_target=batch["score"],
                binary_target=batch["binary"],
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                losses.total.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += losses.total.item() * batch["class7"].shape[0]
        all_outputs.append({k: v.detach().cpu() for k, v in outputs.items() if torch.is_tensor(v)})
        all_class7.append(batch["class7"].detach().cpu())
        all_scores.append(batch["score"].detach().cpu())

    merged_outputs = {}
    for key in all_outputs[0].keys():
        merged_outputs[key] = torch.cat([o[key] for o in all_outputs], dim=0)

    class7 = torch.cat(all_class7, dim=0)
    scores = torch.cat(all_scores, dim=0)
    metrics = compute_metrics(merged_outputs, class7, scores)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--segment-length", type=int, default=32)
    parser.add_argument("--target-per-class", type=int, default=1000)
    parser.add_argument(
        "--early-stop-monitor",
        type=str,
        default="mae",
        choices=["mae", "acc7", "acc2", "macro_f1"],
    )
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = HCFNetConfig(segment_length=args.segment_length)
    records = read_manifest(args.manifest)
    dataset = HCFNetFeatureDataset(records, cfg)

    train_indices = [i for i, r in enumerate(records) if r.split.lower() == "train"]
    val_indices = [i for i, r in enumerate(records) if r.split.lower() in {"val", "valid", "validation"}]
    test_indices = [i for i, r in enumerate(records) if r.split.lower() == "test"]

    train_labels = [records[i].class7 for i in train_indices]
    oversampled_rel = build_remainder_oversampled_indices(train_labels, args.target_per_class)
    oversampled_train_indices = [train_indices[i] for i in oversampled_rel]

    train_ds = Subset(dataset, oversampled_train_indices)
    val_ds = Subset(dataset, val_indices)
    test_ds = Subset(dataset, test_indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_hcfnet)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_hcfnet)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_hcfnet)

    model = HCFNet(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min" if args.early_stop_monitor == "mae" else "max", factor=0.5, patience=5)
    loss_fn = HCFNetLoss(alpha_cd=cfg.alpha_cd)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_metric = None
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        val_loss, val_metrics = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False)

        monitor_value = {
            "mae": val_metrics.mae,
            "acc7": val_metrics.acc7,
            "acc2": val_metrics.acc2,
            "macro_f1": val_metrics.macro_f1,
        }[args.early_stop_monitor]

        if monitor_value is None:
            raise RuntimeError(f"Validation monitor {args.early_stop_monitor} is unavailable.")

        scheduler.step(monitor_value)

        improved = False
        if best_metric is None:
            improved = True
        elif args.early_stop_monitor == "mae":
            improved = monitor_value < best_metric
        else:
            improved = monitor_value > best_metric

        if improved:
            best_metric = monitor_value
            patience_left = args.patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg.__dict__,
                    "epoch": epoch,
                    "best_metric": best_metric,
                },
                output_dir / "best.pt",
            )
        else:
            patience_left -= 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_metrics.mae,
            "val_mse": val_metrics.mse,
            "val_r": val_metrics.pearson_r,
            "val_acc2": val_metrics.acc2,
            "val_acc7": val_metrics.acc7,
            "val_macro_f1": val_metrics.macro_f1,
            "val_weighted_f1": val_metrics.weighted_f1,
            "val_binary_f1": val_metrics.binary_f1,
            "val_auroc": val_metrics.auroc,
        }
        history.append(row)
        print(json.dumps(row))

        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch}")
            break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_metrics = run_epoch(model, test_loader, optimizer, loss_fn, device, train=False)

    summary = {
        "best_monitor": args.early_stop_monitor,
        "best_value": best_metric,
        "test_loss": test_loss,
        "test_mae": test_metrics.mae,
        "test_mse": test_metrics.mse,
        "test_r": test_metrics.pearson_r,
        "test_acc2": test_metrics.acc2,
        "test_acc7": test_metrics.acc7,
        "test_macro_f1": test_metrics.macro_f1,
        "test_weighted_f1": test_metrics.weighted_f1,
        "test_binary_f1": test_metrics.binary_f1,
        "test_auroc": test_metrics.auroc,
    }

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Test summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
