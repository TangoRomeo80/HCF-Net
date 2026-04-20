from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import HCFNetConfig
from .utils import chunk_sequence, score_to_class7


@dataclass
class SampleRecord:
    utterance_id: str
    split: str
    text_path: str
    audio_path: str
    visual_path: str
    score: float
    class7: int
    binary: Optional[int] = None


def _load_feature(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p)
    elif p.suffix in {".h5", ".hdf5"}:
        with h5py.File(p, "r") as f:
            # Expect a single main dataset named 'features' or the first key.
            if "features" in f:
                arr = f["features"][:]
            else:
                key = list(f.keys())[0]
                arr = f[key][:]
    else:
        raise ValueError(f"Unsupported feature file: {path}")

    if arr.ndim != 2:
        raise ValueError(f"Expected [T, D] features in {path}, got shape {arr.shape}")
    return arr.astype(np.float32)


class HCFNetFeatureDataset(Dataset):
    def __init__(self, records: list[SampleRecord], config: HCFNetConfig):
        self.records = records
        self.config = config

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        text = _load_feature(rec.text_path)
        audio = _load_feature(rec.audio_path)
        visual = _load_feature(rec.visual_path)

        # align modalities to a shared utterance timeline before training.
        # We enforce here for fidelity and simplicity.
        t_text, _ = text.shape
        t_audio, _ = audio.shape
        t_visual, _ = visual.shape
        t = min(t_text, t_audio, t_visual)
        if not (t_text == t_audio == t_visual):
            text = text[:t]
            audio = audio[:t]
            visual = visual[:t]

        text_seg, text_mask = chunk_sequence(text, self.config.segment_length)
        audio_seg, audio_mask = chunk_sequence(audio, self.config.segment_length)
        visual_seg, visual_mask = chunk_sequence(visual, self.config.segment_length)

        class7 = rec.class7 if rec.class7 is not None else score_to_class7(rec.score)
        binary = rec.binary if rec.binary is not None else int(class7 >= 3)

        return {
            "utterance_id": rec.utterance_id,
            "text": torch.from_numpy(text_seg),
            "audio": torch.from_numpy(audio_seg),
            "visual": torch.from_numpy(visual_seg),
            "text_mask": torch.from_numpy(text_mask),
            "audio_mask": torch.from_numpy(audio_mask),
            "visual_mask": torch.from_numpy(visual_mask),
            "class7": torch.tensor(class7, dtype=torch.long),
            "binary": torch.tensor(binary, dtype=torch.long),
            "score": torch.tensor(rec.score, dtype=torch.float32),
        }


def collate_hcfnet(batch: list[dict[str, Any]]) -> dict[str, Any]:
    # Variable number of segments S across utterances -> pad on S dimension.
    max_s = max(item["text"].shape[0] for item in batch)

    def pad_segments(x: torch.Tensor, max_s: int, pad_value: float = 0.0) -> torch.Tensor:
        s, l, d = x.shape
        if s == max_s:
            return x
        out = torch.full((max_s, l, d), pad_value, dtype=x.dtype)
        out[:s] = x
        return out

    def pad_masks(x: torch.Tensor, max_s: int) -> torch.Tensor:
        s, l = x.shape
        if s == max_s:
            return x
        out = torch.zeros((max_s, l), dtype=x.dtype)
        out[:s] = x
        return out

    out: dict[str, Any] = {
        "utterance_id": [item["utterance_id"] for item in batch],
        "text": torch.stack([pad_segments(item["text"], max_s) for item in batch], dim=0),
        "audio": torch.stack([pad_segments(item["audio"], max_s) for item in batch], dim=0),
        "visual": torch.stack([pad_segments(item["visual"], max_s) for item in batch], dim=0),
        "text_mask": torch.stack([pad_masks(item["text_mask"], max_s) for item in batch], dim=0),
        "audio_mask": torch.stack([pad_masks(item["audio_mask"], max_s) for item in batch], dim=0),
        "visual_mask": torch.stack([pad_masks(item["visual_mask"], max_s) for item in batch], dim=0),
        "class7": torch.stack([item["class7"] for item in batch], dim=0),
        "binary": torch.stack([item["binary"] for item in batch], dim=0),
        "score": torch.stack([item["score"] for item in batch], dim=0),
    }
    return out


def read_manifest(path: str) -> list[SampleRecord]:
    p = Path(path)
    if p.suffix == ".jsonl":
        records: list[SampleRecord] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                records.append(
                    SampleRecord(
                        utterance_id=str(row["utterance_id"]),
                        split=str(row["split"]),
                        text_path=str(row["text_path"]),
                        audio_path=str(row["audio_path"]),
                        visual_path=str(row["visual_path"]),
                        score=float(row["score"]),
                        class7=int(row.get("class7", score_to_class7(float(row["score"])))),
                        binary=None if row.get("binary") is None else int(row["binary"]),
                    )
                )
        return records

    if p.suffix == ".csv":
        records = []
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(
                    SampleRecord(
                        utterance_id=str(row["utterance_id"]),
                        split=str(row["split"]),
                        text_path=str(row["text_path"]),
                        audio_path=str(row["audio_path"]),
                        visual_path=str(row["visual_path"]),
                        score=float(row["score"]),
                        class7=int(row.get("class7") or score_to_class7(float(row["score"]))),
                        binary=None if row.get("binary") in {None, ""} else int(row["binary"]),
                    )
                )
        return records

    raise ValueError("Manifest must be .csv or .jsonl")
