# HCF-Net reference implementation (PyTorch)

This is a faithful, implementation-first version of the architecture described in the paper:
**HCF-Net: Hierarchical and Conflict-Aware Fusion for Robust Multimodal Sentiment Understanding**.

## What is implemented

- modality-specific projections to a shared 256-d space
- hierarchical graph encoder per modality
  - local intra-segment multi-head self-attention
  - mean pooling to segment nodes
  - global inter-segment multi-head self-attention
  - utterance-level mean pooling
- conflict-aware dual-branch fusion
  - alignment branch: 2 cross-modal transformer-style blocks
  - residual branch: low-rank shared-subspace removal with `k=32`
  - scalar gate
  - consistency-discrepancy loss
- task heads
  - 7-way classification head
  - regression head
  - optional binary head
- training loop with
  - AdamW
  - ReduceLROnPlateau
  - gradient clipping
  - early stopping
  - controlled oversampling toward 1000 examples per class
  - modality-drop

## Expected data format

Create a CSV or JSONL manifest with columns/keys:

- `utterance_id`
- `split`  (`train`, `validation`, `test`)
- `text_path`
- `audio_path`
- `visual_path`
- `score`   (continuous sentiment in [-3, 3])
- `class7`  (optional, integer in [0, 6], where 0=-3 and 6=+3)
- `binary`  (optional, 0/1)

Each feature file should store a **2D tensor** `[T, D]`:

- text: `[T, 768]`
- audio: `[T, 1024]`
- visual: `[T, 2048]`

Supported formats:
- `.npy`
- `.h5` / `.hdf5`

The implementation assumes the three modalities are already aligned to a common utterance timeline before training, which is exactly how the paper describes its training pipeline.

## Run training

```bash
python scripts/train.py \
  --manifest /path/to/manifest.csv \
  --output-dir /path/to/output \
  --batch-size 64 \
  --epochs 40 \
  --segment-length 32 \
  --early-stop-monitor mae
```

## Package requirements

```bash
pip install torch torchvision torchaudio numpy h5py scikit-learn
```

If you want raw-feature extraction as well, add:

```bash
pip install transformers sentencepiece librosa opencv-python mtcnn
```

## What this code does **not** do automatically

This repo intentionally focuses on the **model and training system**. It does not fully automate:

- raw transcript -> BERT extraction
- raw waveform -> wav2vec2.0 extraction
- raw video -> frame sampling + MTCNN face crops + ResNet-50 extraction
- CMU-MOSI SDK download / alignment scripts

Those are separate preprocessing stages and are best handled as dedicated scripts.
