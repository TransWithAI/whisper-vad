# Whisper-Based Voice Activity Detection (VAD)

Real-world voice activity detection over long-form audio, powered by Whisper encoder refinements. This repo presents the architectural refinements proposed by this project—covering encoder-only, encoder+decoder, DETR-style, and WhisperSeg-inspired generative approaches—with a shared PyTorch Lightning training stack plus ONNX export/inference utilities. The codebase is maintained by the TransWithAI organization to enable reproducible VAD experimentation.

---

## Table of contents

1. [Highlights](#highlights)
2. [Repository layout](#repository-layout)
3. [Environment & installation](#environment--installation)
4. [Dataset schema](#dataset-schema)
5. [Configuration knobs](#configuration-knobs)
6. [Training & evaluation](#training--evaluation)
7. [Inference (PyTorch & ONNX)](#inference-pytorch--onnx)
8. [Experiment report](#experiment-report)
9. [Logging & monitoring](#logging--monitoring)
10. [Roadmap & references](#roadmap--references)

---

## Highlights

- ✅ Encoder-only linear head, encoder+lightweight decoder, DETR set-prediction, and WhisperSeg-like autoregressive fine-tuning in one code base
- ✅ Hugging Face `datasets` streaming, Lightning DataModules, and Silero-style post-processing utilities
- ✅ Extensive validation metrics: frame-level, segment-level, DER, ROC, AUC, plus TensorBoard/W&B logging
- ✅ Ready-to-export ONNX graphs with matching Silero-style inference wrapper for low-latency deployment
- ✅ Reproducible environment via `pyproject.toml`, `uv.lock`, and optional extras (`onnx`, `onnx-gpu`, `accelerate`)

## Repository layout

```
whisper_vad/
├── configs/                 # YAML configs per architecture
├── models/                  # LightningModules + DETR/encoder heads
├── utils/                   # Data pipeline, metrics, WhisperSeg datamodule
├── train.py                 # Main training CLI
├── inference.py             # PyTorch checkpoint inference & segment export
├── enc_dev_export_to_onnx.py# ONNX exporter for encoder+decoder models
├── onnx_inference.py        # Silero-style ONNX runtime
├── reports/{metrics_summary.json, metric_curves.csv, experiments.md} # Latest experiment digests
├── pyproject.toml / uv.lock # Project metadata & dependency pins
└── requirements.txt         # Minimal pip requirements
```

## Environment & installation

Python 3.13 is recommended (per `pyproject.toml`). Create a virtual environment and install dependencies:

```bash
# create + activate an isolated env (uv is fastest, but venv/conda works too)
uv venv
source .venv/bin/activate

# install the core requirements
uv pip install -r requirements.txt

# optional extras
uv pip install "whisper-vad[onnx]"       # CPU ONNX runtime
uv pip install "whisper-vad[onnx-gpu]"   # GPU ONNX runtime
uv pip install "whisper-vad[accelerate]" # DeepSpeed/Accelerate support
```

> **Torch builds:** the `pyproject` already points `uv`/pip to the CUDA 12.1 index. If you need a different CUDA/cuDNN combo, override `TORCH_CUDA_ARCH_LIST` or install the matching wheel manually.

## Dataset schema

All models share the same Hugging Face Arrow schema:

```python
from datasets import Audio, Features, Value

features = Features({
      "audio": Audio(sampling_rate=16_000),
      "segments": [
            {
                  "start_ms": Value("int32"),
                  "end_ms": Value("int32"),
                  "text": Value("string"),  # optional
            }
      ],
      "sample_rate": Value("int32"),
      # other task-specific metadata columns are passed through untouched
})
```

Specify the dataset location via `data.dataset_path` inside your chosen config. Both local Arrow files and `datasets.load_dataset` streaming sources are supported (set `data.streaming: true` and provide `train_dataset_length`/`val_dataset_length` for progress bars).

## Configuration knobs

Every architecture has a YAML file in `configs/`. Key sections:

- `model.*` – Whisper backbone (`openai/whisper-base` by default), decoder depth, dropout, number of DETR queries, etc.
- `data.*` – dataset path/name, streaming flags, dataloader workers, and optional `max_*_samples` for quick smoke tests.
- `training.*` – batch size, LR, scheduler choice, gradient accumulation, precision (`bf16-true`, `16`, `32`).
- `loss.*` – Focal loss parameters (`focal_alpha`, `focal_gamma`) or DETR matching costs (`cost_class`, `cost_span`, `cost_giou`).
- `monitor_metric`, `monitor_mode`, `save_top_k` – decide which metric drives checkpoints and early stopping.

Edit the YAML, then point the CLI at it. Example (`configs/encoder_decoder_base.yaml`):

```yaml
model:
   whisper_model_name: "openai/whisper-base"
   freeze_encoder: true
   decoder_layers: 2
training:
   batch_size: 128
   learning_rate: 1.5e-3
   max_epochs: 20
loss:
   focal_alpha: 0.25
   focal_gamma: 2.0
monitor_metric: "val/f1"
monitor_mode: "max"
```

## Training & evaluation

### CLI reference

```bash
uv run python train.py \
   --model {encoder_only_linear|encoder_only_decoder|detr|whisperseg} \
   --config configs/<your-config>.yaml \
   [--resume_from checkpoints/last.ckpt] \
   [--test]
```

What happens under the hood:

- Dataloaders are built via `utils.dataset.create_dataloaders` (encoder-only, DETR) or `utils.whisperseg_data.WhisperSegDataModule` (WhisperSeg).
- Callbacks: checkpoints, early stopping, LR monitor, rich progress bar & summary.
- Loggers: TensorBoard by default; set `logger: wandb` and `project_name` to enable W&B.

### Tips

1. **Batch sizing:** Lightning’s `accumulate_grad_batches` lets you simulate large batches if VRAM is tight.
2. **Precision:** `bf16-true` is the sweet spot on modern GPUs; fall back to `16` if you hit hardware limits.
3. **DET R patience:** set `training.max_epochs >= 100` and raise `early_stopping.patience` to avoid premature stops.
4. **Resume/multi-run:** every run writes to `checkpoints/<experiment_name>/`; pass `--resume_from` to continue training or to run evaluation only (`--test`).

## Inference (PyTorch & ONNX)

### PyTorch checkpoints

```bash
uv run python inference.py \
   --checkpoint checkpoints/encoder_decoder_base/epoch=14-val_f1=0.9274.ckpt \
   --model_type encoder_only_decoder \
   --audio samples/podcast.wav \
   --output runs/podcast_segments.json \
   --format json \
   --threshold 0.55 \
   --show_progress
```

- Outputs both console summaries and optional TXT/CSV/JSON/SRT files.
- Silero-style hysteresis parameters (`--min_speech_duration`, `--min_silence_duration`, `--speech_pad`, `--neg_threshold`) live here.
- DETR models fall back to span decoding via `predict_detr`, while encoder-only models emit frame-level probabilities that are post-processed into segments.

### ONNX export & runtime

1. **Export (encoder+decoder models):**

    ```bash
    uv run python enc_dev_export_to_onnx.py \
       --checkpoint checkpoints/encoder_decoder_base/best.ckpt \
       --config configs/encoder_decoder_base.yaml \
       --output artifacts/encoder_decoder_base.onnx \
       --opset 17
    ```

    This writes both `*.onnx` and `*_metadata.json` (input/output shapes, frame duration, etc.) and optionally verifies the graph with `onnxruntime`.

2. **Run ONNX inference (Silero-inspired):**

    ```bash
    uv run python onnx_inference.py \
       --model artifacts/encoder_decoder_base.onnx \
       --audio samples/podcast.wav \
       --output runs/podcast_segments.srt \
       --format srt \
       --threshold 0.6 \
       --min_silence_duration 150
    ```

    The script mirrors Silero’s chunked streaming logic, supports hysteresis thresholds, and can operate in real time via the `VADIterator` helper.

> **Need a ready-to-run exported model?** Grab the finetuned ONNX checkpoint plus an example inference script from [Hugging Face: TransWithAI / Whisper-Vad-EncDec-ASMR-onnx](https://huggingface.co/TransWithAI/Whisper-Vad-EncDec-ASMR-onnx). The repo bundles the latest encoder-decoder weights, matching metadata, and a ready-to-run Python example showing how to load the graph with ONNX Runtime.

## Experiment report

Latest TensorBoard scalars (exported via `reports/metrics_summary.json` + `reports/metric_curves.csv`). `val` metrics are reported on the shared evaluation dataset; F1 is the primary discriminator except for WhisperSeg, which also tracks DER.

| Architecture | Final metric | Peak metric (step) | Notes |
| --- | --- | --- | --- |
| DETR-VAD Base | Detection F1 **0.270** | 0.270 @ 2,981 | Recall climbs to 0.61 but precision stalls near 0.17; best loss (1.71) appears mid-training. |
| Encoder-Only Decoder Base | Frame F1 **0.925** | 0.927 @ 4,259 | Precision spikes early (0.973 @ 851) while recall keeps rising; export around step ~4.2k for balanced trade-off. |
| Encoder-Only Decoder Large v2 | Frame F1 **0.937** | 0.937 @ 4,259 | Slight boost over base; precision dips after 6k steps, so mid-training checkpoints avoid overfitting. |
| Encoder-Only Linear Base | Frame F1 **0.855** | 0.855 @ 4,046 | High-precision (0.97) / moderate-recall (0.77) operating point suitable when false positives are costly. |
| WhisperSeg Base | Segment F1 **0.669** (best 0.704 @ 4,254) / Frame F1 **0.790** (best 0.823 @ 4,254); DER **0.042** final (**0.031** @ 5,105) | Mid-training checkpoints dominate; both segment + frame metrics soften in the final third of training. |

See `reports/experiments.md` for deeper per-model curve commentary and checkpoint recommendations.

## Logging & monitoring

- **TensorBoard:** default logger writes to `logs/<run_name>/<version>/`. Launch with `tensorboard --logdir logs` or `uv run tensorboard --logdir logs`.
- **Weights & Biases:** set `logger: wandb`, fill `project_name`, and optionally `experiment_name` inside the config.
- **Checkpoints:** `ModelCheckpoint` stores `checkpoints/<experiment_name>/{epoch}-{metric:.4f}.ckpt` plus `last.ckpt`.
- **Metrics tooling:** use `reports/metric_curves.csv` with pandas/seaborn to plot cross-model F1/precision/recall evolution.

## Roadmap & references

- [ ] Multi-lingual fine-tuning recipes (Whisper large-v2, small).
- [ ] Native streaming dataloaders for online datasets.
- [ ] Quantization + INT8/INT4 export for edge deployment.

If you build on this work, please cite the `whisper-vad` repository (see [Citation](#citation)) and link back here so others can follow the architectural refinements outlined in this project. Contributions and issue reports are welcome!

## Citation

Please cite the GitHub repository directly when referencing the implementation (maintained by the TransWithAI organization):

```bibtex
@software{whisper_vad_2025,
   title        = {whisper-vad: Whisper-Based Voice Activity Detection Refinements},
   author       = {TransWithAI contributors},
   year         = {2025},
   url          = {https://github.com/asmrone/whisper-vad},
   version      = {0.1.0},
   note         = {GitHub repository}
}
```