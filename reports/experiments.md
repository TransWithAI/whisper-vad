# Experiment Performance Summary

This note consolidates the validation metrics logged on the shared evaluation dataset for all architectures that have matching runs in `logs/`. Raw scalars and full curves were exported to `reports/metrics_summary.json` and `reports/metric_curves.csv` for future plotting or automated README generation.

## Primary metric snapshot

| Architecture | Final F1 (or DER) | Best F1/DER | Step of best | Curve takeaway |
| --- | --- | --- | --- | --- |
| DETR-VAD Base | Detection F1 **0.270** | 0.270 @ 2,981 | 2,981 | Curve is flat after step 2.5k; precision stalls at ~0.17 despite recall gains. |
| Encoder-Only Decoder Base | Classification F1 **0.925** | 0.927 @ 4,259 | 4,259 | F1 plateaus after step 4k; early spike in precision (~0.973) at step 851. |
| Encoder-Only Decoder Large v2 | Classification F1 **0.937** | 0.937 @ 4,259 | 4,259 | Slightly higher headroom than base; precision dips late as loss climbs. |
| Encoder-Only Linear Base | Classification F1 **0.855** | 0.855 @ 4,046 | 4,046 | Gains are slow but steady; accuracy tops out just under 0.89. |
| WhisperSeg Base | Segment F1 **0.669**, Frame F1 **0.790** | Seg 0.704 & Frame 0.823 @ 4,254 | 4,254 | Both segment & frame curves peak mid-training, DER bottoms at step 5,105. |

*DET R reports DER-style metrics indirectly, so the detection-F1 is used as the primary comparison metric. WhisperSeg also reports DER (lower is better); at step 5,105 DER reaches **0.031** before regressing to 0.042 by the final checkpoint.*

## Detailed findings by architecture

### DETR-VAD Base
- **Final detection metrics:** precision 0.173, recall 0.610, F1 0.270 at step 3,620.
- **Curve behavior:** recall improves consistently (∆+0.15 from start), but precision remains low, limiting F1. Best F1 (0.2704) happens slightly before training ends, implying little benefit from further epochs.
- **Loss trajectory:** validation loss bottoms near 1.71 at step 2,981 before inching back up, matching the F1 plateau.

### Encoder-Only Decoder Base
- **Final metrics:** loss 0.051, accuracy 0.935, precision 0.956, recall 0.896, F1 0.925, AUC 0.985.
- **Curve highlights:**
  - Precision peaks very early (0.973 @ step 851) then gently declines while recall keeps improving to 0.907 @ 4,259.
  - Best F1 (0.927) also occurs at 4,259, after which the curve flattens; the final checkpoint is just 0.0026 lower.
  - Loss shows a sharp early drop (0.018 @ step 425) before stabilizing ~0.05, suggesting earlier checkpoints may offer a better calibration/overfit balance.

### Encoder-Only Decoder Large v2
- **Final metrics:** loss 0.088, accuracy 0.946, precision 0.945, recall 0.930, F1 0.937, AUC 0.987.
- **Curve highlights:**
  - Compared with the base model, Large v2 maintains higher recall throughout and a marginally better F1 peak (0.93745).
  - Precision softens after step 6k while loss creeps upward, indicating mild overfitting; best loss (0.020) occurs extremely early (step 851).
  - Because the final and best F1 differ by <0.0002, late checkpoints remain safe defaults, but snapshotting around step 4.2k maximizes both precision and recall simultaneously.

### Encoder-Only Linear Base
- **Final metrics:** loss 0.026, accuracy 0.888, precision 0.968, recall 0.766, F1 0.855, AUC 0.967.
- **Curve highlights:**
  - Precision stays high (>0.95) throughout, but recall is the main limiter; it climbs 0.11 absolute points over training with best values right before the final epoch.
  - F1 improves steadily then plateaus at ~0.855 between steps 4k–4.3k; earlier checkpoints underperform primarily due to weaker recall.
  - This model therefore offers a high-precision, moderate-recall operating point that might be useful when false positives are more costly.

### WhisperSeg Base
- **Final segment metrics:** precision 0.711, recall 0.656, F1 0.669.
- **Final frame metrics:** precision 0.808, recall 0.795, F1 0.790; DER 0.0418.
- **Curve highlights:**
  - Segment-level F1 hits 0.704 at step 4,254 before decaying as both precision and recall drop, pointing to overfitting in the second half of training.
  - Frame-level metrics follow a similar arc with best F1 0.823 at step 4,254 and best DER 0.031 at step 5,105.
  - When exporting checkpoints, consider capturing the mid-training snapshot to retain the stronger DER/F1 trade-off.

## Next steps
- Use `reports/metric_curves.csv` to plot comparative F1/precision/recall curves for the README (e.g., via seaborn line plots of `value` vs `step`).
- If a single checkpoint per architecture is needed, prefer:
  - DETR: final epoch (no significant differences).
  - Encoder-only models: ~step 4,200 snapshot for best F1.
  - WhisperSeg: step 4,254 for best segment/frame F1 or step 5,105 for minimum DER, depending on the target metric.
