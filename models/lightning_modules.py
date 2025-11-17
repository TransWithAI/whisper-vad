"""
PyTorch Lightning modules for training VAD models.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, List, Optional, Any
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from .encoder_only_vad import (
    WhisperEncoderVAD,
    WhisperEncoderLightweightDecoderVAD,
    FocalLoss
)
from .detr_vad import WhisperDETRVAD, SetCriterion, HungarianMatcher


class EncoderOnlyVADModule(pl.LightningModule):
    """
    Lightning module for Encoder-Only VAD models (Refinement I).
    """

    def __init__(
        self,
        model_type: str = "linear",  # "linear" or "lightweight_decoder"
        whisper_model_name: str = "openai/whisper-base",
        freeze_encoder: bool = False,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 2,
        max_epochs: int = 50,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        decoder_layers: int = 2,
        decoder_heads: int = 8,
        decoder_ff_dim: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize model based on type
        if model_type == "linear":
            self.model = WhisperEncoderVAD(
                whisper_model_name=whisper_model_name,
                freeze_encoder=freeze_encoder,
                dropout=dropout
            )
        else:  # lightweight_decoder
            self.model = WhisperEncoderLightweightDecoderVAD(
                whisper_model_name=whisper_model_name,
                freeze_encoder=freeze_encoder,
                decoder_layers=decoder_layers,
                decoder_heads=decoder_heads,
                decoder_ff_dim=decoder_ff_dim,
                dropout=dropout
            )

        # Loss function
        self.criterion = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma
        )

        # Metrics storage
        self.validation_outputs = []
        self.validation_targets = []

    def forward(self, input_features):
        """Forward pass."""
        if hasattr(self.model, 'forward'):
            output = self.model(input_features)
            if isinstance(output, tuple):
                return output[0]  # Return only frame logits
        return self.model(input_features)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        input_features = batch["input_features"]
        targets = batch["frame_targets"]

        # Forward pass
        logits = self.forward(input_features)

        # Compute loss
        loss = self.criterion(logits, targets)

        # Log metrics
        self.log("train/loss", loss, prog_bar=True)

        # Calculate and log accuracy
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == targets).float().mean()
            self.log("train/accuracy", acc, prog_bar=True)

        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        """Validation step."""
        input_features = batch["input_features"]
        targets = batch["frame_targets"]

        # Forward pass
        logits = self.forward(input_features)

        # Compute loss
        loss = self.criterion(logits, targets)

        # Store outputs for epoch-end metrics
        probs = torch.sigmoid(logits)
        self.validation_outputs.append(probs.detach().cpu())
        self.validation_targets.append(targets.detach().cpu())

        # Log loss
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        # Calculate and log accuracy
        preds = (probs > 0.5).float()
        acc = (preds == targets).float().mean()
        self.log("val/accuracy", acc, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        """Calculate epoch-level validation metrics."""
        if not self.validation_outputs:
            return

        # Concatenate all outputs
        # Convert to float32 first to handle BFloat16 tensors
        all_outputs = torch.cat(self.validation_outputs, dim=0).float().cpu().numpy()
        all_targets = torch.cat(self.validation_targets, dim=0).float().cpu().numpy()

        # Clear storage
        self.validation_outputs.clear()
        self.validation_targets.clear()

        # Flatten for frame-level metrics
        outputs_flat = all_outputs.flatten()
        targets_flat = all_targets.flatten()

        # Calculate metrics
        preds_flat = (outputs_flat > 0.5).astype(int)

        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets_flat, preds_flat, average='binary'
        )

        # AUC-ROC
        try:
            auc_roc = roc_auc_score(targets_flat, outputs_flat)
        except:
            auc_roc = 0.0

        # Log metrics
        self.log("val/precision", precision, sync_dist=True)
        self.log("val/recall", recall, sync_dist=True)
        self.log("val/f1", f1, sync_dist=True)
        self.log("val/auc_roc", auc_roc, sync_dist=True)

        print(f"\nValidation Metrics - Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc_roc:.4f}")

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        # Create warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.hparams.warmup_epochs
        )

        # Create cosine annealing scheduler for remaining epochs
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
            eta_min=self.hparams.learning_rate * 0.01
        )

        # Combine schedulers
        scheduler = {
            'scheduler': SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.hparams.warmup_epochs]
            ),
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [scheduler]


class DETRVADModule(pl.LightningModule):
    """
    Lightning module for DETR-style VAD model (Refinement II).
    """

    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-base",
        num_queries: int = 20,
        decoder_layers: int = 3,
        decoder_heads: int = 8,
        freeze_encoder: bool = False,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        cost_class: float = 1.0,
        cost_span: float = 5.0,
        cost_giou: float = 2.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize model
        self.model = WhisperDETRVAD(
            whisper_model_name=whisper_model_name,
            num_queries=num_queries,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            freeze_encoder=freeze_encoder,
            dropout=dropout
        )

        # Initialize matcher and criterion
        self.matcher = HungarianMatcher(
            cost_class=cost_class,
            cost_span=cost_span,
            cost_giou=cost_giou
        )

        self.criterion = SetCriterion(
            matcher=self.matcher,
            weight_dict={
                'loss_ce': cost_class,
                'loss_span': cost_span,
                'loss_giou': cost_giou
            }
        )

        # Metrics storage
        self.validation_outputs = []
        self.validation_targets = []

    def forward(self, input_features):
        """Forward pass."""
        return self.model(input_features)

    def prepare_targets(self, batch: Dict) -> List[Dict]:
        """Prepare targets for DETR loss computation."""
        targets = []
        batch_size = batch["span_labels"].size(0)

        for i in range(batch_size):
            # Get valid spans (where label == 1)
            valid_mask = batch["span_labels"][i] == 1
            valid_labels = batch["span_labels"][i][valid_mask]
            valid_spans = batch["span_coords"][i][valid_mask]

            targets.append({
                "labels": valid_labels,
                "spans": valid_spans
            })

        return targets

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        input_features = batch["input_features"]
        targets = self.prepare_targets(batch)

        # Forward pass
        outputs = self.forward(input_features)

        # Compute losses
        losses = self.criterion(outputs, targets)

        # Total loss
        total_loss = sum(losses.values())

        # Log metrics
        self.log("train/loss", total_loss, prog_bar=True)
        for k, v in losses.items():
            self.log(f"train/{k}", v)

        return total_loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        """Validation step."""
        input_features = batch["input_features"]
        targets = self.prepare_targets(batch)

        # Forward pass
        outputs = self.forward(input_features)

        # Compute losses
        losses = self.criterion(outputs, targets)
        total_loss = sum(losses.values())

        # Store outputs for metrics
        self.validation_outputs.append(outputs)
        self.validation_targets.append(targets)

        # Log metrics
        self.log("val/loss", total_loss, prog_bar=True, sync_dist=True)
        for k, v in losses.items():
            self.log(f"val/{k}", v, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        """Calculate epoch-level validation metrics."""
        if not self.validation_outputs:
            return

        # Calculate span-level metrics
        all_pred_spans = []
        all_true_spans = []

        for outputs, targets_batch in zip(self.validation_outputs, self.validation_targets):
            batch_size = outputs["pred_logits"].size(0)

            for b in range(batch_size):
                # Get predicted spans (class 1 = speech)
                pred_probs = F.softmax(outputs["pred_logits"][b], dim=-1)
                pred_mask = pred_probs[:, 1] > 0.5
                pred_spans = outputs["pred_spans"][b][pred_mask]

                # Get true spans
                if b < len(targets_batch):
                    true_spans = targets_batch[b]["spans"]
                else:
                    true_spans = torch.tensor([])

                all_pred_spans.append(pred_spans.detach().cpu())
                all_true_spans.append(true_spans.detach().cpu())

        # Clear storage
        self.validation_outputs.clear()
        self.validation_targets.clear()

        # Calculate detection metrics
        precision, recall, f1 = self.calculate_detection_metrics(
            all_pred_spans, all_true_spans
        )

        # Log metrics
        self.log("val/detection_precision", precision, sync_dist=True)
        self.log("val/detection_recall", recall, sync_dist=True)
        self.log("val/detection_f1", f1, sync_dist=True)

        print(f"\nValidation Detection Metrics - "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    def calculate_detection_metrics(
        self,
        pred_spans: List[torch.Tensor],
        true_spans: List[torch.Tensor],
        iou_threshold: float = 0.5
    ) -> tuple:
        """Calculate precision, recall, and F1 for span detection."""
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for preds, trues in zip(pred_spans, true_spans):
            if len(preds) == 0 and len(trues) == 0:
                continue

            if len(preds) == 0:
                total_fn += len(trues)
                continue

            if len(trues) == 0:
                total_fp += len(preds)
                continue

            # Calculate IoU matrix
            iou_matrix = self.calculate_span_iou(preds, trues)

            # Hungarian matching for best assignment
            matched_preds = set()
            matched_trues = set()

            for _ in range(min(len(preds), len(trues))):
                if iou_matrix.numel() == 0:
                    break

                max_iou, max_idx = iou_matrix.max(), iou_matrix.argmax()
                if max_iou < iou_threshold:
                    break

                pred_idx = max_idx // len(trues)
                true_idx = max_idx % len(trues)

                if pred_idx not in matched_preds and true_idx not in matched_trues:
                    matched_preds.add(pred_idx.item())
                    matched_trues.add(true_idx.item())
                    total_tp += 1

                    # Remove matched entries
                    iou_matrix[pred_idx, :] = -1
                    iou_matrix[:, true_idx] = -1

            total_fp += len(preds) - len(matched_preds)
            total_fn += len(trues) - len(matched_trues)

        # Calculate metrics
        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        return precision, recall, f1

    def calculate_span_iou(
        self,
        spans1: torch.Tensor,
        spans2: torch.Tensor
    ) -> torch.Tensor:
        """Calculate IoU between two sets of spans."""
        if len(spans1) == 0 or len(spans2) == 0:
            return torch.tensor([])

        # Expand for broadcasting
        s1_start = spans1[:, 0:1]  # (N, 1)
        s1_end = spans1[:, 1:2]     # (N, 1)
        s2_start = spans2[:, 0:1].t()  # (1, M)
        s2_end = spans2[:, 1:2].t()    # (1, M)

        # Calculate intersection
        inter_start = torch.max(s1_start, s2_start)
        inter_end = torch.min(s1_end, s2_end)
        inter = (inter_end - inter_start).clamp(min=0)

        # Calculate union
        union = (s1_end - s1_start) + (s2_end - s2_start) - inter

        # Calculate IoU
        iou = inter / (union + 1e-6)

        return iou

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Different learning rates for encoder and decoder
        param_groups = [
            {"params": self.model.encoder.parameters(), "lr": self.hparams.learning_rate * 0.1},
            {"params": self.model.decoder.parameters(), "lr": self.hparams.learning_rate},
            {"params": self.model.event_queries.parameters(), "lr": self.hparams.learning_rate},
            {"params": self.model.class_head.parameters(), "lr": self.hparams.learning_rate},
            {"params": self.model.span_head.parameters(), "lr": self.hparams.learning_rate},
        ]

        optimizer = AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay
        )

        # Create warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,  # Start from 1% of target LR for DETR
            end_factor=1.0,
            total_iters=self.hparams.warmup_epochs
        )

        # Create cosine annealing scheduler for remaining epochs
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
            eta_min=self.hparams.learning_rate * 0.001  # Lower minimum for DETR
        )

        # Combine schedulers
        scheduler = {
            'scheduler': SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.hparams.warmup_epochs]
            ),
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [scheduler]