"""
Refinement II: DETR-style Set-Prediction VAD Model

This module implements a DETR-based architecture for direct set prediction
of speech spans, inspired by Sound Event Detection Transformer (SEDT).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional
import numpy as np


class WhisperDETRVAD(nn.Module):
    """
    Whisper Encoder + DETR-style Decoder for set-based VAD.

    This model directly predicts a set of speech spans without
    requiring frame-level predictions.
    """

    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-base",
        num_queries: int = 20,
        decoder_layers: int = 3,
        decoder_heads: int = 8,
        decoder_ff_dim: int = 2048,
        freeze_encoder: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()

        # Load pre-trained Whisper encoder
        whisper_model = WhisperModel.from_pretrained(whisper_model_name)
        self.encoder = whisper_model.encoder
        self.hidden_dim = self.encoder.config.d_model

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Learnable event queries
        self.num_queries = num_queries
        self.event_queries = nn.Embedding(num_queries, self.hidden_dim)

        # DETR-style Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for better stability
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers
        )

        # Positional encoding for encoder features
        self.pos_encoder = SinusoidalPositionalEncoding(self.hidden_dim)

        # Prediction heads
        self.class_head = MLP(self.hidden_dim, 256, 2, 3, dropout)  # Binary: speech/no-event
        self.span_head = MLP(self.hidden_dim, 256, 2, 3, dropout)   # Predict (start, end)

    def forward(self, input_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_features: (batch_size, 80, 3000) - mel spectrogram features

        Returns:
            Dict with:
                - pred_logits: (batch, num_queries, 2) - classification logits
                - pred_spans: (batch, num_queries, 2) - normalized span coordinates
        """
        batch_size = input_features.size(0)

        # Encode audio features
        encoder_outputs = self.encoder(input_features)
        memory = encoder_outputs.last_hidden_state  # (batch, 1500, hidden_dim)

        # Add positional encoding to encoder features
        memory = self.pos_encoder(memory)

        # Prepare queries
        query_embed = self.event_queries.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (batch, num_queries, hidden_dim)

        # Decode queries attending to encoder features
        hs = self.decoder(
            tgt=query_embed,
            memory=memory
        )  # (batch, num_queries, hidden_dim)

        # Apply prediction heads
        outputs_class = self.class_head(hs)  # (batch, num_queries, 2)
        outputs_coord = self.span_head(hs).sigmoid()  # (batch, num_queries, 2)

        return {
            'pred_logits': outputs_class,
            'pred_spans': outputs_coord
        }


class MLP(nn.Module):
    """Simple MLP with configurable layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_layers = num_layers

        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = self.dropout(F.relu(layer(x))) if i < self.num_layers - 1 else layer(x)
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1)]


class HungarianMatcher(nn.Module):
    """
    Hungarian matcher for bipartite matching between predictions and ground truth.

    This module computes an assignment between the targets and the predictions
    of the network using the Hungarian algorithm.
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_span: float = 5.0,
        cost_giou: float = 2.0
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            outputs: Dict with 'pred_logits' and 'pred_spans'
            targets: List of dicts with 'labels' and 'spans' per batch item

        Returns:
            List of (pred_indices, target_indices) matches
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten predictions
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # (bs*nq, 2)
        out_spans = outputs["pred_spans"].flatten(0, 1)  # (bs*nq, 2)

        # Concatenate target labels and spans
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_spans = torch.cat([v["spans"] for v in targets])

        # Compute classification cost
        cost_class = -out_prob[:, tgt_ids.long()]

        # Compute L1 cost between spans
        # Handle BFloat16 compatibility - cdist doesn't support BFloat16 on CUDA
        original_dtype = out_spans.dtype
        if original_dtype == torch.bfloat16:
            out_spans_float = out_spans.float()
            tgt_spans_float = tgt_spans.float()
            cost_span = torch.cdist(out_spans_float, tgt_spans_float, p=1)
            cost_span = cost_span.to(original_dtype)
        else:
            cost_span = torch.cdist(out_spans, tgt_spans, p=1)

        # Compute GIoU cost between spans
        cost_giou = -generalized_span_iou(
            spans_to_segments(out_spans),
            spans_to_segments(tgt_spans)
        )

        # Final cost matrix
        C = (
            self.cost_span * cost_span +
            self.cost_class * cost_class +
            self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu()

        # Perform Hungarian matching per batch element
        sizes = [len(v["labels"]) for v in targets]
        indices = []

        for i, c in enumerate(C.split(sizes, -1)):
            # Convert to float32 for scipy compatibility (BFloat16 not supported)
            c_float = c[i].float() if c[i].dtype == torch.bfloat16 else c[i]
            indices.append(linear_sum_assignment(c_float))

        return [(torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]


class SetCriterion(nn.Module):
    """
    Loss computation for DETR-style VAD model.

    The process is two-step:
    1. Compute Hungarian matching between predictions and ground truth
    2. Supervise each matched prediction with ground truth
    """

    def __init__(
        self,
        num_classes: int = 2,
        matcher: HungarianMatcher = None,
        weight_dict: Dict[str, float] = None,
        eos_coef: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher or HungarianMatcher()

        # Loss weights
        self.weight_dict = weight_dict or {
            'loss_ce': 1.0,
            'loss_span': 5.0,
            'loss_giou': 2.0
        }

        # Weight for no-object class
        empty_weight = torch.ones(self.num_classes)
        empty_weight[0] = eos_coef  # Assuming class 0 is "no event"
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple],
        num_spans: int
    ) -> Dict[str, torch.Tensor]:
        """Classification loss."""
        assert 'pred_logits' in outputs

        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([
            t["labels"][J] for t, (_, J) in zip(targets, indices)
        ])
        target_classes = torch.full(
            src_logits.shape[:2],
            0,  # no-event class
            dtype=torch.int64,
            device=src_logits.device
        )
        target_classes[idx] = target_classes_o.long()

        # Temporarily allow non-deterministic operations for cross_entropy
        # This is needed because cross_entropy doesn't have a deterministic
        # CUDA implementation for certain tensor configurations
        if torch.are_deterministic_algorithms_enabled():
            torch.use_deterministic_algorithms(False, warn_only=True)
            loss_ce = F.cross_entropy(
                src_logits.transpose(1, 2),
                target_classes,
                self.empty_weight
            )
            torch.use_deterministic_algorithms(True)
        else:
            loss_ce = F.cross_entropy(
                src_logits.transpose(1, 2),
                target_classes,
                self.empty_weight
            )

        losses = {'loss_ce': loss_ce}
        return losses

    def loss_spans(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple],
        num_spans: int
    ) -> Dict[str, torch.Tensor]:
        """Span regression loss."""
        assert 'pred_spans' in outputs

        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]
        target_spans = torch.cat([
            t['spans'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)

        loss_span = F.l1_loss(src_spans, target_spans, reduction='none')
        losses = {'loss_span': loss_span.sum() / num_spans}

        # Also compute GIoU loss
        src_segments = spans_to_segments(src_spans)
        target_segments = spans_to_segments(target_spans)
        loss_giou = 1 - torch.diag(generalized_span_iou(src_segments, target_segments))
        losses['loss_giou'] = loss_giou.sum() / num_spans

        return losses

    def _get_src_permutation_idx(self, indices):
        """Get permutation indices for matching."""
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Model outputs with 'pred_logits' and 'pred_spans'
            targets: List of ground truth dicts per batch

        Returns:
            Dict of losses
        """
        # Perform Hungarian matching
        indices = self.matcher(outputs, targets)

        # Count total number of spans for normalization
        num_spans = sum(len(t["labels"]) for t in targets)
        num_spans = torch.as_tensor([num_spans], dtype=torch.float,
                                   device=outputs["pred_logits"].device)
        num_spans = torch.clamp(num_spans, min=1)

        # Compute all losses
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_spans))
        losses.update(self.loss_spans(outputs, targets, indices, num_spans))

        # Apply loss weights
        weighted_losses = {}
        for k, v in losses.items():
            if k in self.weight_dict:
                weighted_losses[k] = v * self.weight_dict[k]

        return weighted_losses


def spans_to_segments(spans: torch.Tensor) -> torch.Tensor:
    """
    Convert normalized (start, end) spans to segment format.

    Args:
        spans: (N, 2) tensor with normalized [start, end]

    Returns:
        segments: (N, 2) tensor with [start, duration]
    """
    segments = spans.clone()
    segments[:, 1] = spans[:, 1] - spans[:, 0]  # duration = end - start
    return segments


def generalized_span_iou(spans1: torch.Tensor, spans2: torch.Tensor) -> torch.Tensor:
    """
    Compute generalized IoU between 1D spans.

    Args:
        spans1: (N, 2) tensor with [start, duration]
        spans2: (M, 2) tensor with [start, duration]

    Returns:
        giou: (N, M) tensor of GIoU values
    """
    # Convert to [start, end] format
    end1 = spans1[:, 0] + spans1[:, 1]
    end2 = spans2[:, 0] + spans2[:, 1]

    # Compute intersection
    inter_start = torch.max(spans1[:, 0:1], spans2[:, 0:1].t())
    inter_end = torch.min(end1[:, None], end2[None, :])
    inter = (inter_end - inter_start).clamp(min=0)

    # Compute union
    union = spans1[:, 1:2] + spans2[:, 1:2].t() - inter

    # Compute IoU
    iou = inter / (union + 1e-6)

    # Compute enclosing span
    enclosing_start = torch.min(spans1[:, 0:1], spans2[:, 0:1].t())
    enclosing_end = torch.max(end1[:, None], end2[None, :])
    enclosing = enclosing_end - enclosing_start

    # Compute GIoU
    giou = iou - (enclosing - union) / (enclosing + 1e-6)

    return giou