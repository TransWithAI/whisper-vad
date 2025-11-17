"""
Refinement I: Encoder-Only VAD Models

This module implements two variants:
1. Encoder + Linear Head (simple frame-level classifier)
2. Encoder + Lightweight Decoder (ApneaWhisper-style with context aggregation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, WhisperConfig
from typing import Optional, Tuple
import math


class WhisperEncoderVAD(nn.Module):
    """
    Whisper Encoder + Linear Head for frame-level VAD.

    This is the simplest implementation of the encoder-only approach.
    The linear head is applied independently to each frame.
    """

    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-base",
        freeze_encoder: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()

        # Load pre-trained Whisper encoder
        whisper_model = WhisperModel.from_pretrained(whisper_model_name)
        self.encoder = whisper_model.encoder

        # Get hidden dimension from config
        self.hidden_dim = self.encoder.config.d_model

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classification_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: (batch_size, 80, 3000) - mel spectrogram features

        Returns:
            logits: (batch_size, 1500) - frame-level VAD logits
        """
        # Get encoder hidden states
        encoder_outputs = self.encoder(input_features)
        hidden_states = encoder_outputs.last_hidden_state  # (batch, 1500, hidden_dim)

        # Apply dropout
        hidden_states = self.dropout(hidden_states)

        # Apply linear head to each frame
        logits = self.classification_head(hidden_states)  # (batch, 1500, 1)

        # Squeeze last dimension
        return logits.squeeze(-1)  # (batch, 1500)


class LightweightTransformerDecoder(nn.Module):
    """
    Lightweight non-autoregressive Transformer decoder for context aggregation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Create decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout, max_len=1500)

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            encoder_hidden_states: (batch, seq_len, hidden_dim)
            memory_mask: Optional attention mask

        Returns:
            decoded_states: (batch, seq_len, hidden_dim)
        """
        # Add positional encoding
        tgt = self.pos_encoding(encoder_hidden_states)

        # Pass through decoder (using encoder output as both tgt and memory)
        # This allows the model to refine features with self-attention
        decoded = self.decoder(
            tgt=tgt,
            memory=encoder_hidden_states,
            memory_mask=memory_mask
        )

        return decoded


class WhisperEncoderLightweightDecoderVAD(nn.Module):
    """
    Whisper Encoder + Lightweight Decoder for frame-level VAD.

    This implements the ApneaWhisper-style architecture with a
    non-autoregressive decoder for context aggregation.
    """

    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-base",
        freeze_encoder: bool = False,
        decoder_layers: int = 2,
        decoder_heads: int = 8,
        decoder_ff_dim: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        # Load pre-trained Whisper encoder
        whisper_model = WhisperModel.from_pretrained(whisper_model_name)
        self.encoder = whisper_model.encoder

        # Get hidden dimension from config
        self.hidden_dim = self.encoder.config.d_model

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Lightweight decoder for context aggregation
        self.decoder = LightweightTransformerDecoder(
            hidden_dim=self.hidden_dim,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            ff_dim=decoder_ff_dim,
            dropout=dropout
        )

        # Classification heads
        self.frame_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1)
        )

        # Optional: Clip-level classification head
        self.clip_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(
        self,
        input_features: torch.Tensor,
        return_clip_prediction: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_features: (batch_size, 80, 3000) - mel spectrogram features
            return_clip_prediction: Whether to return clip-level prediction

        Returns:
            frame_logits: (batch_size, 1500) - frame-level VAD logits
            clip_logits: (batch_size,) - clip-level VAD logits (optional)
        """
        # Get encoder hidden states
        encoder_outputs = self.encoder(input_features)
        hidden_states = encoder_outputs.last_hidden_state  # (batch, 1500, hidden_dim)

        # Pass through lightweight decoder for context aggregation
        decoded_states = self.decoder(hidden_states)  # (batch, 1500, hidden_dim)

        # Frame-level classification
        frame_logits = self.frame_classifier(decoded_states).squeeze(-1)  # (batch, 1500)

        clip_logits = None
        if return_clip_prediction:
            # Global average pooling for clip-level prediction
            pooled = decoded_states.mean(dim=1)  # (batch, hidden_dim)
            clip_logits = self.clip_classifier(pooled).squeeze(-1)  # (batch,)

        return frame_logits, clip_logits


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in VAD.

    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch, seq_len) - logits
            targets: (batch, seq_len) - binary targets

        Returns:
            loss: scalar tensor
        """
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)

        # Calculate focal weights
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        # Apply alpha weighting
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss