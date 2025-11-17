"""PyTorch Lightning module that wraps the WhisperSeg fine-tuning procedure."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import LRSchedulerConfig, OptimizerLRScheduler
from transformers import (
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    get_linear_schedule_with_warmup,
)

from utils.metrics import (
    compute_detection_error_rate,
    compute_frame_f1,
    compute_segment_f1,
)


class WhisperSegLightning(pl.LightningModule):
    """Lightning module for WhisperSeg-style generative VAD."""

    def __init__(
        self,
        model_name_or_path: str = "openai/whisper-base",
        learning_rate: float = 3e-6,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        dropout: float = 0.1,
        total_spec_columns: int = 3000,
        freeze_encoder: bool = False,
        gradient_checkpointing: bool = False,
        label_smoothing: float = 0.0,
        scheduler_type: str = "linear",
        num_training_steps: Optional[int] = None,
        ignore_cluster: bool = False,
        cluster_codebook: Optional[Dict[str, int]] = None,
        val_max_gen_length: int = 256,
        val_early_stopping: bool = True,
        val_num_beams: int = 1,
        val_batch_metrics: bool = True,
        val_use_cache: bool = True,
        val_length_margin: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cluster_codebook = cluster_codebook or {"Vocal": 0}
        self.id_to_cluster = {v: k for k, v in self.cluster_codebook.items()}
        self.total_spec_columns = total_spec_columns
        self.dropout_rate = dropout
        self.freeze_encoder_flag = freeze_encoder
        self.gradient_checkpoint_flag = gradient_checkpointing
        self.val_max_length = val_max_gen_length
        self.val_num_beams = val_num_beams
        self.val_early_stopping = val_early_stopping
        self.val_use_cache = val_use_cache
        self.val_length_margin = val_length_margin
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.num_training_steps = num_training_steps
        self.warmup_steps = warmup_steps
        self.model_name_or_path = model_name_or_path

        self.model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            model_name_or_path, language="english", task="transcribe"
        )

        self._add_special_tokens()
        self._configure_model()

        self.model.config.total_spec_columns = self.total_spec_columns
        self.model.config.dropout = self.dropout_rate
        self.model.config.cluster_codebook = self.cluster_codebook

        self.validation_outputs: List[Dict[str, Any]] = []
        self.spec_time_step = 0.01
        self.ratio_decoding_time_step = 1
        self.segment_pattern = re.compile(r"<\|(\d+)\|>(\d+)<\|(\d+)\|>")

        self.metrics_executor = (
            ThreadPoolExecutor(max_workers=max(torch.get_num_threads() // 2, 1))
            if val_batch_metrics
            else None
        )

    def _add_special_tokens(self):
        existing_vocab = set(self.tokenizer.get_vocab().keys())
        tokens_to_add: List[str] = []

        for i in range(self.total_spec_columns + 1):
            token = f"<|{i}|>"
            if token not in existing_vocab:
                tokens_to_add.append(token)

        if "<|human|>" not in existing_vocab:
            tokens_to_add.append("<|human|>")

        if tokens_to_add:
            self.tokenizer.add_tokens(tokens_to_add, special_tokens=True)
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.species_tokens = {"human": self.tokenizer.convert_tokens_to_ids("<|human|>")}

    def _configure_model(self):
        max_source_positions = int(0.5 * self.total_spec_columns)
        self.model.config.max_source_positions = max_source_positions

        with torch.no_grad():
            embed_positions = self.model.model.encoder.embed_positions.weight
            if embed_positions.shape[0] != max_source_positions:
                new_emb = torch.zeros(max_source_positions, embed_positions.shape[1])
                copy_len = min(embed_positions.shape[0], max_source_positions)
                new_emb[:copy_len] = embed_positions[:copy_len]
                self.model.model.encoder.embed_positions.weight = torch.nn.Parameter(new_emb)
                self.model.model.encoder.embed_positions.num_embeddings = max_source_positions

        self.model.config.dropout = self.dropout_rate
        self.model.model.encoder.dropout = self.dropout_rate
        self.model.model.decoder.dropout = self.dropout_rate

        if self.freeze_encoder_flag:
            for param in self.model.model.encoder.parameters():
                param.requires_grad = False

        if self.gradient_checkpoint_flag:
            self.model.gradient_checkpointing_enable()

    def forward(
        self,
        input_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
    ):
        return self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=True,
        )

    @lru_cache(maxsize=128)
    def _cached_decode(self, token_ids_tuple: tuple) -> str:
        return self.tokenizer.decode(list(token_ids_tuple), skip_special_tokens=False)

    def decode_predictions(self, token_ids: torch.Tensor) -> List[List[Dict]]:
        batch_segments = []
        token_ids_cpu = token_ids.detach().cpu().numpy()

        for ids in token_ids_cpu:
            valid_mask = (ids != self.tokenizer.pad_token_id) & (ids >= 0)
            ids_list = ids[valid_mask].tolist()

            if not ids_list:
                batch_segments.append([])
                continue

            text = self._cached_decode(tuple(ids_list))
            matches = self.segment_pattern.finditer(text)
            segments = []

            for match in matches:
                onset_idx = int(match.group(1))
                cluster_id = int(match.group(2))
                offset_idx = int(match.group(3))

                start_time = onset_idx * self.spec_time_step * self.ratio_decoding_time_step
                end_time = offset_idx * self.spec_time_step * self.ratio_decoding_time_step

                segments.append(
                    {
                        "start": start_time,
                        "end": end_time,
                        "cluster": self.id_to_cluster.get(cluster_id, "unknown"),
                    }
                )

            batch_segments.append(segments)

        return batch_segments

    def decode_labels(self, labels: torch.Tensor) -> List[List[Dict]]:
        batch_segments = []
        labels_cpu = labels.detach().cpu().numpy()

        for label_ids in labels_cpu:
            valid_mask = (label_ids != -100) & (label_ids != self.tokenizer.pad_token_id)
            ids_list = label_ids[valid_mask].tolist()

            if not ids_list:
                batch_segments.append([])
                continue

            text = self._cached_decode(tuple(ids_list))
            matches = self.segment_pattern.finditer(text)
            segments = []

            for match in matches:
                onset_idx = int(match.group(1))
                cluster_id = int(match.group(2))
                offset_idx = int(match.group(3))

                start_time = onset_idx * self.spec_time_step * self.ratio_decoding_time_step
                end_time = offset_idx * self.spec_time_step * self.ratio_decoding_time_step

                segments.append(
                    {
                        "start": start_time,
                        "end": end_time,
                        "cluster": self.id_to_cluster.get(cluster_id, "unknown"),
                    }
                )

            batch_segments.append(segments)

        return batch_segments

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self.forward(
            input_features=batch["input_features"],
            labels=batch["labels"],
            attention_mask=batch.get("attention_mask"),
            decoder_input_ids=batch.get("decoder_input_ids"),
        )
        loss = outputs.loss
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self.forward(
            input_features=batch["input_features"],
            labels=batch["labels"],
            attention_mask=batch.get("attention_mask"),
            decoder_input_ids=batch.get("decoder_input_ids"),
        )

        loss = outputs.loss
        label_lengths = (batch["labels"] != -100).sum(dim=1)
        max_label_length = label_lengths.max().item() if label_lengths.numel() > 0 else 64
        dynamic_max_length = min(
            max(max_label_length + self.val_length_margin, 32), self.val_max_length
        )

        batch_size = batch["input_features"].size(0)
        prompt_tokens = self.tokenizer.convert_tokens_to_ids(
            ["<|startoftranscript|>", "<|en|>", "<|transcribe|>"]
        )
        decoder_input_ids = torch.tensor(
            [prompt_tokens for _ in range(batch_size)], dtype=torch.long, device=self.device
        )

        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones(
                batch["input_features"].shape[0],
                batch["input_features"].shape[-1],
                device=batch["input_features"].device,
                dtype=torch.long,
            )

        generation_output = self.model.generate(
            input_features=batch["input_features"],
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            max_length=dynamic_max_length,
            num_beams=self.val_num_beams,
            early_stopping=self.val_early_stopping,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=self.val_use_cache,
            return_dict_in_generate=True,
        )

        if isinstance(generation_output, torch.Tensor):
            predicted_ids = generation_output
        elif isinstance(generation_output, tuple):
            predicted_ids = generation_output[0]
        elif isinstance(generation_output, dict):
            predicted_ids = generation_output.get("sequences")
            if predicted_ids is None:
                predicted_ids = next(iter(generation_output.values()))
            predicted_ids = torch.as_tensor(predicted_ids)
        elif hasattr(generation_output, "sequences"):
            predicted_ids = torch.as_tensor(generation_output.sequences)
        else:
            raise TypeError("Unexpected generate output type")

        predicted_segments = self.decode_predictions(predicted_ids)
        ground_truth_segments = self.decode_labels(batch["labels"])
        audio_duration = self.total_spec_columns * self.spec_time_step

        if self.metrics_executor and len(predicted_segments) > 1:
            futures = [
                self.metrics_executor.submit(
                    self.compute_metrics_parallel, pred, gt, audio_duration
                )
                for pred, gt in zip(predicted_segments, ground_truth_segments)
            ]
            batch_metrics = [future.result() for future in futures]
        else:
            batch_metrics = [
                self.compute_metrics_parallel(pred, gt, audio_duration)
                for pred, gt in zip(predicted_segments, ground_truth_segments)
            ]

        self.validation_outputs.append({"loss": loss.detach(), "batch_metrics": batch_metrics})
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def compute_metrics_parallel(
        self, pred_segs: List[Dict], gt_segs: List[Dict], audio_duration: float
    ) -> Dict[str, float]:
        segment_metrics = compute_segment_f1(pred_segs, gt_segs)
        frame_metrics = compute_frame_f1(pred_segs, gt_segs, duration=audio_duration, frame_length=0.01)
        der = compute_detection_error_rate(pred_segs, gt_segs, duration=audio_duration)

        return {
            "segment_f1": segment_metrics["f1"],
            "segment_precision": segment_metrics["precision"],
            "segment_recall": segment_metrics["recall"],
            "frame_f1": frame_metrics["f1"],
            "frame_precision": frame_metrics["precision"],
            "frame_recall": frame_metrics["recall"],
            "der": der,
            "n_pred": len(pred_segs),
            "n_gt": len(gt_segs),
        }

    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return

        all_metrics = []
        for output in self.validation_outputs:
            all_metrics.extend(output["batch_metrics"])

        if all_metrics:
            mean_segment_f1 = float(np.mean([m["segment_f1"] for m in all_metrics]))
            mean_segment_precision = float(np.mean([m["segment_precision"] for m in all_metrics]))
            mean_segment_recall = float(np.mean([m["segment_recall"] for m in all_metrics]))
            mean_frame_f1 = float(np.mean([m["frame_f1"] for m in all_metrics]))
            mean_frame_precision = float(np.mean([m["frame_precision"] for m in all_metrics]))
            mean_frame_recall = float(np.mean([m["frame_recall"] for m in all_metrics]))
            mean_der = float(np.mean([m["der"] for m in all_metrics]))

            total_pred = sum(m["n_pred"] for m in all_metrics)
            total_gt = sum(m["n_gt"] for m in all_metrics)

            self.log("val/segment_f1", mean_segment_f1, sync_dist=True)
            self.log("val/segment_precision", mean_segment_precision, sync_dist=True)
            self.log("val/segment_recall", mean_segment_recall, sync_dist=True)
            self.log("val/frame_f1", mean_frame_f1, sync_dist=True)
            self.log("val/frame_precision", mean_frame_precision, sync_dist=True)
            self.log("val/frame_recall", mean_frame_recall, sync_dist=True)
            self.log("val/der", mean_der, sync_dist=True)
            self.log("val/total_predictions", float(total_pred), sync_dist=True)
            self.log("val/total_ground_truth", float(total_gt), sync_dist=True)

        self.validation_outputs.clear()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if self.scheduler_type != "linear" or self.num_training_steps is None:
            return optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        scheduler_config = cast(
            LRSchedulerConfig,
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        )

        return [optimizer], [scheduler_config]

    def on_save_checkpoint(self, checkpoint):
        checkpoint["whisperseg_config"] = {
            "total_spec_columns": self.total_spec_columns,
            "cluster_codebook": self.cluster_codebook,
            "dropout": self.dropout_rate,
            "model_name_or_path": self.model_name_or_path,
        }

    def on_load_checkpoint(self, checkpoint):
        if "whisperseg_config" in checkpoint:
            config = checkpoint["whisperseg_config"]
            self.total_spec_columns = config.get("total_spec_columns", 1000)
            self.cluster_codebook = config.get("cluster_codebook", {"Vocal": 0})
            self.id_to_cluster = {v: k for k, v in self.cluster_codebook.items()}

    def __del__(self):
        if hasattr(self, "metrics_executor") and self.metrics_executor:
            self.metrics_executor.shutdown(wait=False)
