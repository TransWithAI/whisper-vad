"""WhisperSeg-compatible dataset utilities for generative VAD fine-tuning."""

from __future__ import annotations

import glob
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import pytorch_lightning as pl
import torch
from datasets import Dataset as HFDataset
from datasets import IterableDataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer

RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP = 2


class WhisperSegDataset(Dataset):
    """PyTorch dataset that produces WhisperSeg-style training targets."""

    def __init__(
        self,
        dataset: HFDataset,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: WhisperTokenizer,
        total_spec_columns: int = 1000,
        spec_time_step: float = 0.01,
        sampling_rate: int = 16000,
        spec_augment: bool = False,
        time_mask_param: int = 10,
        freq_mask_param: int = 27,
        species: str = "human",
        cluster_codebook: Optional[Dict[str, int]] = None,
        ignore_cluster: bool = False,
        max_length: int = 448,
    ):
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.total_spec_columns = total_spec_columns
        self.spec_time_step = spec_time_step
        self.sampling_rate = sampling_rate
        self.spec_augment = spec_augment
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.species = species if species == "human" else "human"
        self.ignore_cluster = ignore_cluster
        self.max_length = max_length

        self.cluster_codebook = cluster_codebook or {"Vocal": 0}
        self.max_duration_seconds = total_spec_columns * spec_time_step
        self.max_length_samples = int(self.max_duration_seconds * sampling_rate)
        self.species_token_id = tokenizer.convert_tokens_to_ids("<|human|>")

    def __len__(self) -> int:
        return len(self.dataset)

    def _time_to_spec_column_index(self, time_seconds: float) -> int:
        spec_column = int(
            np.round(time_seconds / self.spec_time_step / RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP)
        )
        return int(np.clip(spec_column, 0, self.total_spec_columns - 1))

    def _process_audio_length(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) > self.max_length_samples:
            audio = audio[: self.max_length_samples]
        elif len(audio) < self.max_length_samples:
            audio = np.pad(audio, (0, self.max_length_samples - len(audio)))
        return audio

    def _apply_spec_augment(self, features: torch.Tensor) -> torch.Tensor:
        if self.time_mask_param > 0:
            time_mask_len = torch.randint(0, self.time_mask_param, (1,)).item()
            if time_mask_len > 0 and features.shape[-1] > time_mask_len:
                time_start = torch.randint(0, features.shape[-1] - time_mask_len, (1,)).item()
                features[:, time_start : time_start + time_mask_len] = features.min()

        if self.freq_mask_param > 0:
            freq_mask_len = torch.randint(0, self.freq_mask_param, (1,)).item()
            if freq_mask_len > 0 and features.shape[-2] > freq_mask_len:
                freq_start = torch.randint(0, features.shape[-2] - freq_mask_len, (1,)).item()
                features[freq_start : freq_start + freq_mask_len, :] = features.min()

        return features

    def _create_labels_from_segments(self, segments: List[Dict]) -> torch.Tensor:
        label_text = f"<|{self.species}|>"

        for seg in segments:
            if "start_ms" in seg and "end_ms" in seg:
                start_seconds = seg["start_ms"] / 1000.0
                end_seconds = seg["end_ms"] / 1000.0
            elif "start" in seg and "end" in seg:
                start_seconds = seg["start"]
                end_seconds = seg["end"]
            else:
                continue

            onset_idx = self._time_to_spec_column_index(start_seconds)
            offset_idx = self._time_to_spec_column_index(end_seconds)

            if self.ignore_cluster:
                cluster_id = 0
            else:
                cluster_name = seg.get("cluster", seg.get("text", "Vocal"))
                if isinstance(cluster_name, str) and cluster_name.strip():
                    cluster_name = "Vocal" if cluster_name.strip() else "Vocal"
                cluster_id = self.cluster_codebook.get(cluster_name, 0)

            label_text += f"<|{onset_idx}|>{cluster_id}<|{offset_idx}|>"

        tokens = self.tokenizer.encode(label_text, add_special_tokens=True)
        labels = torch.tensor(tokens, dtype=torch.long)

        if len(labels) > self.max_length:
            labels = labels[: self.max_length]
        else:
            padding_length = self.max_length - len(labels)
            labels = torch.nn.functional.pad(labels, (0, padding_length), value=-100)

        return labels

    def _create_decoder_input_ids(self, labels: torch.Tensor) -> torch.Tensor:
        decoder_input_ids = labels.clone()
        decoder_input_ids[decoder_input_ids == -100] = self.tokenizer.pad_token_id
        decoder_input_ids[1:] = decoder_input_ids[:-1].clone()
        decoder_input_ids[0] = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        return decoder_input_ids

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]

        if "audio" in item and item["audio"] is not None:
            audio_array = item["audio"]["array"]
            audio_sr = item["audio"]["sampling_rate"]
        else:
            audio_array = item.get("array", np.zeros(self.max_length_samples))
            audio_sr = item.get("sampling_rate", self.sampling_rate)

        if audio_sr != self.sampling_rate:
            audio_array = librosa.resample(audio_array, orig_sr=audio_sr, target_sr=self.sampling_rate)

        audio_array = self._process_audio_length(audio_array)

        input_features = self.feature_extractor(
            audio_array, sampling_rate=self.sampling_rate, return_tensors="pt"
        ).input_features[0]

        if input_features.shape[-1] > self.total_spec_columns:
            input_features = input_features[:, : self.total_spec_columns]
        elif input_features.shape[-1] < self.total_spec_columns:
            min_val = input_features.min() if input_features.numel() > 0 else 0
            padding = self.total_spec_columns - input_features.shape[-1]
            pad_tensor = torch.full((input_features.shape[0], padding), min_val)
            input_features = torch.cat([input_features, pad_tensor], dim=-1)

        if self.spec_augment and getattr(self, "training", False):
            input_features = self._apply_spec_augment(input_features)

        labels = self._create_labels_from_segments(item.get("segments", []))
        decoder_input_ids = self._create_decoder_input_ids(labels)

        return {
            "input_features": input_features,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
        }


class WhisperSegStreamingDataset(torch.utils.data.IterableDataset):
    """Streaming dataset wrapper that shares logic with the map-style dataset."""

    def __init__(
        self,
        dataset: IterableDataset,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: WhisperTokenizer,
        total_spec_columns: int = 1000,
        spec_time_step: float = 0.01,
        sampling_rate: int = 16000,
        spec_augment: bool = False,
        time_mask_param: int = 10,
        freq_mask_param: int = 27,
        species: str = "human",
        cluster_codebook: Optional[Dict[str, int]] = None,
        ignore_cluster: bool = False,
        max_length: int = 448,
        estimated_length: Optional[int] = None,
    ):
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.total_spec_columns = total_spec_columns
        self.spec_time_step = spec_time_step
        self.sampling_rate = sampling_rate
        self.spec_augment = spec_augment
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.species = species if species == "human" else "human"
        self.ignore_cluster = ignore_cluster
        self.max_length = max_length
        self.estimated_length = estimated_length

        self.cluster_codebook = cluster_codebook or {"Vocal": 0}
        self.max_duration_seconds = total_spec_columns * spec_time_step
        self.max_length_samples = int(self.max_duration_seconds * sampling_rate)
        self.species_token_id = tokenizer.convert_tokens_to_ids("<|human|>")

        self._processor = WhisperSegDataset(
            dataset=[],
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            total_spec_columns=total_spec_columns,
            spec_time_step=spec_time_step,
            sampling_rate=sampling_rate,
            spec_augment=spec_augment,
            time_mask_param=time_mask_param,
            freq_mask_param=freq_mask_param,
            species=species,
            cluster_codebook=self.cluster_codebook,
            ignore_cluster=ignore_cluster,
            max_length=max_length,
        )

    def __len__(self):
        if self.estimated_length is None:
            raise NotImplementedError("Streaming dataset length not available")
        return self.estimated_length

    def __iter__(self):
        for item in self.dataset:
            if "audio" in item and item["audio"] is not None:
                audio_array = item["audio"]["array"]
                audio_sr = item["audio"]["sampling_rate"]
            else:
                audio_array = item.get("array", np.zeros(self.max_length_samples))
                audio_sr = item.get("sampling_rate", self.sampling_rate)

            if audio_sr != self.sampling_rate:
                audio_array = librosa.resample(audio_array, orig_sr=audio_sr, target_sr=self.sampling_rate)

            audio_array = self._processor._process_audio_length(audio_array)

            input_features = self.feature_extractor(
                audio_array, sampling_rate=self.sampling_rate, return_tensors="pt"
            ).input_features[0]

            if input_features.shape[-1] > self.total_spec_columns:
                input_features = input_features[:, : self.total_spec_columns]
            elif input_features.shape[-1] < self.total_spec_columns:
                min_val = input_features.min() if input_features.numel() > 0 else 0
                padding = self.total_spec_columns - input_features.shape[-1]
                pad_tensor = torch.full((input_features.shape[0], padding), min_val)
                input_features = torch.cat([input_features, pad_tensor], dim=-1)

            if self.spec_augment and getattr(self, "training", False):
                input_features = self._processor._apply_spec_augment(input_features)

            labels = self._processor._create_labels_from_segments(item.get("segments", []))
            decoder_input_ids = self._processor._create_decoder_input_ids(labels)

            yield {
                "input_features": input_features,
                "labels": labels,
                "decoder_input_ids": decoder_input_ids,
            }


class WhisperSegDataModule(pl.LightningDataModule):
    """Lightning DataModule that mirrors the original WhisperSeg pipeline."""

    def __init__(
        self,
        dataset_name_or_path: str,
        model_name_or_path: str = "openai/whisper-base",
        train_split: str = "train",
        val_split: str = "validation",
        test_split: str = "test",
        batch_size: int = 8,
        num_workers: int = 4,
        total_spec_columns: int = 1000,
        spec_time_step: float = 0.01,
        sampling_rate: int = 16000,
        spec_augment: bool = True,
        time_mask_param: int = 10,
        freq_mask_param: int = 27,
        species: str = "human",
        cluster_codebook: Optional[Dict[str, int]] = None,
        ignore_cluster: bool = False,
        cache_dir: Optional[str] = None,
        streaming: bool = False,
        preprocessing_num_workers: int = 4,
        train_samples: Optional[int] = None,
        val_samples: Optional[int] = None,
        test_samples: Optional[int] = None,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            model_name_or_path, language="english", task="transcribe"
        )
        self._add_special_tokens()
        self.cluster_codebook = cluster_codebook or {"Vocal": 0}

    def _add_special_tokens(self):
        existing_vocab = set(self.tokenizer.get_vocab().keys())
        tokens_to_add = []

        for i in range(self.hparams.total_spec_columns + 1):
            token = f"<|{i}|>"
            if token not in existing_vocab:
                tokens_to_add.append(token)

        if "<|human|>" not in existing_vocab:
            tokens_to_add.append("<|human|>")

        if tokens_to_add:
            self.tokenizer.add_tokens(tokens_to_add, special_tokens=True)

    def prepare_data(self):
        if not self.hparams.streaming:
            try:
                load_from_disk(self.hparams.dataset_name_or_path)
            except Exception:
                pass

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            dataset = self._load_dataset()
            if self.hparams.streaming:
                self.train_dataset = self._build_streaming_split(
                    dataset, self.hparams.train_split, True, self.hparams.train_samples
                )
                self.val_dataset = self._build_streaming_split(
                    dataset, self.hparams.val_split, False, self.hparams.val_samples
                )
            else:
                self.train_dataset = self._build_map_split(dataset, self.hparams.train_split, True)
                self.val_dataset = self._build_map_split(dataset, self.hparams.val_split, False)

        if stage == "test":
            dataset = self._load_dataset(test_only=True)
            if self.hparams.streaming:
                self.test_dataset = self._build_streaming_split(
                    dataset, self.hparams.test_split, False, self.hparams.test_samples
                )
            else:
                self.test_dataset = self._build_map_split(dataset, self.hparams.test_split, False)

    def _load_dataset(self, test_only: bool = False):
        if self.hparams.streaming:
            dataset_path = self.hparams.dataset_name_or_path
            data_files = {}

            if not test_only:
                train_files = glob.glob(f"{dataset_path}/train/*.arrow")
                if train_files:
                    data_files["train"] = train_files

                val_files = glob.glob(f"{dataset_path}/validation/*.arrow")
                if val_files:
                    data_files["validation"] = val_files

            test_files = glob.glob(f"{dataset_path}/test/*.arrow")
            if test_files:
                data_files["test"] = test_files

            if not data_files:
                raise ValueError(f"No arrow files found in {dataset_path}")

            return load_dataset("arrow", data_files=data_files, streaming=True)

        return load_from_disk(self.hparams.dataset_name_or_path)

    def _build_map_split(self, dataset, split_name: str, training: bool):
        if split_name not in dataset:
            raise ValueError(f"Split '{split_name}' not found in dataset")

        ds = WhisperSegDataset(
            dataset=dataset[split_name],
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            total_spec_columns=self.hparams.total_spec_columns,
            spec_time_step=self.hparams.spec_time_step,
            sampling_rate=self.hparams.sampling_rate,
            spec_augment=self.hparams.spec_augment,
            time_mask_param=self.hparams.time_mask_param,
            freq_mask_param=self.hparams.freq_mask_param,
            species=self.hparams.species,
            cluster_codebook=self.cluster_codebook,
            ignore_cluster=self.hparams.ignore_cluster,
        )
        ds.training = training
        return ds

    def _build_streaming_split(self, dataset, split_name: str, training: bool, est_length: Optional[int]):
        if split_name not in dataset:
            raise ValueError(f"Split '{split_name}' not found in dataset")

        ds = WhisperSegStreamingDataset(
            dataset=dataset[split_name],
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            total_spec_columns=self.hparams.total_spec_columns,
            spec_time_step=self.hparams.spec_time_step,
            sampling_rate=self.hparams.sampling_rate,
            spec_augment=self.hparams.spec_augment if training else False,
            time_mask_param=self.hparams.time_mask_param,
            freq_mask_param=self.hparams.freq_mask_param,
            species=self.hparams.species,
            cluster_codebook=self.cluster_codebook,
            ignore_cluster=self.hparams.ignore_cluster,
            estimated_length=est_length,
        )
        ds.training = training
        return ds

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=not self.hparams.streaming,
            drop_last=not self.hparams.streaming,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int):
        if isinstance(batch, dict):
            return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
