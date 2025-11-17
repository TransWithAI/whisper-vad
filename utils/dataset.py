"""
Dataset loader for HuggingFace Arrow format with audio and segments.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset, Features, Audio, Value
import numpy as np
from transformers import WhisperFeatureExtractor
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


class WhisperVADDataset(IterableDataset):
    """
    Stream-iterable dataset for VAD training with Whisper encoder.

    Expects dataset with features:
    - audio: Audio object
    - segments: List of {"start_ms", "end_ms", "text"}
    - sample_rate: int
    """

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        whisper_model_name: str = "openai/whisper-base",
        chunk_duration_sec: float = 30.0,
        frame_duration_ms: float = 20.0,  # Whisper encoder frame size
        streaming: bool = True,
        max_samples: Optional[int] = None,
        dataset_length: Optional[int] = None  # Manually specify dataset length for streaming
    ):
        """
        Args:
            dataset_path: Path to HuggingFace dataset
            split: Dataset split to use
            whisper_model_name: Whisper model for feature extractor
            chunk_duration_sec: Duration of audio chunks in seconds
            frame_duration_ms: Frame duration in milliseconds (20ms for Whisper)
            streaming: Whether to stream the dataset
            max_samples: Maximum number of samples to use (for debugging)
            dataset_length: Known dataset length (for progress bars when streaming)
        """
        self.dataset = load_dataset(dataset_path, split=split, streaming=streaming)
        if max_samples and not streaming:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)
        self.chunk_duration_sec = chunk_duration_sec
        self.frame_duration_ms = frame_duration_ms
        self.encoder_output_frames = 1500  # For 30s audio with Whisper encoder
        self.streaming = streaming
        self.max_samples = max_samples
        self.dataset_length = dataset_length

    def create_frame_level_targets(
        self,
        segments: List[Dict],
        total_duration_ms: float
    ) -> np.ndarray:
        """
        Convert segment timestamps to frame-level binary targets.

        Args:
            segments: List of {"start_ms", "end_ms", "text"} dictionaries
            total_duration_ms: Total duration in milliseconds

        Returns:
            Binary array of shape (1500,) with 1s for speech frames
        """
        # Initialize target array to all zeros (silence)
        target_array = np.zeros(self.encoder_output_frames, dtype=np.float32)

        # Fill in speech segments
        for segment in segments:
            start_ms = segment["start_ms"]
            end_ms = segment["end_ms"]

            # Convert milliseconds to frame indices
            start_frame = int(start_ms / self.frame_duration_ms)
            end_frame = int(end_ms / self.frame_duration_ms)

            # Clamp to array bounds
            start_frame = max(0, start_frame)
            end_frame = min(self.encoder_output_frames, end_frame)

            # Set speech frames to 1
            if start_frame < end_frame:
                target_array[start_frame:end_frame] = 1.0

        return target_array

    def create_span_targets(
        self,
        segments: List[Dict],
        max_spans: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create span-level targets for DETR-style model.

        Args:
            segments: List of {"start_ms", "end_ms", "text"} dictionaries
            max_spans: Maximum number of spans to predict

        Returns:
            Tuple of (labels, spans) where:
            - labels: Binary array (max_spans,) with 1 for valid spans
            - spans: Array (max_spans, 2) with [start_time, end_time] normalized to [0, 1]
        """
        labels = np.zeros(max_spans, dtype=np.float32)
        spans = np.zeros((max_spans, 2), dtype=np.float32)

        num_segments = min(len(segments), max_spans)

        for i in range(num_segments):
            # Mark this as a valid span
            labels[i] = 1.0

            # Normalize times to [0, 1]
            start_norm = segments[i]["start_ms"] / (self.chunk_duration_sec * 1000)
            end_norm = segments[i]["end_ms"] / (self.chunk_duration_sec * 1000)

            # Clamp to [0, 1]
            start_norm = np.clip(start_norm, 0, 1)
            end_norm = np.clip(end_norm, 0, 1)

            spans[i] = [start_norm, end_norm]

        return labels, spans

    def __iter__(self):
        """Iterate through the dataset."""
        count = 0
        for sample in self.dataset:
            if self.max_samples and count >= self.max_samples:
                break

            try:
                # Extract audio array
                audio_array = sample["audio"]["array"]
                sample_rate = sample["audio"]["sampling_rate"]

                # Resample if necessary (Whisper expects 16kHz)
                if sample_rate != 16000:
                    import librosa
                    audio_array = librosa.resample(
                        audio_array,
                        orig_sr=sample_rate,
                        target_sr=16000
                    )

                # Pad or truncate to exact duration
                target_length = int(self.chunk_duration_sec * 16000)
                if len(audio_array) < target_length:
                    # Pad with zeros
                    audio_array = np.pad(audio_array, (0, target_length - len(audio_array)))
                else:
                    # Truncate
                    audio_array = audio_array[:target_length]

                # Extract Whisper features
                inputs = self.feature_extractor(
                    audio_array,
                    sampling_rate=16000,
                    return_tensors="pt"
                )

                # Get segments
                segments = sample.get("segments", [])

                # Create frame-level targets
                frame_targets = self.create_frame_level_targets(
                    segments,
                    self.chunk_duration_sec * 1000
                )

                # Create span-level targets for DETR model
                span_labels, span_coords = self.create_span_targets(segments)

                yield {
                    "input_features": inputs.input_features.squeeze(0),  # (80, 3000)
                    "frame_targets": torch.tensor(frame_targets),  # (1500,)
                    "span_labels": torch.tensor(span_labels),  # (max_spans,)
                    "span_coords": torch.tensor(span_coords),  # (max_spans, 2)
                    "num_segments": len(segments),
                    "audio_uuid": sample.get("audio_uuid", ""),
                }

                count += 1

            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

    def __len__(self):
        """Return the dataset length if known."""
        if self.dataset_length:
            # Use manually specified length
            if self.max_samples:
                return min(self.max_samples, self.dataset_length)
            return self.dataset_length
        elif not self.streaming:
            # Non-streaming mode, length is known
            if self.max_samples:
                return min(self.max_samples, len(self.dataset))
            return len(self.dataset)
        elif self.max_samples:
            # Streaming mode with max_samples specified
            return self.max_samples
        else:
            # Streaming mode without known length - return a large number
            # PyTorch Lightning will handle this gracefully
            raise TypeError("Dataset length unknown for streaming dataset without max_samples or dataset_length")


def create_dataloaders(
    dataset_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    whisper_model_name: str = "openai/whisper-base",
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    streaming: bool = False,  # Changed default to False for proper epoch tracking
    train_dataset_length: Optional[int] = None,  # For manual dataset length
    val_dataset_length: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        dataset_path: Path to the dataset
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        whisper_model_name: Name of the Whisper model
        max_train_samples: Maximum training samples (for debugging)
        max_val_samples: Maximum validation samples (for debugging)
        streaming: Whether to use streaming mode
        train_dataset_length: Known length of training dataset (for streaming mode)
        val_dataset_length: Known length of validation dataset (for streaming mode)
    """
    train_dataset = WhisperVADDataset(
        dataset_path=dataset_path,
        split="train",
        whisper_model_name=whisper_model_name,
        streaming=streaming,
        max_samples=max_train_samples,
        dataset_length=train_dataset_length
    )

    val_dataset = WhisperVADDataset(
        dataset_path=dataset_path,
        split="validation",
        whisper_model_name=whisper_model_name,
        streaming=streaming,
        max_samples=max_val_samples,
        dataset_length=val_dataset_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader