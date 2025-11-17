"""
Inference script for trained VAD models.
"""

import argparse
import warnings
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from transformers import WhisperFeatureExtractor

warnings.filterwarnings("ignore")

from models import DETRVADModule, EncoderOnlyVADModule
from utils.metrics import VADMetrics


def seconds_to_srt(seconds: float) -> str:
    """Convert seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_prob_line(segment: Dict[str, float]) -> str:
    """Render probability statistics for human-readable outputs."""
    if all(key in segment for key in ("avg_prob", "min_prob", "max_prob")):
        return (
            "Speech "
            f"[Avg: {segment['avg_prob']:.2%}, "
            f"Min: {segment['min_prob']:.2%}, "
            f"Max: {segment['max_prob']:.2%}]"
        )
    if "confidence" in segment:
        return f"Speech [Confidence: {segment['confidence']:.2%}]"
    return "[Speech]"


class VADInference:
    """Inference wrapper for VAD models."""

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = 'encoder_only_linear',
        device: str = 'cuda',
        whisper_model: str = 'openai/whisper-base'
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            model_type: Type of model ('encoder_only_linear', 'encoder_only_decoder', 'detr')
            device: Device to run inference on
            whisper_model: Whisper model name for feature extraction
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type

        # Load model
        if model_type in ['encoder_only_linear', 'encoder_only_decoder']:
            self.model = EncoderOnlyVADModule.load_from_checkpoint(
                checkpoint_path,
                map_location=self.device
            )
        else:  # detr
            self.model = DETRVADModule.load_from_checkpoint(
                checkpoint_path,
                map_location=self.device
            )

        self.model.eval()
        self.model.to(self.device)

        # Initialize feature extractor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)

        # VAD metrics for post-processing
        self.vad_metrics = VADMetrics()

    def process_audio(
        self,
        audio_path: str,
        threshold: float = 0.5,
        chunk_duration: float = 30.0,
        overlap: float = 0.0,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        max_speech_duration_s: float = float('inf'),
        neg_threshold: Optional[float] = None,
        progress: bool = False
    ) -> List[Dict[str, float]]:
        """
        Process audio file and return speech segments (Silero-style).

        Args:
            audio_path: Path to audio file
            threshold: Decision threshold for VAD
            chunk_duration: Duration of chunks in seconds
            overlap: Overlap between chunks in seconds (currently unused)
            min_speech_duration_ms: Minimum speech segment length
            min_silence_duration_ms: Minimum silence to separate segments
            speech_pad_ms: Padding applied to each segment
            max_speech_duration_s: Maximum speech duration before forced split
            neg_threshold: Negative threshold for hysteresis
            progress: Whether to print progress percentage while processing

        Returns:
            List of speech segment dictionaries with probability stats
        """
        audio, sr = librosa.load(audio_path, sr=16000)
        sr = int(sr)

        if self.model_type == 'detr':
            return self.process_audio_detr(
                audio=audio,
                threshold=threshold,
                chunk_duration=chunk_duration
            )

        speech_probs = self.compute_frame_probabilities(
            audio=audio,
            sr=sr,
            chunk_duration=chunk_duration,
            show_progress=progress
        )

        segments = self.extract_segments_from_probs(
            speech_probs=speech_probs,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            max_speech_duration_s=max_speech_duration_s,
            neg_threshold=neg_threshold
        )

        return segments

    def compute_frame_probabilities(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        chunk_duration: float = 30.0,
        show_progress: bool = False
    ) -> np.ndarray:
        """Run the encoder-only model over the entire audio to get frame probs."""
        chunk_samples = int(chunk_duration * sr)
        frame_probs: List[np.ndarray] = []
        frame_duration_ms = getattr(self.vad_metrics, 'frame_duration_ms', 20.0)

        with torch.no_grad():
            for chunk_start in range(0, len(audio), chunk_samples):
                chunk = audio[chunk_start:chunk_start + chunk_samples]
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), constant_values=0)

                inputs = self.feature_extractor(
                    chunk,
                    sampling_rate=sr,
                    return_tensors="pt"
                )
                input_features = inputs.input_features.to(self.device)
                logits = self.model(input_features)
                probs = torch.sigmoid(logits).float().cpu().numpy()[0]
                frame_probs.append(probs)

                if show_progress:
                    progress = min((chunk_start + chunk_samples) / len(audio), 1.0) * 100
                    print(f"\rProcessing chunks: {progress:.1f}%", end='', flush=True)

        if show_progress:
            print()

        if frame_probs:
            return np.concatenate(frame_probs)
        return np.array([])

    def extract_segments_from_probs(
        self,
        speech_probs: np.ndarray,
        threshold: float,
        min_speech_duration_ms: int,
        min_silence_duration_ms: int,
        speech_pad_ms: int,
        max_speech_duration_s: float,
        neg_threshold: Optional[float]
    ) -> List[Dict[str, float]]:
        """Convert frame-level probabilities into speech segments."""
        if speech_probs.size == 0:
            return []

        frame_duration_ms = getattr(self.vad_metrics, 'frame_duration_ms', 20.0)
        frame_duration_s = frame_duration_ms / 1000.0
        min_speech_frames = int(min_speech_duration_ms / frame_duration_ms)
        min_silence_frames = int(min_silence_duration_ms / frame_duration_ms)
        speech_pad_frames = int(speech_pad_ms / frame_duration_ms)
        max_speech_frames = int(max_speech_duration_s * 1000 / frame_duration_ms) \
            if max_speech_duration_s != float('inf') else len(speech_probs)
        neg_threshold = neg_threshold if neg_threshold is not None else max(threshold - 0.15, 0.01)

        triggered = False
        temp_end: Optional[int] = None
        speeches: List[Dict[str, float]] = []
        current_speech: Dict[str, float] = {}
        current_probs: List[float] = []

        for frame_idx, prob in enumerate(speech_probs):
            if triggered:
                current_probs.append(float(prob))

            if prob >= threshold and not triggered:
                triggered = True
                current_speech = {'start': frame_idx}
                current_probs = [float(prob)]
                temp_end = None
                continue

            if triggered and 'start' in current_speech:
                duration_frames = frame_idx - current_speech['start']
                if duration_frames > max_speech_frames:
                    current_speech['end'] = current_speech['start'] + max_speech_frames
                    stats = current_probs[:max_speech_frames]
                    if stats:
                        current_speech['avg_prob'] = float(np.mean(stats))
                        current_speech['min_prob'] = float(np.min(stats))
                        current_speech['max_prob'] = float(np.max(stats))
                    speeches.append(current_speech)
                    current_speech = {}
                    current_probs = []
                    triggered = False
                    temp_end = None
                    continue

            if prob < neg_threshold and triggered:
                if temp_end is None:
                    temp_end = frame_idx
                elif frame_idx - temp_end >= min_silence_frames:
                    current_speech['end'] = temp_end
                    segment_length = current_speech['end'] - current_speech['start']
                    if segment_length >= min_speech_frames:
                        valid_count = segment_length
                        stats = current_probs[:valid_count]
                        if stats:
                            current_speech['avg_prob'] = float(np.mean(stats))
                            current_speech['min_prob'] = float(np.min(stats))
                            current_speech['max_prob'] = float(np.max(stats))
                        speeches.append(current_speech)

                    current_speech = {}
                    current_probs = []
                    triggered = False
                    temp_end = None

            elif prob >= threshold and temp_end is not None:
                temp_end = None

        if triggered and 'start' in current_speech:
            current_speech['end'] = len(speech_probs)
            segment_length = current_speech['end'] - current_speech['start']
            if segment_length >= min_speech_frames:
                stats = current_probs[:segment_length]
                if stats:
                    current_speech['avg_prob'] = float(np.mean(stats))
                    current_speech['min_prob'] = float(np.min(stats))
                    current_speech['max_prob'] = float(np.max(stats))
                speeches.append(current_speech)

        for idx, speech in enumerate(speeches):
            if idx == 0:
                speech['start'] = max(0, speech['start'] - speech_pad_frames)
            else:
                speech['start'] = max(speeches[idx - 1]['end'], speech['start'] - speech_pad_frames)

            if idx < len(speeches) - 1:
                speech['end'] = min(speeches[idx + 1]['start'], speech['end'] + speech_pad_frames)
            else:
                speech['end'] = min(len(speech_probs), speech['end'] + speech_pad_frames)

        formatted_segments: List[Dict[str, float]] = []
        for speech in speeches:
            start_sec = speech['start'] * frame_duration_s
            end_sec = speech['end'] * frame_duration_s
            segment_dict: Dict[str, float] = {
                'start': start_sec,
                'end': end_sec,
                'duration': end_sec - start_sec
            }
            for key in ('avg_prob', 'min_prob', 'max_prob'):
                if key in speech:
                    segment_dict[key] = speech[key]
            formatted_segments.append(segment_dict)

        return formatted_segments

    def process_audio_detr(
        self,
        audio: np.ndarray,
        threshold: float,
        chunk_duration: float
    ) -> List[Dict[str, float]]:
        """Fallback processing for DETR models using span predictions."""
        chunk_samples = int(chunk_duration * 16000)
        all_segments: List[Tuple[float, float, float]] = []

        with torch.no_grad():
            for start_idx in range(0, len(audio), chunk_samples):
                chunk = audio[start_idx:start_idx + chunk_samples]
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

                inputs = self.feature_extractor(
                    chunk,
                    sampling_rate=16000,
                    return_tensors="pt"
                )
                input_features = inputs.input_features.to(self.device)
                segments = self.predict_detr(input_features, chunk_duration, threshold)
                offset_sec = start_idx / 16000
                for seg in segments:
                    all_segments.append((
                        seg['start'] + offset_sec,
                        seg['end'] + offset_sec,
                        seg['confidence']
                    ))

        merged = self.merge_segments([(s[0], s[1]) for s in all_segments])
        formatted: List[Dict[str, float]] = []
        for start, end in merged:
            confs = [seg[2] for seg in all_segments if seg[0] <= end and seg[1] >= start]
            stats = {
                'avg_prob': float(np.mean(confs)) if confs else 0.0,
                'min_prob': float(np.min(confs)) if confs else 0.0,
                'max_prob': float(np.max(confs)) if confs else 0.0,
            }
            formatted.append({
                'start': start,
                'end': end,
                'duration': end - start,
                **stats
            })
        return formatted

    def predict_detr(
        self,
        input_features: torch.Tensor,
        duration: float,
        threshold: float = 0.5
    ) -> List[Dict[str, float]]:
        """
        Predict using DETR model.

        Args:
            input_features: Whisper features (batch, 80, 3000)
            duration: Chunk duration in seconds
            threshold: Confidence threshold

        Returns:
            List of speech segments
        """
        # Get predictions
        outputs = self.model(input_features)

        # Get speech predictions (class 1)
        pred_probs = torch.softmax(outputs['pred_logits'], dim=-1)[0]  # Remove batch
        pred_mask = pred_probs[:, 1] > threshold  # Class 1 is speech

        # Get valid spans
        # Convert to float32 first to handle BFloat16 tensors
        pred_spans = outputs['pred_spans'][0][pred_mask].float().cpu().numpy()

        # Convert normalized spans to seconds
        segments = []
        for span, conf in zip(pred_spans, pred_probs[pred_mask][:, 1].cpu().numpy()):
            start_sec = span[0] * duration
            end_sec = span[1] * duration
            segments.append({
                'start': float(start_sec),
                'end': float(end_sec),
                'confidence': float(conf)
            })

        return segments

    def merge_segments(
        self,
        segments: List[Tuple[float, float]],
        gap_threshold: float = 0.3
    ) -> List[Tuple[float, float]]:
        """
        Merge nearby segments.

        Args:
            segments: List of speech segment dictionaries
            gap_threshold: Maximum gap to merge (seconds)

        Returns:
            Merged segments
        """
        if not segments:
            return []

        # Sort by start time
        sorted_segments = sorted(segments)
        merged = [sorted_segments[0]]

        for start, end in sorted_segments[1:]:
            last_start, last_end = merged[-1]

            # Check if should merge
            if start - last_end <= gap_threshold:
                # Merge
                merged[-1] = (last_start, max(last_end, end))
            else:
                # Add new segment
                merged.append((start, end))

        return merged

    def save_segments(
        self,
        segments: List[Dict[str, float]],
        output_path: str,
        format: str = 'txt'
    ):
        """
        Save segments to file.

        Args:
            segments: List of speech segment dictionaries
            output_path: Output file path
            format: Output format ('txt', 'csv', 'json', 'srt')
        """
        if format == 'txt':
            with open(output_path, 'w') as f:
                for i, seg in enumerate(segments, 1):
                    start = seg['start']
                    end = seg['end']
                    duration = seg['duration']
                    f.write(
                        f"{i:3d}. {start:8.3f}s - {end:8.3f}s "
                        f"(duration: {duration:6.3f}s) {format_prob_line(seg)}\n"
                    )

        elif format == 'csv':
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['start', 'end', 'duration', 'avg_prob', 'min_prob', 'max_prob'])
                for seg in segments:
                    writer.writerow([
                        f"{seg['start']:.3f}",
                        f"{seg['end']:.3f}",
                        f"{seg['duration']:.3f}",
                        f"{seg.get('avg_prob', 0.0):.4f}",
                        f"{seg.get('min_prob', 0.0):.4f}",
                        f"{seg.get('max_prob', 0.0):.4f}",
                    ])

        elif format == 'json':
            import json
            data = [
                {
                    "start": float(f"{seg['start']:.3f}"),
                    "end": float(f"{seg['end']:.3f}"),
                    "duration": float(f"{seg['duration']:.3f}"),
                    "avg_prob": seg.get('avg_prob'),
                    "min_prob": seg.get('min_prob'),
                    "max_prob": seg.get('max_prob')
                }
                for seg in segments
            ]
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

        elif format == 'srt':
            with open(output_path, 'w') as f:
                for idx, seg in enumerate(segments, 1):
                    f.write(f"{idx}\n")
                    f.write(f"{seconds_to_srt(seg['start'])} --> {seconds_to_srt(seg['end'])}\n")
                    f.write(f"{format_prob_line(seg)}\n\n")


def main():
    parser = argparse.ArgumentParser(description="VAD Inference")

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--model_type',
        type=str,
        default='encoder_only_linear',
        choices=['encoder_only_linear', 'encoder_only_decoder', 'detr'],
        help='Model type'
    )

    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='Path to audio file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path for segments'
    )

    parser.add_argument(
        '--format',
        type=str,
        default='txt',
        choices=['txt', 'csv', 'json', 'srt'],
        help='Output format'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='VAD decision threshold'
    )

    parser.add_argument(
        '--chunk_duration',
        type=float,
        default=30.0,
        help='Chunk duration in seconds'
    )

    parser.add_argument(
        '--min_speech_duration',
        type=int,
        default=250,
        help='Minimum speech duration in milliseconds'
    )

    parser.add_argument(
        '--min_silence_duration',
        type=int,
        default=100,
        help='Minimum silence duration in milliseconds'
    )

    parser.add_argument(
        '--speech_pad',
        type=int,
        default=30,
        help='Padding to add before/after speech segments (ms)'
    )

    parser.add_argument(
        '--max_speech_duration',
        type=float,
        default=float('inf'),
        help='Maximum speech segment duration in seconds'
    )

    parser.add_argument(
        '--neg_threshold',
        type=float,
        default=None,
        help='Negative threshold for hysteresis (defaults to threshold - 0.15)'
    )

    parser.add_argument(
        '--show_progress',
        action='store_true',
        help='Show chunk processing progress bar'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on'
    )

    args = parser.parse_args()

    # Initialize inference
    vad = VADInference(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        device=args.device
    )

    # Process audio
    print(f"Processing {args.audio}...")
    segments = vad.process_audio(
        audio_path=args.audio,
        threshold=args.threshold,
        chunk_duration=args.chunk_duration,
        min_speech_duration_ms=args.min_speech_duration,
        min_silence_duration_ms=args.min_silence_duration,
        speech_pad_ms=args.speech_pad,
        max_speech_duration_s=args.max_speech_duration,
        neg_threshold=args.neg_threshold,
        progress=args.show_progress
    )

    # Print results
    print(f"\nFound {len(segments)} speech segments:")
    for i, seg in enumerate(segments, 1):
        start = seg['start']
        end = seg['end']
        duration = seg.get('duration', end - start)
        print(
            f"  {i}. {start:.3f}s - {end:.3f}s (duration: {duration:.3f}s) "
            f"{format_prob_line(seg)}"
        )

    # Save if requested
    if args.output:
        vad.save_segments(segments, args.output, args.format)
        print(f"\nSegments saved to {args.output}")


if __name__ == "__main__":
    main()