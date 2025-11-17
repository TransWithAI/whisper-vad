#!/usr/bin/env python
"""ONNX inference script for encoder_only_decoder VAD model - Silero-style implementation.

This implementation follows Silero VAD's architecture for cleaner, more efficient processing:
- Fixed-size chunk processing for consistent behavior
- State management for streaming capability
- Hysteresis-based speech detection (dual threshold)
- Simplified segment extraction with proper padding
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from transformers import WhisperFeatureExtractor


class WhisperVADOnnxWrapper:
    """ONNX wrapper for Whisper-based VAD model following Silero's architecture."""

    def __init__(
        self,
        model_path: str,
        metadata_path: Optional[str] = None,
        force_cpu: bool = False,
        num_threads: int = 1,
    ):
        """Initialize ONNX model wrapper.

        Args:
            model_path: Path to ONNX model file
            metadata_path: Path to metadata JSON file (optional)
            force_cpu: Force CPU execution even if GPU is available
            num_threads: Number of CPU threads for inference
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed. Install with:\n"
                "  pip install onnxruntime      # For CPU\n"
                "  pip install onnxruntime-gpu  # For GPU"
            )

        self.model_path = model_path

        # Load metadata
        if metadata_path is None:
            metadata_path = model_path.replace('.onnx', '_metadata.json')

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            warnings.warn("No metadata file found. Using default values.")
            self.metadata = {
                'whisper_model_name': 'openai/whisper-base',
                'frame_duration_ms': 20,
                'total_duration_ms': 30000,
            }

        # Initialize feature extractor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.metadata['whisper_model_name']
        )

        # Set up ONNX Runtime session
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = num_threads
        opts.intra_op_num_threads = num_threads

        providers = ['CPUExecutionProvider']
        if not force_cpu and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(model_path, providers=providers, sess_options=opts)

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

        # Model parameters
        self.sample_rate = 16000  # Whisper uses 16kHz
        self.frame_duration_ms = self.metadata.get('frame_duration_ms', 20)
        self.chunk_duration_ms = self.metadata.get('total_duration_ms', 30000)
        self.chunk_samples = int(self.chunk_duration_ms * self.sample_rate / 1000)
        self.frames_per_chunk = int(self.chunk_duration_ms / self.frame_duration_ms)

        # Initialize state
        self.reset_states()

        print(f"Model loaded: {model_path}")
        print(f"  Providers: {providers}")
        print(f"  Chunk duration: {self.chunk_duration_ms}ms")
        print(f"  Frame duration: {self.frame_duration_ms}ms")

    def reset_states(self):
        """Reset internal states for new audio stream."""
        self._context = None
        self._last_chunk = None

    def _validate_input(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Validate and preprocess input audio.

        Args:
            audio: Input audio array
            sr: Sample rate

        Returns:
            Preprocessed audio at 16kHz
        """
        if audio.ndim > 1:
            # Convert to mono if multi-channel
            audio = audio.mean(axis=0 if audio.shape[0] > audio.shape[1] else 1)

        # Resample if needed
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        return audio

    def __call__(self, audio_chunk: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Process a single audio chunk.

        Args:
            audio_chunk: Audio chunk to process
            sr: Sample rate

        Returns:
            Frame-level speech probabilities
        """
        # Validate input
        audio_chunk = self._validate_input(audio_chunk, sr)

        # Ensure chunk is correct size
        if len(audio_chunk) < self.chunk_samples:
            audio_chunk = np.pad(
                audio_chunk,
                (0, self.chunk_samples - len(audio_chunk)),
                mode='constant'
            )
        elif len(audio_chunk) > self.chunk_samples:
            audio_chunk = audio_chunk[:self.chunk_samples]

        # Extract features
        inputs = self.feature_extractor(
            audio_chunk,
            sampling_rate=self.sample_rate,
            return_tensors="np"
        )

        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: inputs.input_features}
        )

        # Apply sigmoid to get probabilities
        frame_logits = outputs[0][0]  # Remove batch dimension
        frame_probs = 1 / (1 + np.exp(-frame_logits))

        return frame_probs

    def audio_forward(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Process full audio file in chunks (Silero-style).

        Args:
            audio: Full audio array
            sr: Sample rate

        Returns:
            Concatenated frame probabilities for entire audio
        """
        audio = self._validate_input(audio, sr)
        self.reset_states()

        all_probs = []

        # Process in chunks
        for i in range(0, len(audio), self.chunk_samples):
            chunk = audio[i:i + self.chunk_samples]

            # Pad last chunk if needed
            if len(chunk) < self.chunk_samples:
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)), mode='constant')

            # Get predictions for chunk
            chunk_probs = self.__call__(chunk, self.sample_rate)
            all_probs.append(chunk_probs)

        # Concatenate all probabilities
        if all_probs:
            return np.concatenate(all_probs)
        return np.array([])


def get_speech_timestamps(
    audio: np.ndarray,
    model,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float('inf'),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    return_seconds: bool = False,
    neg_threshold: Optional[float] = None,
    progress_tracking_callback: Optional[Callable[[float], None]] = None,
) -> List[Dict[str, float]]:
    """Extract speech timestamps from audio using Silero-style processing.

    This function implements Silero VAD's approach with:
    - Dual threshold (positive and negative) for hysteresis
    - Proper segment padding
    - Minimum duration filtering
    - Maximum duration handling with intelligent splitting

    Args:
        audio: Input audio array
        model: VAD model (WhisperVADOnnxWrapper instance)
        threshold: Speech threshold (default: 0.5)
        sampling_rate: Audio sample rate
        min_speech_duration_ms: Minimum speech segment duration
        max_speech_duration_s: Maximum speech segment duration
        min_silence_duration_ms: Minimum silence to split segments
        speech_pad_ms: Padding to add to speech segments
        return_seconds: Return times in seconds vs samples
        neg_threshold: Negative threshold for hysteresis (default: threshold - 0.15)
        progress_tracking_callback: Progress callback function

    Returns:
        List of speech segments with start/end times
    """
    # Convert to numpy if torch tensor
    if torch.is_tensor(audio):
        audio = audio.numpy()

    # Validate audio
    if audio.ndim > 1:
        audio = audio.mean(axis=0 if audio.shape[0] > audio.shape[1] else 1)

    # Get frame probabilities for entire audio
    model.reset_states()
    speech_probs = model.audio_forward(audio, sampling_rate)

    # Calculate frame parameters
    frame_duration_ms = model.frame_duration_ms
    frame_samples = int(sampling_rate * frame_duration_ms / 1000)

    # Convert durations to frames
    min_speech_frames = int(min_speech_duration_ms / frame_duration_ms)
    min_silence_frames = int(min_silence_duration_ms / frame_duration_ms)
    speech_pad_frames = int(speech_pad_ms / frame_duration_ms)
    max_speech_frames = int(max_speech_duration_s * 1000 / frame_duration_ms) if max_speech_duration_s != float('inf') else len(speech_probs)

    # Set negative threshold for hysteresis
    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)

    # Track speech segments
    triggered = False
    speeches = []
    current_speech = {}
    current_probs = []  # Track probabilities for current segment
    temp_end = 0

    # Process each frame
    for i, speech_prob in enumerate(speech_probs):
        # Report progress
        if progress_tracking_callback:
            progress = (i + 1) / len(speech_probs) * 100
            progress_tracking_callback(progress)

        # Track probabilities for current segment
        if triggered:
            current_probs.append(float(speech_prob))

        # Speech onset detection
        if speech_prob >= threshold and not triggered:
            triggered = True
            current_speech['start'] = i
            current_probs = [float(speech_prob)]  # Start tracking probabilities
            continue

        # Check for maximum speech duration
        if triggered and 'start' in current_speech:
            duration = i - current_speech['start']
            if duration > max_speech_frames:
                # Force end segment at max duration
                current_speech['end'] = current_speech['start'] + max_speech_frames
                # Calculate probability statistics for segment
                if current_probs:
                    current_speech['avg_prob'] = np.mean(current_probs)
                    current_speech['min_prob'] = np.min(current_probs)
                    current_speech['max_prob'] = np.max(current_probs)
                speeches.append(current_speech)
                current_speech = {}
                current_probs = []
                triggered = False
                temp_end = 0
                continue

        # Speech offset detection with hysteresis
        if speech_prob < neg_threshold and triggered:
            if not temp_end:
                temp_end = i

            # Check if silence is long enough
            if i - temp_end >= min_silence_frames:
                # End current speech segment
                current_speech['end'] = temp_end

                # Check minimum duration
                if current_speech['end'] - current_speech['start'] >= min_speech_frames:
                    # Calculate probability statistics for segment
                    if current_probs:
                        current_speech['avg_prob'] = np.mean(current_probs[:temp_end - current_speech['start']])
                        current_speech['min_prob'] = np.min(current_probs[:temp_end - current_speech['start']])
                        current_speech['max_prob'] = np.max(current_probs[:temp_end - current_speech['start']])
                    speeches.append(current_speech)

                current_speech = {}
                current_probs = []
                triggered = False
                temp_end = 0

        # Reset temp_end if speech resumes
        elif speech_prob >= threshold and temp_end:
            temp_end = 0

    # Handle speech that continues to the end
    if triggered and 'start' in current_speech:
        current_speech['end'] = len(speech_probs)
        if current_speech['end'] - current_speech['start'] >= min_speech_frames:
            # Calculate probability statistics for segment
            if current_probs:
                current_speech['avg_prob'] = np.mean(current_probs)
                current_speech['min_prob'] = np.min(current_probs)
                current_speech['max_prob'] = np.max(current_probs)
            speeches.append(current_speech)

    # Apply padding to segments
    for i, speech in enumerate(speeches):
        # Add padding
        if i == 0:
            speech['start'] = max(0, speech['start'] - speech_pad_frames)
        else:
            speech['start'] = max(speeches[i-1]['end'], speech['start'] - speech_pad_frames)

        if i < len(speeches) - 1:
            speech['end'] = min(speeches[i+1]['start'], speech['end'] + speech_pad_frames)
        else:
            speech['end'] = min(len(speech_probs), speech['end'] + speech_pad_frames)

    # Convert to time units
    if return_seconds:
        for speech in speeches:
            speech['start'] = speech['start'] * frame_duration_ms / 1000
            speech['end'] = speech['end'] * frame_duration_ms / 1000
    else:
        # Convert frames to samples
        for speech in speeches:
            speech['start'] = speech['start'] * frame_samples
            speech['end'] = speech['end'] * frame_samples

    return speeches


class VADIterator:
    """Stream iterator for real-time VAD processing (Silero-style)."""

    def __init__(
        self,
        model,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ):
        """Initialize VAD iterator for streaming.

        Args:
            model: WhisperVADOnnxWrapper instance
            threshold: Speech detection threshold
            sampling_rate: Audio sample rate
            min_silence_duration_ms: Minimum silence duration
            speech_pad_ms: Speech padding in milliseconds
        """
        self.model = model
        self.threshold = threshold
        self.neg_threshold = max(threshold - 0.15, 0.01)
        self.sampling_rate = sampling_rate

        # Calculate frame-based parameters
        self.frame_duration_ms = model.frame_duration_ms
        self.min_silence_frames = min_silence_duration_ms / self.frame_duration_ms
        self.speech_pad_frames = speech_pad_ms / self.frame_duration_ms

        self.reset_states()

    def reset_states(self):
        """Reset iterator state."""
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_frame = 0
        self.buffer = np.array([])
        self.speech_start = 0

    def __call__(self, audio_chunk: np.ndarray, return_seconds: bool = False) -> Optional[Dict]:
        """Process audio chunk and detect speech boundaries.

        Args:
            audio_chunk: Audio chunk to process
            return_seconds: Return times in seconds vs samples

        Returns:
            Dict with 'start' or 'end' key when speech boundary detected
        """
        # Add to buffer
        self.buffer = np.concatenate([self.buffer, audio_chunk]) if len(self.buffer) > 0 else audio_chunk

        # Check if we have enough samples for a full chunk
        if len(self.buffer) < self.model.chunk_samples:
            return None

        # Process full chunk
        chunk = self.buffer[:self.model.chunk_samples]
        self.buffer = self.buffer[self.model.chunk_samples:]

        # Get frame predictions
        frame_probs = self.model(chunk, self.sampling_rate)

        results = []

        # Process each frame
        for prob in frame_probs:
            self.current_frame += 1

            # Speech onset
            if prob >= self.threshold and not self.triggered:
                self.triggered = True
                self.speech_start = self.current_frame - self.speech_pad_frames
                start_time = max(0, self.speech_start * self.frame_duration_ms / 1000) if return_seconds else \
                            max(0, self.speech_start * self.frame_duration_ms * 16)
                return {'start': start_time}

            # Speech offset
            if prob < self.neg_threshold and self.triggered:
                if not self.temp_end:
                    self.temp_end = self.current_frame
                elif self.current_frame - self.temp_end >= self.min_silence_frames:
                    # End speech
                    end_frame = self.temp_end + self.speech_pad_frames
                    end_time = end_frame * self.frame_duration_ms / 1000 if return_seconds else \
                              end_frame * self.frame_duration_ms * 16
                    self.triggered = False
                    self.temp_end = 0
                    return {'end': end_time}
            elif prob >= self.threshold and self.temp_end:
                self.temp_end = 0

        return None


def load_audio(audio_path: str, sampling_rate: int = 16000) -> np.ndarray:
    """Load audio file and convert to target sample rate.

    Args:
        audio_path: Path to audio file
        sampling_rate: Target sample rate

    Returns:
        Audio array at target sample rate
    """
    audio, sr = librosa.load(audio_path, sr=sampling_rate)
    return audio


def save_segments(segments: List[Dict], output_path: str, format: str = 'json'):
    """Save speech segments to file.

    Args:
        segments: List of speech segments
        output_path: Output file path
        format: Output format (json, txt, csv, srt)
    """
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump({'segments': segments}, f, indent=2)

    elif format == 'txt':
        with open(output_path, 'w') as f:
            for i, seg in enumerate(segments, 1):
                start = seg['start']
                end = seg['end']
                duration = end - start
                f.write(f"{i:3d}. {start:8.3f}s - {end:8.3f}s (duration: {duration:6.3f}s)\n")

    elif format == 'csv':
        import csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['start', 'end', 'duration'])
            writer.writeheader()
            for seg in segments:
                row = {
                    'start': seg['start'],
                    'end': seg['end'],
                    'duration': seg['end'] - seg['start']
                }
                writer.writerow(row)

    elif format == 'srt':
        with open(output_path, 'w') as f:
            for i, seg in enumerate(segments, 1):
                start_s = seg['start']
                end_s = seg['end']

                # Convert to SRT timestamp format
                def seconds_to_srt(seconds):
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    secs = int(seconds % 60)
                    millis = int((seconds % 1) * 1000)
                    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

                f.write(f"{i}\n")
                f.write(f"{seconds_to_srt(start_s)} --> {seconds_to_srt(end_s)}\n")

                # Write speech probability information if available
                if 'avg_prob' in seg:
                    f.write(f"Speech [Avg: {seg['avg_prob']:.2%}, Min: {seg['min_prob']:.2%}, Max: {seg['max_prob']:.2%}]\n\n")
                else:
                    f.write(f"[Speech]\n\n")


def main():
    parser = argparse.ArgumentParser(
        description='Silero-style ONNX inference for Whisper-based VAD model'
    )
    parser.add_argument('--model', required=True, help='Path to ONNX model file')
    parser.add_argument('--audio', required=True, help='Path to audio file')
    parser.add_argument('--output', help='Output file path (default: audio_path.vad.json)')
    parser.add_argument('--format', choices=['json', 'txt', 'csv', 'srt'],
                      default='json', help='Output format')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Speech detection threshold (0.0-1.0)')
    parser.add_argument('--neg-threshold', type=float, default=None,
                      help='Negative threshold for hysteresis (default: threshold - 0.15)')
    parser.add_argument('--min-speech-duration', type=int, default=250,
                      help='Minimum speech duration in ms')
    parser.add_argument('--min-silence-duration', type=int, default=100,
                      help='Minimum silence duration in ms')
    parser.add_argument('--speech-pad', type=int, default=30,
                      help='Speech padding in ms')
    parser.add_argument('--max-speech-duration', type=float, default=float('inf'),
                      help='Maximum speech duration in seconds')
    parser.add_argument('--metadata', help='Path to metadata JSON file')
    parser.add_argument('--force-cpu', action='store_true',
                      help='Force CPU execution even if GPU is available')
    parser.add_argument('--threads', type=int, default=1,
                      help='Number of CPU threads')
    parser.add_argument('--stream', action='store_true',
                      help='Use streaming mode (demonstrate VADIterator)')

    args = parser.parse_args()

    # Check files exist
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1

    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        return 1

    try:
        # Initialize model
        print("Loading model...")
        model = WhisperVADOnnxWrapper(
            model_path=args.model,
            metadata_path=args.metadata,
            force_cpu=args.force_cpu,
            num_threads=args.threads,
        )

        # Load audio
        print(f"Loading audio: {args.audio}")
        audio = load_audio(args.audio)
        duration = len(audio) / 16000
        print(f"Audio duration: {duration:.2f}s")

        if args.stream:
            # Demonstrate streaming mode
            print("\nUsing streaming mode (VADIterator)...")
            vad_iterator = VADIterator(
                model=model,
                threshold=args.threshold,
                min_silence_duration_ms=args.min_silence_duration,
                speech_pad_ms=args.speech_pad,
            )

            # Simulate streaming by processing in small chunks
            chunk_size = 16000  # 1 second chunks
            segments = []
            current_segment = {}

            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                result = vad_iterator(chunk, return_seconds=True)

                if result:
                    if 'start' in result:
                        current_segment = {'start': result['start'] + i/16000}
                        print(f"  Speech started: {current_segment['start']:.2f}s")
                    elif 'end' in result and current_segment:
                        current_segment['end'] = result['end'] + i/16000
                        segments.append(current_segment)
                        print(f"  Speech ended: {current_segment['end']:.2f}s")
                        current_segment = {}

            # Handle ongoing speech at end
            if current_segment and 'start' in current_segment:
                current_segment['end'] = duration
                segments.append(current_segment)
        else:
            # Use batch mode with Silero-style processing
            print("\nProcessing with Silero-style speech detection...")

            # Progress callback
            def progress_callback(percent):
                print(f"\rProgress: {percent:.1f}%", end='', flush=True)

            # Get speech timestamps
            segments = get_speech_timestamps(
                audio=audio,
                model=model,
                threshold=args.threshold,
                sampling_rate=16000,
                min_speech_duration_ms=args.min_speech_duration,
                min_silence_duration_ms=args.min_silence_duration,
                speech_pad_ms=args.speech_pad,
                max_speech_duration_s=args.max_speech_duration,
                return_seconds=True,
                neg_threshold=args.neg_threshold,
                progress_tracking_callback=progress_callback,
            )
            print()  # New line after progress

        # Display results
        print(f"\nFound {len(segments)} speech segments:")
        total_speech = sum(seg['end'] - seg['start'] for seg in segments)
        print(f"Total speech: {total_speech:.2f}s ({total_speech/duration*100:.1f}%)")

        if segments:
            print("\nSegments:")
            for i, seg in enumerate(segments[:10], 1):  # Show first 10
                duration_seg = seg['end'] - seg['start']
                print(f"  {i:2d}. {seg['start']:7.3f}s - {seg['end']:7.3f}s (duration: {duration_seg:5.3f}s)")
            if len(segments) > 10:
                print(f"  ... and {len(segments) - 10} more segments")

        # Save results
        output_path = args.output
        if not output_path:
            base = os.path.splitext(args.audio)[0]
            output_path = f"{base}.vad.{args.format}"

        save_segments(segments, output_path, format=args.format)
        print(f"\nResults saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())