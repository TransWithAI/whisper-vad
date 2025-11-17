"""
Evaluation metrics for VAD models.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")


class VADMetrics:
    """
    Comprehensive metrics for VAD evaluation.

    Includes both frame-level and segment-level metrics.
    """

    def __init__(
        self,
        frame_duration_ms: float = 20.0,
        tolerance_ms: float = 50.0,
        min_speech_duration_ms: float = 200.0,
        min_silence_duration_ms: float = 300.0
    ):
        """
        Args:
            frame_duration_ms: Duration of each frame in milliseconds
            tolerance_ms: Tolerance for segment boundary matching
            min_speech_duration_ms: Minimum duration for speech segments
            min_silence_duration_ms: Minimum duration for silence segments
        """
        self.frame_duration_ms = frame_duration_ms
        self.tolerance_ms = tolerance_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms

    def calculate_frame_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate frame-level metrics.

        Args:
            predictions: Predicted probabilities (N,) or (batch, N)
            targets: Ground truth binary labels (N,) or (batch, N)
            threshold: Decision threshold

        Returns:
            Dictionary of metrics
        """
        # Flatten if batched
        if predictions.ndim > 1:
            predictions = predictions.flatten()
            targets = targets.flatten()

        # Binarize predictions
        pred_binary = (predictions > threshold).astype(int)

        # Calculate metrics
        metrics = {}

        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, pred_binary, average='binary', zero_division=0
        )
        metrics['frame_precision'] = float(precision)
        metrics['frame_recall'] = float(recall)
        metrics['frame_f1'] = float(f1)

        # Accuracy

        # AUC and AP
        try:
            metrics['frame_auc_roc'] = float(roc_auc_score(targets, predictions))
            metrics['frame_avg_precision'] = float(average_precision_score(targets, predictions))
        except:
            metrics['frame_auc_roc'] = 0.0
            metrics['frame_avg_precision'] = 0.0

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(targets, pred_binary, labels=[0, 1]).ravel()

        # False Positive Rate (FAR - False Alarm Rate)
        metrics['frame_fpr'] = float(fp / (fp + tn + 1e-6))

        # False Negative Rate (MDR - Miss Detection Rate)
        metrics['frame_fnr'] = float(fn / (fn + tp + 1e-6))

        # Detection Cost Function (DCF)
        # Common in VAD evaluation with configurable costs
        c_miss = 1.0  # Cost of missing speech
        c_fa = 0.5    # Cost of false alarm
        p_target = 0.5  # Prior probability of speech

        dcf = c_miss * p_target * metrics['frame_fnr'] + \
              c_fa * (1 - p_target) * metrics['frame_fpr']
        metrics['frame_dcf'] = float(dcf)

        return metrics

    def frames_to_segments(
        self,
        frames: np.ndarray,
        threshold: float = 0.5
    ) -> List[Tuple[float, float]]:
        """
        Convert frame predictions to segment list.

        Args:
            frames: Frame predictions (N,)
            threshold: Decision threshold

        Returns:
            List of (start_ms, end_ms) tuples
        """
        # Binarize
        binary = (frames > threshold).astype(int)

        # Apply minimum duration filtering
        binary = self.apply_duration_filtering(binary)

        # Find segment boundaries
        segments = []
        diff = np.diff(np.concatenate(([0], binary, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for start, end in zip(starts, ends):
            start_ms = start * self.frame_duration_ms
            end_ms = end * self.frame_duration_ms
            segments.append((start_ms, end_ms))

        return segments

    def apply_duration_filtering(self, binary: np.ndarray) -> np.ndarray:
        """
        Apply minimum duration constraints to binary predictions.

        Args:
            binary: Binary frame predictions

        Returns:
            Filtered binary predictions
        """
        filtered = binary.copy()

        # Minimum speech duration
        min_speech_frames = int(self.min_speech_duration_ms / self.frame_duration_ms)

        # Find speech segments
        diff = np.diff(np.concatenate(([0], binary, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for start, end in zip(starts, ends):
            if end - start < min_speech_frames:
                filtered[start:end] = 0

        # Minimum silence duration
        min_silence_frames = int(self.min_silence_duration_ms / self.frame_duration_ms)

        # Find silence segments
        inverted = 1 - filtered
        diff = np.diff(np.concatenate(([0], inverted, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for start, end in zip(starts, ends):
            if end - start < min_silence_frames:
                filtered[start:end] = 1

        return filtered

    def calculate_segment_metrics(
        self,
        pred_segments: List[Tuple[float, float]],
        true_segments: List[Tuple[float, float]],
        total_duration_ms: float = 30000.0
    ) -> Dict[str, float]:
        """
        Calculate segment-level metrics.

        Args:
            pred_segments: Predicted segments [(start_ms, end_ms), ...]
            true_segments: Ground truth segments [(start_ms, end_ms), ...]
            total_duration_ms: Total duration in milliseconds

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Segment-level precision, recall, F1
        tp, fp, fn = self.match_segments(pred_segments, true_segments)

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        metrics['segment_precision'] = float(precision)
        metrics['segment_recall'] = float(recall)
        metrics['segment_f1'] = float(f1)

        # Boundary precision (how accurate are the boundaries)
        boundary_errors = self.calculate_boundary_errors(pred_segments, true_segments)
        if boundary_errors:
            metrics['boundary_mae_ms'] = float(np.mean(boundary_errors))
            metrics['boundary_rmse_ms'] = float(np.sqrt(np.mean(np.square(boundary_errors))))
        else:
            metrics['boundary_mae_ms'] = 0.0
            metrics['boundary_rmse_ms'] = 0.0

        # Coverage metrics
        pred_coverage = self.calculate_coverage(pred_segments, total_duration_ms)
        true_coverage = self.calculate_coverage(true_segments, total_duration_ms)

        metrics['predicted_speech_ratio'] = float(pred_coverage)
        metrics['true_speech_ratio'] = float(true_coverage)
        metrics['speech_ratio_error'] = float(abs(pred_coverage - true_coverage))

        # Diarization Error Rate (DER) components
        # Note: Full DER requires speaker labels, here we calculate VAD-only components
        missed_speech_ms = self.calculate_missed_speech(pred_segments, true_segments)
        false_alarm_ms = self.calculate_false_alarm(pred_segments, true_segments)

        metrics['missed_speech_rate'] = float(missed_speech_ms / total_duration_ms)
        metrics['false_alarm_rate'] = float(false_alarm_ms / total_duration_ms)

        return metrics

    def match_segments(
        self,
        pred_segments: List[Tuple[float, float]],
        true_segments: List[Tuple[float, float]]
    ) -> Tuple[int, int, int]:
        """
        Match predicted and true segments with tolerance.

        Returns:
            (true_positives, false_positives, false_negatives)
        """
        matched_preds = set()
        matched_trues = set()

        # Try to match each predicted segment
        for i, (p_start, p_end) in enumerate(pred_segments):
            best_iou = 0
            best_j = -1

            for j, (t_start, t_end) in enumerate(true_segments):
                if j in matched_trues:
                    continue

                # Calculate IoU
                inter_start = max(p_start, t_start)
                inter_end = min(p_end, t_end)
                inter = max(0, inter_end - inter_start)

                union = (p_end - p_start) + (t_end - t_start) - inter
                iou = inter / (union + 1e-6)

                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            # Consider matched if IoU > 0.5 or boundaries within tolerance
            if best_j >= 0:
                t_start, t_end = true_segments[best_j]
                start_error = abs(p_start - t_start)
                end_error = abs(p_end - t_end)

                if best_iou > 0.5 or (start_error < self.tolerance_ms and
                                      end_error < self.tolerance_ms):
                    matched_preds.add(i)
                    matched_trues.add(best_j)

        tp = len(matched_preds)
        fp = len(pred_segments) - len(matched_preds)
        fn = len(true_segments) - len(matched_trues)

        return tp, fp, fn

    def calculate_boundary_errors(
        self,
        pred_segments: List[Tuple[float, float]],
        true_segments: List[Tuple[float, float]]
    ) -> List[float]:
        """
        Calculate boundary timing errors for matched segments.
        """
        errors = []

        for p_start, p_end in pred_segments:
            best_error = float('inf')

            for t_start, t_end in true_segments:
                # Check if segments overlap
                if min(p_end, t_end) > max(p_start, t_start):
                    start_error = abs(p_start - t_start)
                    end_error = abs(p_end - t_end)
                    avg_error = (start_error + end_error) / 2

                    if avg_error < best_error:
                        best_error = avg_error

            if best_error < float('inf'):
                errors.append(best_error)

        return errors

    def calculate_coverage(
        self,
        segments: List[Tuple[float, float]],
        total_duration_ms: float
    ) -> float:
        """
        Calculate the ratio of time covered by segments.
        """
        if not segments:
            return 0.0

        # Merge overlapping segments
        merged = self.merge_segments(segments)

        # Calculate total coverage
        coverage_ms = sum(end - start for start, end in merged)

        return coverage_ms / total_duration_ms

    def merge_segments(
        self,
        segments: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Merge overlapping segments.
        """
        if not segments:
            return []

        # Sort by start time
        sorted_segments = sorted(segments)
        merged = [sorted_segments[0]]

        for start, end in sorted_segments[1:]:
            last_start, last_end = merged[-1]

            if start <= last_end:
                # Overlapping, merge
                merged[-1] = (last_start, max(last_end, end))
            else:
                # Non-overlapping, add new
                merged.append((start, end))

        return merged

    def calculate_missed_speech(
        self,
        pred_segments: List[Tuple[float, float]],
        true_segments: List[Tuple[float, float]]
    ) -> float:
        """
        Calculate total duration of missed speech.
        """
        missed = 0.0

        for t_start, t_end in true_segments:
            covered = 0.0

            for p_start, p_end in pred_segments:
                # Calculate intersection
                inter_start = max(t_start, p_start)
                inter_end = min(t_end, p_end)

                if inter_end > inter_start:
                    covered += inter_end - inter_start

            missed += (t_end - t_start) - covered

        return max(0.0, missed)

    def calculate_false_alarm(
        self,
        pred_segments: List[Tuple[float, float]],
        true_segments: List[Tuple[float, float]]
    ) -> float:
        """
        Calculate total duration of false alarms.
        """
        false_alarm = 0.0

        for p_start, p_end in pred_segments:
            covered = 0.0

            for t_start, t_end in true_segments:
                # Calculate intersection
                inter_start = max(p_start, t_start)
                inter_end = min(p_end, t_end)

                if inter_end > inter_start:
                    covered += inter_end - inter_start

            false_alarm += (p_end - p_start) - covered

        return max(0.0, false_alarm)

    def calculate_all_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate all metrics (frame-level and segment-level).

        Args:
            predictions: Frame predictions
            targets: Frame targets
            threshold: Decision threshold

        Returns:
            Dictionary containing all metrics
        """
        # Frame-level metrics
        metrics = self.calculate_frame_metrics(predictions, targets, threshold)

        # Convert to segments
        pred_segments = self.frames_to_segments(predictions, threshold)
        true_segments = self.frames_to_segments(targets, threshold=0.5)

        # Segment-level metrics
        total_duration_ms = len(predictions) * self.frame_duration_ms
        segment_metrics = self.calculate_segment_metrics(
            pred_segments, true_segments, total_duration_ms
        )

        metrics.update(segment_metrics)

        return metrics


def _segment_bounds(segment: Dict) -> tuple:
    """Extract (start, end) seconds from various segment formats."""
    if segment is None:
        return 0.0, 0.0

    if isinstance(segment, dict):
        if "start" in segment and "end" in segment:
            return float(segment["start"]), float(segment["end"])
        if "start_ms" in segment and "end_ms" in segment:
            return float(segment["start_ms"]) / 1000.0, float(segment["end_ms"]) / 1000.0

    if isinstance(segment, (list, tuple)) and len(segment) >= 2:
        return float(segment[0]), float(segment[1])

    raise ValueError(f"Unsupported segment format: {segment}")


def compute_segment_f1(
    pred_segments: List[Dict],
    gt_segments: List[Dict],
    tolerance: float = 0.05,
    min_overlap_ratio: float = 0.5
) -> Dict[str, float]:
    """Calculate segment-level precision/recall/F1 with relaxed matching."""
    matched_preds = set()
    matched_gts = set()

    def iou(a_start, a_end, b_start, b_end):
        inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
        union = (a_end - a_start) + (b_end - b_start) - inter
        if union <= 0:
            return 0.0
        return inter / union

    for i, pred in enumerate(pred_segments):
        p_start, p_end = _segment_bounds(pred)
        best_match = None
        best_score = 0.0

        for j, gt in enumerate(gt_segments):
            if j in matched_gts:
                continue

            g_start, g_end = _segment_bounds(gt)
            overlap = iou(p_start, p_end, g_start, g_end)
            start_err = abs(p_start - g_start)
            end_err = abs(p_end - g_end)

            if overlap >= min_overlap_ratio or (start_err <= tolerance and end_err <= tolerance):
                if overlap > best_score:
                    best_score = overlap
                    best_match = j

        if best_match is not None:
            matched_preds.add(i)
            matched_gts.add(best_match)

    tp = len(matched_preds)
    fp = max(0, len(pred_segments) - tp)
    fn = max(0, len(gt_segments) - len(matched_gts))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _segments_to_frame_mask(segments: List[Dict], total_frames: int, frame_length: float) -> np.ndarray:
    mask = np.zeros(total_frames, dtype=np.float32)
    for seg in segments:
        start, end = _segment_bounds(seg)
        start_idx = int(np.floor(start / frame_length))
        end_idx = int(np.ceil(end / frame_length))
        start_idx = max(0, min(total_frames, start_idx))
        end_idx = max(0, min(total_frames, end_idx))
        if end_idx > start_idx:
            mask[start_idx:end_idx] = 1.0
    return mask


def compute_frame_f1(
    pred_segments: List[Dict],
    gt_segments: List[Dict],
    duration: float,
    frame_length: float = 0.01
) -> Dict[str, float]:
    """Compute frame-level precision/recall/F1 by rasterizing segments."""
    total_frames = max(1, int(np.ceil(duration / frame_length)))
    pred_mask = _segments_to_frame_mask(pred_segments, total_frames, frame_length)
    gt_mask = _segments_to_frame_mask(gt_segments, total_frames, frame_length)

    tp = float(np.sum((pred_mask == 1) & (gt_mask == 1)))
    fp = float(np.sum((pred_mask == 1) & (gt_mask == 0)))
    fn = float(np.sum((pred_mask == 0) & (gt_mask == 1)))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_detection_error_rate(
    pred_segments: List[Dict],
    gt_segments: List[Dict],
    duration: float,
    frame_length: float = 0.01
) -> float:
    """Compute miss + false-alarm duration normalized by total duration."""
    total_frames = max(1, int(np.ceil(duration / frame_length)))
    pred_mask = _segments_to_frame_mask(pred_segments, total_frames, frame_length)
    gt_mask = _segments_to_frame_mask(gt_segments, total_frames, frame_length)

    missed = float(np.sum(np.clip(gt_mask - pred_mask, 0, 1))) * frame_length
    false_alarm = float(np.sum(np.clip(pred_mask - gt_mask, 0, 1))) * frame_length
    total = max(duration, 1e-8)

    return (missed + false_alarm) / total