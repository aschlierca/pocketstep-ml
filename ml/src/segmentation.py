"""Step windows and automatic step detection.

Supervised training uses the *annotated* footfall times as step boundaries. Peak detection is a
separate tool for exploration and inference; it never creates ground truth, so its errors can be
measured on their own with `match_steps`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from . import schema
from .config import SegmentationConfig

# Window statuses. Anything other than OK means no features are computed for the step.
OK = "ok"
OUT_OF_RANGE = "out_of_range"
OVERLAPS_GAP = "overlaps_gap"
INSUFFICIENT_SAMPLES = "insufficient_samples"


@dataclass
class StepWindow:
    center: float
    start: float
    end: float
    status: str
    data: pd.DataFrame

    @property
    def ok(self) -> bool:
        return self.status == OK


def extract_window(processed: pd.DataFrame, center: float, median_interval: float,
                   gaps: list[tuple[float, float]], config: SegmentationConfig) -> StepWindow:
    """Samples with time in [center - before, center + after], selected by time rather than by a fixed
    sample count so irregular sampling cannot shift the window."""
    start = center - config.window_before_seconds
    end = center + config.window_after_seconds
    t = processed[schema.TIME].to_numpy()
    empty = processed.iloc[0:0]
    if len(t) == 0 or start < t[0] or end > t[-1]:
        return StepWindow(center, start, end, OUT_OF_RANGE, empty)
    if any(g_start < end and g_end > start for g_start, g_end in gaps):
        return StepWindow(center, start, end, OVERLAPS_GAP, empty)
    lo, hi = np.searchsorted(t, start, side="left"), np.searchsorted(t, end, side="right")
    data = processed.iloc[lo:hi]
    expected = (end - start) / median_interval
    if len(data) < config.min_sample_coverage * expected:
        return StepWindow(center, start, end, INSUFFICIENT_SAMPLES, data)
    return StepWindow(center, start, end, OK, data)


def label_exclusion_reason(foot_label: str, is_valid: bool, reviewed: bool, include_unreviewed: bool) -> str:
    """Why a label cannot be used as a binary training target ('' when it can)."""
    if not is_valid:
        return "invalid_label"
    if foot_label == "unknown":
        return "unknown_label"
    if not reviewed and not include_unreviewed:
        return "unreviewed_label"
    return ""


def overlapping_neighbors(times: np.ndarray, config: SegmentationConfig) -> np.ndarray:
    """True where a step's window overlaps the window of the previous or next step. Overlap is expected
    at normal cadence; it is reported so analyses can account for shared samples."""
    span = config.window_before_seconds + config.window_after_seconds
    gaps = np.diff(times)
    close = gaps < span
    result = np.zeros(len(times), dtype=bool)
    result[:-1] |= close
    result[1:] |= close
    return result


# ---------------------------------------------------------------------------
# Automatic step detection
# ---------------------------------------------------------------------------

def peak_signal(processed: pd.DataFrame, median_interval: float, config: SegmentationConfig) -> np.ndarray:
    """Gravity-adjusted magnitude, lightly smoothed, used for peak detection."""
    width = max(1, int(round(config.peak_smoothing_seconds / median_interval)))
    return uniform_filter1d(processed["dynamic_magnitude"].to_numpy(dtype=float), size=width, mode="nearest")


def detect_candidate_steps(processed: pd.DataFrame, median_interval: float, config: SegmentationConfig) -> np.ndarray:
    """Candidate footfall times from scipy.signal.find_peaks, run separately in each gap-free segment."""
    signal = peak_signal(processed, median_interval, config)
    t = processed[schema.TIME].to_numpy()
    distance = max(1, int(round(config.peak_min_distance_seconds / median_interval)))
    times = []
    for segment_id in np.unique(processed["segment"]):
        idx = np.flatnonzero(processed["segment"].to_numpy() == segment_id)
        if len(idx) < 3:
            continue
        peaks, _ = find_peaks(signal[idx], height=config.peak_min_height_g,
                              prominence=config.peak_min_prominence_g, distance=distance)
        times.extend(t[idx[peaks]])
    return np.array(sorted(times), dtype=float)


def match_steps(detected: np.ndarray, annotated: np.ndarray, tolerance: float) -> dict:
    """One-to-one matching of detected to annotated steps within `tolerance` seconds (closest pairs first).

    Returns detection precision/recall/F1, timing error statistics and the matched index pairs.
    """
    detected = np.asarray(detected, dtype=float)
    annotated = np.asarray(annotated, dtype=float)
    pairs = []
    for i, a in enumerate(annotated):
        for j in np.flatnonzero(np.abs(detected - a) <= tolerance):
            pairs.append((abs(detected[j] - a), i, int(j)))
    pairs.sort()
    used_a, used_d, matches = set(), set(), []
    for _, i, j in pairs:
        if i not in used_a and j not in used_d:
            used_a.add(i)
            used_d.add(j)
            matches.append((i, j))
    tp = len(matches)
    fp = len(detected) - tp
    fn = len(annotated) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors = np.array([detected[j] - annotated[i] for i, j in matches])
    return {
        "tolerance_seconds": tolerance,
        "annotated_steps": int(len(annotated)),
        "detected_steps": int(len(detected)),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_abs_timing_error_seconds": float(np.mean(np.abs(errors))) if tp else None,
        "mean_timing_bias_seconds": float(np.mean(errors)) if tp else None,
        "matches": sorted(matches),
    }


def nearest_previous(times: np.ndarray, max_interval: float) -> np.ndarray:
    """Time since the previous entry in a sorted array (NaN for the first or when longer than max_interval)."""
    out = np.full(len(times), np.nan)
    if len(times) > 1:
        d = np.diff(times)
        out[1:] = np.where(d <= max_interval, d, np.nan)
    return out


def nearest_next(times: np.ndarray, max_interval: float) -> np.ndarray:
    out = np.full(len(times), np.nan)
    if len(times) > 1:
        d = np.diff(times)
        out[:-1] = np.where(d <= max_interval, d, np.nan)
    return out


def extract_windows(processed: pd.DataFrame, centers: np.ndarray, median_interval: float,
                    gaps: list[tuple[float, float]], config: SegmentationConfig) -> list[StepWindow]:
    return [extract_window(processed, float(c), median_interval, gaps, config) for c in centers]

