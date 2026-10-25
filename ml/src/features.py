"""Per-step feature extraction.

Each step becomes one row. Features are split into:

- live features: computable from the step's own window plus earlier steps. Because the window
  extends `window_after_seconds` past the footfall, a live prediction is available after that delay.
- future features: need the *next* step (time_until_next_step). They are for offline analysis only
  and are excluded from the default "live" feature set.

Usage:
    python -m ml.src.features --data-dir ml/data/raw --output ml/data/processed/step_features.csv
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences

from . import schema
from .config import FeatureConfig, PipelineConfig
from .preprocessing import Session, load_sessions, preprocess
from .segmentation import (OK, extract_windows, label_exclusion_reason, nearest_next, nearest_previous,
                           overlapping_neighbors)

AXIS_STATS = ("max", "min", "mean", "std")

WINDOW_FEATURES = (
    *[f"{axis}_{stat}" for axis in schema.AXES for stat in AXIS_STATS],
    "magnitude_mean",
    "magnitude_std",
    "magnitude_max",
    "magnitude_peak_to_peak",
    "peak_prominence",
    "peak_time_offset",
    "signal_energy",
    "n_peaks",
    "vertical_max",
    "vertical_min",
    "vertical_std",
    "horizontal_max",
    "horizontal_mean",
)
CONTEXT_LIVE_FEATURES = ("time_since_previous_step", "pocket_side_is_left")
FUTURE_FEATURES = ("time_until_next_step",)

FEATURE_SETS = {
    "live": (*WINDOW_FEATURES, *CONTEXT_LIVE_FEATURES),
    "offline": (*WINDOW_FEATURES, *CONTEXT_LIVE_FEATURES, *FUTURE_FEATURES),
}

ID_COLUMNS = ("session_id", "participant_id", "synthetic", "pocket_side", "step_id", schema.TIME)
LABEL_INFO_COLUMNS = ("foot_label", "label_source", "reviewed", "is_valid")
STATUS_COLUMNS = ("window_status", "overlaps_neighbor", "exclusion_reason")


def feature_names(feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set {feature_set!r}; choose from {sorted(FEATURE_SETS)}")
    return list(FEATURE_SETS[feature_set])


def window_features(window: pd.DataFrame, center: float, config: FeatureConfig) -> dict[str, float]:
    """Features of one step window (see preprocessing.preprocess for the input columns)."""
    out: dict[str, float] = {}
    for axis in schema.AXES:
        values = window[axis].to_numpy(dtype=float)
        out[f"{axis}_max"] = float(values.max())
        out[f"{axis}_min"] = float(values.min())
        out[f"{axis}_mean"] = float(values.mean())
        out[f"{axis}_std"] = float(values.std())

    t = window[schema.TIME].to_numpy(dtype=float)
    magnitude = window["magnitude"].to_numpy(dtype=float)
    dynamic = window["dynamic_magnitude"].to_numpy(dtype=float)
    out["magnitude_mean"] = float(magnitude.mean())
    out["magnitude_std"] = float(magnitude.std())
    out["magnitude_max"] = float(magnitude.max())
    out["magnitude_peak_to_peak"] = float(magnitude.max() - magnitude.min())

    top = int(np.argmax(dynamic))
    if 0 < top < len(dynamic) - 1:
        out["peak_prominence"] = float(peak_prominences(dynamic, [top])[0][0])
    else:
        out["peak_prominence"] = 0.0
    out["peak_time_offset"] = float(t[int(np.argmax(magnitude))] - center)
    # Mean squared gravity-adjusted magnitude (g^2), so windows of different lengths are comparable.
    out["signal_energy"] = float(np.mean(dynamic ** 2))
    peaks, _ = find_peaks(dynamic, prominence=config.window_peak_prominence_g)
    out["n_peaks"] = float(len(peaks))

    vertical = window["vertical"].to_numpy(dtype=float)
    horizontal = window["horizontal"].to_numpy(dtype=float)
    out["vertical_max"] = float(vertical.max())
    out["vertical_min"] = float(vertical.min())
    out["vertical_std"] = float(vertical.std())
    out["horizontal_max"] = float(horizontal.max())
    out["horizontal_mean"] = float(horizontal.mean())
    return out


def step_features(processed: pd.DataFrame, step_times: np.ndarray, session: Session,
                  config: PipelineConfig) -> pd.DataFrame:
    """Window status and all features for the given (sorted) step times.

    Used both with annotated footfalls (training/evaluation) and with detected candidates (inference),
    so the two paths compute identical features. Context features use the same list of times: earlier
    annotated steps during training, earlier detected steps during inference.
    """
    step_times = np.asarray(step_times, dtype=float)
    windows = extract_windows(processed, step_times, session.median_interval, session.gaps, config.segmentation)
    rows = []
    for window in windows:
        row = {"window_status": window.status}
        row.update(window_features(window.data, window.center, config.features) if window.ok
                   else {name: np.nan for name in WINDOW_FEATURES})
        rows.append(row)
    table = pd.DataFrame(rows, columns=["window_status", *WINDOW_FEATURES])
    table["overlaps_neighbor"] = overlapping_neighbors(step_times, config.segmentation) if len(step_times) else []
    max_interval = config.features.max_step_interval_seconds
    table["time_since_previous_step"] = nearest_previous(step_times, max_interval)
    table["time_until_next_step"] = nearest_next(step_times, max_interval)
    table["pocket_side_is_left"] = 1.0 if session.pocket_side == "left" else 0.0
    return table


def session_step_table(session: Session, config: PipelineConfig) -> pd.DataFrame:
    """One row per annotated step with identifiers, label info, window status, exclusion reason and features."""
    labels = session.labels
    if labels.empty:
        return pd.DataFrame(columns=[*ID_COLUMNS, *LABEL_INFO_COLUMNS, *STATUS_COLUMNS, *FEATURE_SETS["offline"]])
    processed = preprocess(session, config.preprocessing)
    times = labels[schema.TIME].to_numpy(dtype=float)
    features = step_features(processed, times, session, config)

    table = pd.DataFrame({
        "session_id": session.session_id,
        "participant_id": session.participant_id,
        "synthetic": session.synthetic,
        "pocket_side": session.pocket_side,
        "step_id": labels["step_id"].to_numpy(),
        schema.TIME: times,
        "foot_label": labels["foot_label"].to_numpy(),
        "label_source": labels["label_source"].to_numpy(),
        "reviewed": labels["reviewed"].to_numpy(),
        "is_valid": labels["is_valid"].to_numpy(),
    })
    table = pd.concat([table, features.reset_index(drop=True)], axis=1)
    include_unreviewed = config.training.include_unreviewed
    reasons = []
    for row in table.itertuples(index=False):
        reason = label_exclusion_reason(row.foot_label, bool(row.is_valid), bool(row.reviewed), include_unreviewed)
        if not reason and row.window_status != OK:
            reason = f"window_{row.window_status}"
        reasons.append(reason)
    table["exclusion_reason"] = reasons
    return table[[*ID_COLUMNS, *LABEL_INFO_COLUMNS, *STATUS_COLUMNS, *FEATURE_SETS["offline"]]]


def build_feature_table(sessions: list[Session], config: PipelineConfig) -> tuple[pd.DataFrame, list[Session]]:
    """Step table for every session that passed validation, plus the list of rejected sessions."""
    usable = [s for s in sessions if s.report.ok]
    rejected = [s for s in sessions if not s.report.ok]
    tables = [session_step_table(s, config) for s in usable]
    tables = [t for t in tables if not t.empty]
    if not tables:
        return pd.DataFrame(columns=[*ID_COLUMNS, *LABEL_INFO_COLUMNS, *STATUS_COLUMNS, *FEATURE_SETS["offline"]]), rejected
    return pd.concat(tables, ignore_index=True), rejected


def exclusion_summary(table: pd.DataFrame) -> dict[str, int]:
    counts = table["exclusion_reason"].replace("", "included").value_counts()
    return {str(k): int(v) for k, v in counts.items()}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract one feature row per annotated step.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="ml/data/processed/step_features.csv")
    parser.add_argument("--config", help="Optional pipeline config JSON")
    parser.add_argument("--include-unreviewed", action="store_true",
                        help="Treat unreviewed assisted labels as usable (not recommended)")
    args = parser.parse_args(argv)

    config = PipelineConfig.load(args.config) if args.config else PipelineConfig()
    if args.include_unreviewed:
        config.training.include_unreviewed = True
    sessions = load_sessions(args.data_dir, config)
    table, rejected = build_feature_table(sessions, config)
    for session in rejected:
        print(f"Skipped rejected session:\n{session.report.format()}")
    table.to_csv(args.output, index=False)
    print(f"Wrote {len(table)} step rows from {len(sessions) - len(rejected)} session(s) to {args.output}")
    for reason, count in exclusion_summary(table).items():
        print(f"  {reason}: {count}")
    if table["synthetic"].any():
        print("Note: the table contains synthetic sessions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
