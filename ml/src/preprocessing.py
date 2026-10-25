"""Load, validate and preprocess exported PocketStep sessions.

Raw files are only read, never modified. Problems are reported as issues. Cleaning is limited to
removing rows that cannot be used (non-numeric values, duplicate timestamps); every removal is
counted, and a session that would lose more than a small fraction of rows is rejected.

Usage:
    python -m ml.src.preprocessing --data-dir ml/data/raw
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from . import schema
from .config import PipelineConfig, PreprocessingConfig

# Median |a| of a phone that is mostly upright and slow-moving should be close to 1 g.
MAGNITUDE_SANITY_RANGE_G = (0.8, 1.2)
# iPhone accelerometers saturate around +-8 g (some models 16 g).
SATURATION_G = 7.9


@dataclass
class Issue:
    level: str  # "error" or "warning"
    code: str
    message: str


@dataclass
class ValidationReport:
    session_dir: str
    session_id: Optional[str] = None
    issues: list[Issue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def error(self, code: str, message: str) -> None:
        self.issues.append(Issue("error", code, message))

    def warn(self, code: str, message: str) -> None:
        self.issues.append(Issue("warning", code, message))

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    def codes(self, level: Optional[str] = None) -> list[str]:
        return [i.code for i in self.issues if level is None or i.level == level]

    def to_dict(self) -> dict:
        return {"session_dir": self.session_dir, "session_id": self.session_id, "ok": self.ok,
                "issues": [asdict(i) for i in self.issues], "stats": self.stats}

    def format(self) -> str:
        name = self.session_id or self.session_dir
        n_err = len(self.codes("error"))
        n_warn = len(self.codes("warning"))
        head = f"{name}: {'OK' if self.ok else 'REJECTED'} ({n_err} error(s), {n_warn} warning(s))"
        lines = [head] + [f"  [{i.level}] {i.code}: {i.message}" for i in self.issues]
        return "\n".join(lines)


@dataclass
class Session:
    path: Path
    metadata: dict
    sensor: pd.DataFrame
    labels: pd.DataFrame
    report: ValidationReport
    gaps: list[tuple[float, float]]
    median_interval: float

    @property
    def session_id(self) -> str:
        return str(self.metadata.get("session_id") or self.path.name)

    @property
    def participant_id(self) -> str:
        return str(self.metadata.get("participant_id", "unknown"))

    @property
    def pocket_side(self) -> Optional[str]:
        return self.metadata.get("pocket_side")

    @property
    def synthetic(self) -> bool:
        return bool(self.metadata.get("synthetic", False))


# ---------------------------------------------------------------------------
# Discovery and loading
# ---------------------------------------------------------------------------

def find_session_dirs(data_dir: str | Path) -> list[Path]:
    """Every directory below data_dir that contains a sensor_data.csv."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")
    return sorted({p.parent for p in root.rglob(schema.SENSOR_FILE)})


def load_sessions(data_dir: str | Path, config: Optional[PipelineConfig] = None) -> list[Session]:
    return [load_session(d, config) for d in find_session_dirs(data_dir)]


def load_session(session_dir: str | Path, config: Optional[PipelineConfig] = None) -> Session:
    config = config or PipelineConfig()
    session_dir = Path(session_dir)
    report = ValidationReport(session_dir=str(session_dir))

    metadata = load_metadata(session_dir / schema.METADATA_FILE, report)
    report.session_id = metadata.get("session_id") or session_dir.name
    sensor, gaps, median_dt = load_sensor(session_dir / schema.SENSOR_FILE, report, config.preprocessing, metadata)
    time_range = (float(sensor[schema.TIME].iloc[0]), float(sensor[schema.TIME].iloc[-1])) if len(sensor) else None
    labels = load_labels(session_dir / schema.LABELS_FILE, report, time_range)
    report.stats["synthetic"] = bool(metadata.get("synthetic", False))
    return Session(session_dir, metadata, sensor, labels, report, gaps, median_dt)


def load_metadata(path: Path, report: ValidationReport) -> dict:
    if not path.exists():
        report.error("metadata_missing", f"{path.name} not found")
        return {}
    try:
        metadata = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        report.error("metadata_unreadable", f"{path.name} is not valid JSON: {exc}")
        return {}

    missing = [k for k in schema.METADATA_REQUIRED if k not in metadata]
    critical = {"schema_version", "participant_id", "session_id", "pocket_side"}
    for key in missing:
        (report.error if key in critical else report.warn)("metadata_missing_key", f"metadata has no '{key}'")
    version = metadata.get("schema_version")
    if version is not None and str(version) not in schema.SUPPORTED_SCHEMA_VERSIONS:
        report.error("schema_version", f"schema_version {version} is not supported "
                                       f"(supported: {', '.join(schema.SUPPORTED_SCHEMA_VERSIONS)})")
    if "pocket_side" in metadata and metadata["pocket_side"] not in schema.SIDES:
        report.error("metadata_value", f"pocket_side must be left or right, got {metadata['pocket_side']!r}")
    if metadata.get("starting_foot") not in (None, *schema.SIDES):
        report.warn("metadata_value", f"starting_foot should be left or right, got {metadata['starting_foot']!r}")
    if metadata.get("recording_status") not in (None, "completed"):
        report.warn("recording_status", f"recording_status is {metadata['recording_status']!r}")
    return metadata


def load_sensor(path: Path, report: ValidationReport, config: PreprocessingConfig,
                metadata: Optional[dict] = None) -> tuple[pd.DataFrame, list[tuple[float, float]], float]:
    """Returns (clean sensor frame sorted by time, gaps, median sampling interval)."""
    empty = pd.DataFrame(columns=list(schema.SENSOR_COLUMNS), dtype=float)
    if not path.exists():
        report.error("sensor_missing", f"{path.name} not found")
        return empty, [], float("nan")
    try:
        raw = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        report.error("sensor_unreadable", f"{path.name} could not be parsed: {exc}")
        return empty, [], float("nan")

    missing = [c for c in schema.SENSOR_COLUMNS if c not in raw.columns]
    if missing:
        report.error("sensor_columns", f"{path.name} is missing columns: {', '.join(missing)}")
        return empty, [], float("nan")
    n_rows = len(raw)
    report.stats["raw_rows"] = n_rows
    if n_rows < 2:
        report.error("sensor_empty", f"{path.name} has {n_rows} data row(s)")
        return empty, [], float("nan")

    df = raw[list(schema.SENSOR_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    bad = df.isna().any(axis=1)
    if bad.any():
        report.warn("missing_values", f"{int(bad.sum())} row(s) have missing or non-numeric values and were removed")
    df = df[~bad]

    diff = (df[schema.TIME] - df[schema.ELAPSED]).abs()
    if len(diff) and diff.max() > config.time_column_tolerance:
        report.error("time_columns_disagree",
                     f"timestamp_seconds and elapsed_seconds differ by up to {diff.max():.6f} s in "
                     f"{int((diff > config.time_column_tolerance).sum())} row(s); they must be the same time basis")

    t = df[schema.TIME].to_numpy()
    backwards = int((np.diff(t) < 0).sum())
    if backwards:
        report.warn("timestamp_order", f"{backwards} timestamp(s) go backwards; rows were sorted by time")
        if backwards > 0.01 * len(t):
            report.error("timestamp_order", "more than 1% of timestamps are out of order; the file may be corrupted")
        df = df.sort_values(schema.TIME, kind="stable")

    duplicated = df[schema.TIME].duplicated(keep="first")
    if duplicated.any():
        report.warn("duplicate_timestamps", f"{int(duplicated.sum())} duplicate timestamp(s); kept the first of each")
        df = df[~duplicated]

    dropped = n_rows - len(df)
    report.stats["dropped_rows"] = dropped
    if dropped > config.max_dropped_fraction * n_rows:
        report.error("too_many_dropped_rows",
                     f"cleaning would drop {dropped} of {n_rows} rows ({dropped / n_rows:.1%}), above the "
                     f"{config.max_dropped_fraction:.0%} limit")
    if len(df) < 2:
        report.error("sensor_empty", "fewer than 2 usable rows remain")
        return df.reset_index(drop=True), [], float("nan")

    df = df.reset_index(drop=True)
    t = df[schema.TIME].to_numpy()
    gaps, median_dt = find_gaps(t, config.gap_factor)
    report.stats.update({
        "rows": len(df),
        "duration_seconds": float(t[-1] - t[0]),
        "median_interval_seconds": median_dt,
        "sampling_rate_hz": 1.0 / median_dt,
        "gap_count": len(gaps),
        "gap_total_seconds": float(sum(b - a for a, b in gaps)),
    })
    if gaps:
        largest = max(b - a for a, b in gaps)
        report.warn("sensor_gaps", f"{len(gaps)} gap(s) longer than {config.gap_factor:g}x the median interval "
                                   f"(largest {largest:.3f} s). Gaps are kept; windows overlapping them are rejected.")

    intervals = np.diff(t)
    regular = intervals[intervals <= config.gap_factor * median_dt]
    irregular = float(np.mean(np.abs(regular - median_dt) > 0.25 * median_dt)) if len(regular) else 0.0
    report.stats["irregular_interval_fraction"] = irregular
    if irregular > config.irregular_fraction_warning:
        report.warn("irregular_sampling", f"{irregular:.1%} of intervals deviate more than 25% from the median "
                                          f"({median_dt * 1000:.2f} ms)")

    stated = (metadata or {}).get("sampling_interval_seconds")
    if isinstance(stated, (int, float)) and stated > 0 and abs(stated - median_dt) / stated > 0.1:
        report.warn("sampling_interval_mismatch", f"metadata sampling_interval_seconds={stated:.5f} but the data's "
                                                  f"median interval is {median_dt:.5f}")
    stated_count = (metadata or {}).get("sample_count")
    if isinstance(stated_count, int) and stated_count not in (0, n_rows):
        report.warn("sample_count_mismatch", f"metadata sample_count={stated_count} but the file has {n_rows} rows")

    magnitude = acceleration_magnitude(df)
    median_mag = float(np.median(magnitude))
    report.stats["median_magnitude_g"] = median_mag
    if median_mag > 5:
        report.error("units", f"median |a| is {median_mag:.2f}; values look like m/s^2, but the schema requires g")
    elif not MAGNITUDE_SANITY_RANGE_G[0] <= median_mag <= MAGNITUDE_SANITY_RANGE_G[1]:
        report.warn("units", f"median |a| is {median_mag:.2f} g; expected about 1 g when the phone includes gravity")
    saturated = int((df[list(schema.AXES)].abs() >= SATURATION_G).any(axis=1).sum())
    if saturated:
        report.warn("saturation", f"{saturated} sample(s) are at or near the sensor's +-8 g limit")
    return df, gaps, median_dt


def load_labels(path: Path, report: ValidationReport, time_range: Optional[tuple[float, float]]) -> pd.DataFrame:
    columns = ["step_id", schema.TIME, "foot_label", "label_source", "is_valid", "reviewed", "notes"]
    empty = pd.DataFrame(columns=columns)
    if not path.exists():
        report.warn("labels_missing", f"{path.name} not found; the session can be used for prediction but not training")
        return empty
    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        report.error("labels_unreadable", f"{path.name} could not be parsed: {exc}")
        return empty
    except pd.errors.EmptyDataError:
        report.error("labels_unreadable", f"{path.name} is empty (not even a header row)")
        return empty

    missing = [c for c in schema.LABEL_COLUMNS if c not in raw.columns]
    if missing:
        report.error("label_columns", f"{path.name} is missing columns: {', '.join(missing)}")
        return empty
    if raw.empty:
        report.warn("labels_empty", "no steps are annotated")
        return empty

    labels = pd.DataFrame({
        "step_id": pd.to_numeric(raw["step_id"], errors="coerce"),
        schema.TIME: pd.to_numeric(raw[schema.TIME], errors="coerce"),
        "foot_label": raw["foot_label"].str.strip().str.lower(),
        "label_source": raw["label_source"].str.strip().str.lower(),
        "is_valid": raw["is_valid"].map(_parse_bool),
        "notes": raw["notes"],
    })
    if "reviewed" in raw.columns:
        labels["reviewed"] = raw["reviewed"].map(_parse_bool)
    else:
        labels["reviewed"] = labels["label_source"] == "manual"
        report.warn("labels_no_reviewed_column", "no 'reviewed' column; manual labels are treated as reviewed and "
                                                 "assisted labels as unreviewed")

    def bad(mask: pd.Series, code: str, what: str) -> None:
        if mask.any():
            rows = ", ".join(str(i + 2) for i in mask[mask].index[:5])
            report.error(code, f"{int(mask.sum())} label row(s) have {what} (file lines {rows}{', ...' if mask.sum() > 5 else ''})")

    bad(labels["step_id"].isna(), "label_value", "a non-numeric step_id")
    bad(labels[schema.TIME].isna(), "label_value", "a missing or non-numeric timestamp")
    bad(~labels["foot_label"].isin(schema.FOOT_LABELS), "label_value", f"foot_label outside {schema.FOOT_LABELS}")
    bad(~labels["label_source"].isin(schema.LABEL_SOURCES), "label_value", f"label_source outside {schema.LABEL_SOURCES}")
    bad(labels["is_valid"].isna(), "label_value", "is_valid that is not true/false")
    bad(labels["reviewed"].isna(), "label_value", "reviewed that is not true/false")
    if labels["step_id"].duplicated().any():
        report.error("label_value", "step_id values are not unique")
    if not report.ok:
        return labels

    labels = labels.astype({"step_id": int, "is_valid": bool, "reviewed": bool})
    labels = labels.sort_values(schema.TIME, kind="stable").reset_index(drop=True)
    close = np.diff(labels[schema.TIME].to_numpy()) < 0.05
    if close.any():
        report.warn("label_duplicates", f"{int(close.sum())} pair(s) of steps are less than 50 ms apart")
    if time_range is not None:
        outside = ~labels[schema.TIME].between(*time_range)
        if outside.any():
            report.warn("labels_outside_recording", f"{int(outside.sum())} step(s) fall outside the recorded time range")
    counts = labels["foot_label"].value_counts()
    report.stats.update({
        "labels": len(labels),
        "labels_left": int(counts.get("left", 0)),
        "labels_right": int(counts.get("right", 0)),
        "labels_unknown": int(counts.get("unknown", 0)),
        "labels_invalid": int((~labels["is_valid"]).sum()),
        "labels_unreviewed": int((~labels["reviewed"]).sum()),
    })
    return labels


def _parse_bool(value) -> Optional[bool]:
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def acceleration_magnitude(df: pd.DataFrame) -> np.ndarray:
    """sqrt(accel_x**2 + accel_y**2 + accel_z**2), in g."""
    xyz = df[list(schema.AXES)].to_numpy(dtype=float)
    return np.sqrt((xyz ** 2).sum(axis=1))


def find_gaps(t: np.ndarray, gap_factor: float) -> tuple[list[tuple[float, float]], float]:
    """Intervals longer than gap_factor x the median interval, plus the median interval."""
    intervals = np.diff(t)
    if len(intervals) == 0:
        return [], float("nan")
    median_dt = float(np.median(intervals[intervals > 0])) if (intervals > 0).any() else float("nan")
    idx = np.flatnonzero(intervals > gap_factor * median_dt)
    return [(float(t[i]), float(t[i + 1])) for i in idx], median_dt


def contiguous_segments(t: np.ndarray, gaps: list[tuple[float, float]]) -> list[slice]:
    """Index ranges between gaps. Filters are applied per segment so they never smear across a gap."""
    if len(t) == 0:
        return []
    breaks = [int(np.searchsorted(t, end)) for _, end in gaps]
    starts = [0, *breaks]
    ends = [*breaks, len(t)]
    return [slice(s, e) for s, e in zip(starts, ends) if e > s]


def lowpass(values: np.ndarray, fs: float, cutoff_hz: float, order: int, segments: list[slice],
            short_segment: str = "keep") -> np.ndarray:
    """Zero-phase Butterworth low-pass per segment. Zero-phase filtering uses future samples, so it is
    for offline analysis; a causal filter would be needed on-device.

    short_segment: what to do with segments too short to filter: "keep" the values or replace with the "mean".
    """
    out = np.array(values, dtype=float, copy=True)
    b, a = butter(order, cutoff_hz, btype="low", fs=fs)
    padlen = 3 * max(len(a), len(b))
    for seg in segments:
        chunk = out[seg]
        if len(chunk) > padlen:
            out[seg] = filtfilt(b, a, chunk, axis=0)
        elif short_segment == "mean" and len(chunk):
            out[seg] = chunk.mean(axis=0)
    return out


def preprocess(session: Session, config: Optional[PreprocessingConfig] = None) -> pd.DataFrame:
    """Derived signals for segmentation and features. Columns:

    - accel_x/y/z: raw axes in g (noise-filtered only if noise_filter_cutoff_hz is set)
    - magnitude_raw: |a| of the unfiltered axes; magnitude: |a| of the (possibly filtered) axes
    - gravity_x/y/z: low-pass estimate of gravity in device axes
    - linear_x/y/z: accel minus gravity
    - vertical: linear acceleration projected on the gravity direction (orientation-independent)
    - horizontal: magnitude of linear acceleration perpendicular to gravity
    - dynamic_magnitude: magnitude minus |gravity| (gravity-adjusted magnitude)
    - segment: index of the gap-free segment each sample belongs to

    The phone's axes are not assumed to line up with the body; the gravity-based components are
    the orientation-independent alternative to the raw axes.
    """
    config = config or PreprocessingConfig()
    sensor = session.sensor
    t = sensor[schema.TIME].to_numpy(dtype=float)
    fs = 1.0 / session.median_interval
    segments = contiguous_segments(t, session.gaps)
    raw = sensor[list(schema.AXES)].to_numpy(dtype=float)

    accel = raw
    if config.noise_filter_cutoff_hz:
        if config.noise_filter_cutoff_hz >= fs / 2:
            raise ValueError(f"noise_filter_cutoff_hz must be below the Nyquist frequency ({fs / 2:.1f} Hz)")
        accel = lowpass(raw, fs, config.noise_filter_cutoff_hz, config.noise_filter_order, segments)

    gravity = lowpass(accel, fs, config.gravity_cutoff_hz, 2, segments, short_segment="mean")
    linear = accel - gravity
    g_norm = np.linalg.norm(gravity, axis=1)
    g_unit = gravity / np.where(g_norm > 1e-9, g_norm, 1.0)[:, None]
    vertical = (linear * g_unit).sum(axis=1)
    horizontal = np.linalg.norm(linear - vertical[:, None] * g_unit, axis=1)
    magnitude = np.linalg.norm(accel, axis=1)

    segment_ids = np.zeros(len(t), dtype=int)
    for i, seg in enumerate(segments):
        segment_ids[seg] = i

    out = pd.DataFrame({schema.TIME: t})
    for i, axis in enumerate(schema.AXES):
        out[axis] = accel[:, i]
    out["magnitude_raw"] = np.linalg.norm(raw, axis=1)
    out["magnitude"] = magnitude
    for i, name in enumerate(("x", "y", "z")):
        out[f"gravity_{name}"] = gravity[:, i]
    for i, name in enumerate(("x", "y", "z")):
        out[f"linear_{name}"] = linear[:, i]
    out["vertical"] = vertical
    out["horizontal"] = horizontal
    out["dynamic_magnitude"] = magnitude - g_norm
    out["segment"] = segment_ids
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate exported PocketStep sessions without modifying them.")
    parser.add_argument("--data-dir", required=True, help="Folder containing session folders (searched recursively)")
    parser.add_argument("--config", help="Optional pipeline config JSON")
    parser.add_argument("--json-output", help="Also write the validation reports to this JSON file")
    args = parser.parse_args(argv)

    config = PipelineConfig.load(args.config) if args.config else PipelineConfig()
    sessions = load_sessions(args.data_dir, config)
    if not sessions:
        print(f"No sessions found under {args.data_dir} (looking for folders with {schema.SENSOR_FILE}).")
        return 1
    for session in sessions:
        print(session.report.format())
        stats = session.report.stats
        if "median_magnitude_g" in stats:
            print(f"  stats: {stats.get('rows')} rows, {stats.get('sampling_rate_hz', float('nan')):.1f} Hz, "
                  f"{stats.get('duration_seconds', 0):.1f} s, median |a| {stats['median_magnitude_g']:.3f} g, "
                  f"{stats.get('labels', 0)} labels")
        if session.synthetic:
            print("  note: synthetic session (not a real recording)")
    rejected = [s for s in sessions if not s.report.ok]
    print(f"\n{len(sessions)} session(s) checked, {len(rejected)} rejected.")
    if args.json_output:
        Path(args.json_output).write_text(json.dumps([s.report.to_dict() for s in sessions], indent=2))
        print(f"Reports written to {args.json_output}")
    return 1 if rejected else 0


if __name__ == "__main__":
    sys.exit(main())
