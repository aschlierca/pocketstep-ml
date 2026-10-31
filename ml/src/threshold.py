"""Per-user step-detection threshold for the Unity app.

The Unity app counts steps with a simple detector (unity/Plugins/iOS/PocketStepCounter.mm):

    filtered = 0.6 * filtered + 0.4 * |a|     once per 60 Hz accelerometer sample, |a| in g with gravity
    step  when filtered > threshold and more than 0.45 s have passed since the previous step

One threshold does not suit everyone: soft footfalls never reach it, and a bouncy walker crosses it
between footfalls. This module learns, from labeled sessions of many people, how to pick the
threshold for a new user from a short calibration walk with a known number of steps. The detector
here is a line-by-line copy of the plugin's, so a threshold chosen here behaves the same on the phone.

Methods, compared with leave-one-participant-out evaluation on the steps after each calibration walk:

    current      the app's built-in threshold (1.45 g)
    fixed        the single threshold that works best for the training participants
    count_match  the threshold at which the calibration walk counts exactly the known number of steps
    model        ridge regression from calibration-walk features to the best threshold
    oracle       each session's best threshold on its own evaluation steps; an upper limit, not usable live

The exported model (threshold_model.json) names whichever of fixed / count_match / model scored best,
and the plugin applies that method.

Usage:
    python -m ml.src.threshold train --data-dir ml/data/raw --output ml/models/threshold_model.json
    python -m ml.src.threshold calibrate --model ml/models/threshold_model.json --input <session dir> --steps 20
    python -m ml.src.threshold golden --output unity/Tests/golden.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import schema
from .config import PipelineConfig, ThresholdConfig
from .preprocessing import ValidationReport, contiguous_segments, load_sensor, load_sessions, Session
from .segmentation import match_steps
from .train import TrainingDataError

MODEL_FORMAT = "pocketstep-threshold-model"
MODEL_VERSION = 1
METHODS = ("current", "fixed", "count_match", "model", "oracle")
# Methods the app can use, simplest first; on a tie the simpler one is recommended.
SELECTABLE = ("fixed", "count_match", "model")
FEATURES = ("count_match_g", "peak_nth_g", "peak_median_g", "peak_next_g")


# ---------------------------------------------------------------------------
# The Unity detector
# ---------------------------------------------------------------------------

def resample(sensor: pd.DataFrame, gaps: list[tuple[float, float]], rate_hz: float) -> tuple[np.ndarray, ...]:
    """Linear interpolation onto the app's sample rate, separately in each gap-free segment, so no
    samples are invented inside a gap. Returns (t, x, y, z)."""
    t = sensor[schema.TIME].to_numpy(dtype=float)
    axes = [sensor[a].to_numpy(dtype=float) for a in schema.AXES]
    dt = 1.0 / rate_hz
    parts: list[list[np.ndarray]] = [[], [], [], []]
    for seg in contiguous_segments(t, gaps):
        ts = t[seg]
        if len(ts) < 2:
            continue
        grid = ts[0] + dt * np.arange(int(math.floor((ts[-1] - ts[0]) / dt)) + 1)
        parts[0].append(grid)
        for k, values in enumerate(axes, start=1):
            parts[k].append(np.interp(grid, ts, values[seg]))
    if not parts[0]:
        return tuple(np.array([], dtype=float) for _ in range(4))
    return tuple(np.concatenate(p) for p in parts)


def filtered_magnitude(x: np.ndarray, y: np.ndarray, z: np.ndarray, cfg: ThresholdConfig) -> np.ndarray:
    """The plugin's low-pass filter, evaluated in the same order so results match bit for bit."""
    magnitude = np.sqrt(x * x + y * y + z * z)
    w = cfg.filter_previous_weight
    v = 1.0 - w
    out = np.empty(len(magnitude))
    f = cfg.filter_initial_g
    for i, m in enumerate(magnitude.tolist()):
        f = w * f + v * m
        out[i] = f
    return out


def detect_steps(times: np.ndarray, filtered: np.ndarray, threshold: float, min_interval: float) -> np.ndarray:
    """Indices of counted steps: filtered > threshold and more than min_interval since the last step."""
    candidates = np.flatnonzero(filtered > threshold)
    ct = times[candidates]
    steps = []
    last = -math.inf
    k = 0
    while k < len(candidates):
        if ct[k] - last > min_interval:
            steps.append(candidates[k])
            last = ct[k]
            k += 1
        else:
            # Skip ahead to the cooldown's end, then check exactly as the plugin does.
            k = max(k + 1, int(np.searchsorted(ct, last + min_interval - 1e-9, side="left")))
    return np.array(steps, dtype=int)


def threshold_grid(cfg: ThresholdConfig) -> list[float]:
    n = int(round((cfg.grid_max_g - cfg.grid_min_g) / cfg.grid_step_g))
    return [cfg.grid_min_g + k * cfg.grid_step_g for k in range(n + 1)]


def _median(sorted_values: list[float]) -> float:
    n = len(sorted_values)
    mid = n // 2
    return sorted_values[mid] if n % 2 else (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


# ---------------------------------------------------------------------------
# Calibration walk
# ---------------------------------------------------------------------------

def count_match_threshold(times: np.ndarray, filtered: np.ndarray, steps: int, cfg: ThresholdConfig) -> float:
    """Median of the thresholds whose step count is closest to the known number of steps."""
    grid = threshold_grid(cfg)
    diffs = [abs(len(detect_steps(times, filtered, g, cfg.min_interval_seconds)) - steps) for g in grid]
    best = min(diffs)
    return _median([g for g, d in zip(grid, diffs) if d == best])


def calibration_peaks(times: np.ndarray, filtered: np.ndarray, count: int, min_interval: float) -> list[float]:
    """Heights of the `count` highest local maxima at least min_interval apart, highest first."""
    f = filtered
    if len(f) < 3:
        return []
    idx = np.flatnonzero((f[1:-1] > f[:-2]) & (f[1:-1] >= f[2:])) + 1
    order = idx[np.argsort(-f[idx], kind="stable")]  # ties keep the earlier sample first
    accepted_times: list[float] = []
    heights: list[float] = []
    for i in order:
        if len(heights) == count:
            break
        if all(abs(times[i] - ta) >= min_interval for ta in accepted_times):
            accepted_times.append(float(times[i]))
            heights.append(float(f[i]))
    return heights


def calibration_features(times: np.ndarray, filtered: np.ndarray, steps: int, cfg: ThresholdConfig) -> dict:
    """What the app measures on the calibration walk. Heights are in g of filtered magnitude.

    peak_nth_g: the steps-th highest peak (the softest real footfall, if peaks are footfalls)
    peak_next_g: the next one down (the largest peak that is probably not a footfall)
    """
    heights = calibration_peaks(times, filtered, steps + 1, cfg.min_interval_seconds)
    top = heights[:steps]
    floor = cfg.filter_initial_g
    return {
        "count_match_g": count_match_threshold(times, filtered, steps, cfg),
        "peak_nth_g": top[-1] if top else floor,
        "peak_median_g": _median(sorted(top)) if top else floor,
        "peak_next_g": heights[steps] if len(heights) > steps else floor,
    }


def linear_threshold(model: dict, features: dict) -> float:
    value = model["intercept"]
    for name, mean, scale, coef in zip(model["features"], model["feature_mean"], model["feature_scale"],
                                       model["coefficients"]):
        value += coef * ((features[name] - mean) / scale)
    return value


def choose_threshold(model_file: dict, features: dict) -> float:
    """The threshold the plugin applies after calibration, clipped to the searched range."""
    method = model_file["recommended_method"]
    if method == "fixed":
        value = model_file["fixed_threshold_g"]
    elif method == "count_match":
        value = features["count_match_g"]
    else:
        value = linear_threshold(model_file["model"], features)
    cal = model_file["calibration"]
    return min(max(value, cal["grid_min_g"]), cal["grid_max_g"])


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@dataclass
class CalibrationSession:
    session_id: str
    participant_id: str
    synthetic: bool
    times: np.ndarray
    filtered: np.ndarray
    calibration_end: float
    truth: np.ndarray            # usable annotated steps after the calibration walk
    ignored: np.ndarray          # invalid / unreviewed labels after the calibration walk
    features: dict
    grid_f1: np.ndarray = field(default_factory=lambda: np.array([]))
    oracle_g: float = float("nan")


def score(cs: CalibrationSession, threshold: float, cfg: ThresholdConfig) -> dict:
    """Detection quality on the steps after the calibration walk. The detector runs over the whole
    session, as it would on the phone, and only steps after the walk are scored."""
    detected = cs.times[detect_steps(cs.times, cs.filtered, threshold, cfg.min_interval_seconds)]
    detected = detected[detected > cs.calibration_end]
    if len(cs.ignored):
        near = np.abs(detected[:, None] - cs.ignored[None, :]).min(axis=1) <= cfg.match_tolerance_seconds
        detected = detected[~near]
    m = match_steps(detected, cs.truth, cfg.match_tolerance_seconds)
    count_error = m["detected_steps"] - m["annotated_steps"]
    return {"threshold_g": float(threshold), "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
            "detected_steps": m["detected_steps"], "annotated_steps": m["annotated_steps"],
            "count_error": count_error, "abs_count_error_pct": 100.0 * abs(count_error) / m["annotated_steps"]}


def _best_of(grid: list[float], f1: np.ndarray, tolerance: float) -> float:
    best = float(np.max(f1))
    return _median([g for g, v in zip(grid, f1) if v >= best - tolerance])


def prepare_session(session: Session, cfg: ThresholdConfig) -> tuple[Optional[CalibrationSession], str]:
    """(session ready for calibration, '') or (None, why it cannot be used)."""
    if not session.report.ok:
        return None, "failed validation"
    labels = session.labels
    if labels.empty:
        return None, "no annotated steps"
    usable = labels["is_valid"] & (labels["reviewed"] | cfg.include_unreviewed)
    times = np.sort(labels.loc[usable, schema.TIME].to_numpy(dtype=float))
    needed = cfg.calibration_steps + cfg.min_evaluation_steps
    if len(times) < needed:
        return None, (f"{len(times)} usable annotated steps; needs {needed} "
                      f"({cfg.calibration_steps} for calibration + {cfg.min_evaluation_steps} to score)")
    calibration_end = times[cfg.calibration_steps - 1] + cfg.calibration_tail_seconds
    truth = times[times > calibration_end]
    if len(truth) < cfg.min_evaluation_steps:
        return None, f"only {len(truth)} annotated steps after the calibration walk"
    other = np.sort(labels.loc[~usable, schema.TIME].to_numpy(dtype=float))
    t, x, y, z = resample(session.sensor, session.gaps, cfg.sample_rate_hz)
    filtered = filtered_magnitude(x, y, z, cfg)
    walk = t <= calibration_end
    cs = CalibrationSession(
        session_id=session.session_id, participant_id=session.participant_id, synthetic=session.synthetic,
        times=t, filtered=filtered, calibration_end=float(calibration_end), truth=truth,
        ignored=other[other > calibration_end],
        features=calibration_features(t[walk], filtered[walk], cfg.calibration_steps, cfg))
    grid = threshold_grid(cfg)
    cs.grid_f1 = np.array([score(cs, g, cfg)["f1"] for g in grid])
    cs.oracle_g = _best_of(grid, cs.grid_f1, cfg.best_f1_tolerance)
    return cs, ""


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def best_fixed_threshold(sessions: list[CalibrationSession], cfg: ThresholdConfig) -> float:
    mean_f1 = np.mean(np.stack([s.grid_f1 for s in sessions]), axis=0)
    return _best_of(threshold_grid(cfg), mean_f1, cfg.best_f1_tolerance)


def fit_linear(sessions: list[CalibrationSession], cfg: ThresholdConfig) -> dict:
    X = np.array([[s.features[name] for name in FEATURES] for s in sessions])
    y = np.array([s.oracle_g for s in sessions])
    scaler = StandardScaler().fit(X)
    ridge = Ridge(alpha=cfg.ridge_alpha).fit(scaler.transform(X), y)
    return {"features": list(FEATURES), "feature_mean": scaler.mean_.tolist(),
            "feature_scale": scaler.scale_.tolist(), "coefficients": ridge.coef_.tolist(),
            "intercept": float(ridge.intercept_)}


def _clip(value: float, cfg: ThresholdConfig) -> float:
    return min(max(value, cfg.grid_min_g), cfg.grid_max_g)


def cross_validate(sessions: list[CalibrationSession], cfg: ThresholdConfig) -> pd.DataFrame:
    """Leave one participant out: every threshold for a participant comes from the other participants
    and that participant's own calibration walk, never from their evaluation steps."""
    rows = []
    for held_out in sorted({s.participant_id for s in sessions}):
        train = [s for s in sessions if s.participant_id != held_out]
        fixed = best_fixed_threshold(train, cfg)
        linear = fit_linear(train, cfg)
        for s in (s for s in sessions if s.participant_id == held_out):
            thresholds = {"current": cfg.current_threshold_g, "fixed": fixed,
                          "count_match": _clip(s.features["count_match_g"], cfg),
                          "model": _clip(linear_threshold(linear, s.features), cfg), "oracle": s.oracle_g}
            for method, value in thresholds.items():
                rows.append({"participant_id": s.participant_id, "session_id": s.session_id, "method": method,
                             **score(s, value, cfg)})
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> dict:
    summary = {}
    for method in METHODS:
        r = rows[rows["method"] == method]
        summary[method] = {"sessions": int(len(r)), "mean_f1": float(r["f1"].mean()),
                           "median_f1": float(r["f1"].median()),
                           "mean_abs_count_error_pct": float(r["abs_count_error_pct"].mean()),
                           "mean_count_error_steps": float(r["count_error"].mean())}
    return summary


def recommend(summary: dict) -> str:
    best = SELECTABLE[0]
    for method in SELECTABLE[1:]:
        if summary[method]["mean_f1"] > summary[best]["mean_f1"]:
            best = method
    return best


def model_file(cfg: ThresholdConfig, recommended: str, fixed: float, linear: dict, trained_on: dict) -> dict:
    return {
        "format": MODEL_FORMAT,
        "version": MODEL_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "detector": {"sample_rate_hz": cfg.sample_rate_hz, "filter_previous_weight": cfg.filter_previous_weight,
                     "filter_initial_g": cfg.filter_initial_g, "min_interval_seconds": cfg.min_interval_seconds},
        "calibration": {"steps": cfg.calibration_steps, "grid_min_g": cfg.grid_min_g, "grid_max_g": cfg.grid_max_g,
                        "grid_step_g": cfg.grid_step_g},
        "recommended_method": recommended,
        "fixed_threshold_g": fixed,
        "model": linear,
        "trained_on": trained_on,
    }


def config_from_model(model: dict, base: Optional[ThresholdConfig] = None) -> ThresholdConfig:
    d, c = model["detector"], model["calibration"]
    return replace(base or ThresholdConfig(), sample_rate_hz=d["sample_rate_hz"],
                   filter_previous_weight=d["filter_previous_weight"], filter_initial_g=d["filter_initial_g"],
                   min_interval_seconds=d["min_interval_seconds"], calibration_steps=c["steps"],
                   grid_min_g=c["grid_min_g"], grid_max_g=c["grid_max_g"], grid_step_g=c["grid_step_g"])


def train_threshold_model(data_dir: str | Path, config: Optional[PipelineConfig] = None,
                          output_path: str | Path = "ml/models/threshold_model.json",
                          reports_dir: str | Path = "ml/reports/threshold", log=print) -> dict:
    """Evaluates every method, then fits the exported model on all usable sessions.
    Raises TrainingDataError when too few participants have usable sessions."""
    config = config or PipelineConfig()
    cfg = config.threshold
    prepared, skipped = [], {}
    for session in load_sessions(data_dir, config):
        cs, reason = prepare_session(session, cfg)
        if cs is None:
            skipped[session.session_id] = reason
        else:
            prepared.append(cs)
    participants = sorted({s.participant_id for s in prepared})
    if len(participants) < cfg.min_participants:
        detail = "; ".join(f"{k}: {v}" for k, v in skipped.items())
        raise TrainingDataError(
            f"Only {len(participants)} participant(s) have sessions usable for threshold calibration; "
            f"leave-one-participant-out evaluation needs at least {cfg.min_participants}. Each session needs "
            f"{cfg.calibration_steps + cfg.min_evaluation_steps}+ reviewed, valid step labels."
            + (f" Skipped: {detail}." if detail else ""))

    rows = cross_validate(prepared, cfg)
    summary = summarize(rows)
    recommended = recommend(summary)
    synthetic = any(s.synthetic for s in prepared)
    trained_on = {"sessions": len(prepared), "participants": len(participants), "synthetic": synthetic}
    exported = model_file(cfg, recommended, best_fixed_threshold(prepared, cfg), fit_linear(prepared, cfg),
                          trained_on)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(exported, indent=2) + "\n")

    per_session = pd.DataFrame([{
        "participant_id": s.participant_id, "session_id": s.session_id, "oracle_threshold_g": s.oracle_g,
        "evaluation_steps": len(s.truth), **s.features,
        "step_gaps_within_cooldown": int((np.diff(s.truth) <= cfg.min_interval_seconds).sum()),
    } for s in prepared])
    limitations = [
        "The Unity detector is reproduced from the plugin source. Recordings are made at 100 Hz and "
        "resampled to 60 Hz by linear interpolation, which can differ slightly from Core Motion's own 60 Hz output.",
        f"Evaluated with leave-one-participant-out over {len(participants)} participant(s); with few "
        "participants these numbers are uncertain.",
        f"The {cfg.min_interval_seconds:.2f} s cooldown allows at most {1 / cfg.min_interval_seconds:.1f} steps/s; "
        f"{int(per_session['step_gaps_within_cooldown'].sum())} annotated step gap(s) are shorter and are "
        "missed at any threshold.",
    ]
    if synthetic:
        limitations.insert(0, "SYNTHETIC DATA: these results only show that the pipeline runs. They say nothing "
                              "about real walkers.")
    report = {
        "recommended_method": recommended,
        "summary": summary,
        "dataset": {**trained_on, "participant_ids": participants, "skipped_sessions": skipped},
        "exported_model": exported,
        "config": asdict(cfg),
        "limitations": limitations,
    }
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "threshold_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    rows.to_csv(reports_dir / "threshold_cv_results.csv", index=False, float_format="%.4f")
    per_session.to_csv(reports_dir / "threshold_sessions.csv", index=False, float_format="%.4f")
    log(format_summary(report))
    return report


def format_summary(report: dict) -> str:
    lines = ["| Method | Threshold source | Mean F1 | Mean step-count error |", "|---|---|---|---|"]
    source = {"current": "built into the app today", "fixed": "best single value for other people",
              "count_match": "calibration walk only", "model": "calibration walk + model",
              "oracle": "best possible per session (upper limit)"}
    for method in METHODS:
        s = report["summary"][method]
        mark = " **(recommended)**" if method == report["recommended_method"] else ""
        lines.append(f"| `{method}`{mark} | {source[method]} | {s['mean_f1']:.3f} | "
                     f"{s['mean_abs_count_error_pct']:.1f}% |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Calibrating a recording
# ---------------------------------------------------------------------------

def calibrate_recording(model: dict, input_path: str | Path, steps: int) -> tuple[float, dict]:
    """Threshold for a recording of a calibration walk with `steps` steps (the whole recording is the walk)."""
    cfg = config_from_model(model)
    input_path = Path(input_path)
    sensor_path = input_path / schema.SENSOR_FILE if input_path.is_dir() else input_path
    report = ValidationReport(session_dir=str(sensor_path.parent))
    sensor, gaps, _ = load_sensor(sensor_path, report, PipelineConfig().preprocessing)
    if not report.ok:
        raise ValueError(f"The recording failed validation:\n{report.format()}")
    t, x, y, z = resample(sensor, gaps, cfg.sample_rate_hz)
    features = calibration_features(t, filtered_magnitude(x, y, z, cfg), steps, cfg)
    return choose_threshold(model, features), features


def golden_fixture(seed: int = 11) -> dict:
    """A recorded-looking calibration walk with the values the plugin must reproduce, for
    unity/Tests (checks the Objective-C++ plugin against this module)."""
    from .synthetic import simulate_session
    cfg = ThresholdConfig()
    rng = np.random.default_rng(seed)
    sensor, _, _ = simulate_session(rng, participant_id="SYN-GOLDEN", session_id="SYN-GOLDEN", pocket_side="right",
                                    starting_foot="left", n_steps=cfg.calibration_steps,
                                    walking_condition="comfortable", traits={"impact_g": 0.9, "midstep_g": 0.4})
    t, x, y, z = (np.round(a, 6) for a in resample(sensor, [], cfg.sample_rate_hz))
    filtered = filtered_magnitude(x, y, z, cfg)
    features = calibration_features(t, filtered, cfg.calibration_steps, cfg)
    linear = {"features": list(FEATURES), "feature_mean": [1.4, 1.5, 1.7, 1.2],
              "feature_scale": [0.2, 0.25, 0.3, 0.1], "coefficients": [0.05, 0.08, 0.02, 0.03], "intercept": 1.45}
    base = model_file(cfg, "model", 1.37, linear, {"sessions": 0, "participants": 0, "synthetic": True})
    base["created_utc"] = "2026-01-01T00:00:00Z"
    expected = {method: choose_threshold({**base, "recommended_method": method}, features) for method in SELECTABLE}
    return {
        "description": "Generated by python -m ml.src.threshold golden. Do not edit by hand.",
        "steps": cfg.calibration_steps,
        "model": base,
        "samples": np.column_stack([t, x, y, z]).tolist(),
        "expected": {
            "filtered_last": float(filtered[-1]),
            "step_count_at_current_threshold": int(len(detect_steps(t, filtered, cfg.current_threshold_g,
                                                                    cfg.min_interval_seconds))),
            "features": [features[name] for name in FEATURES],
            "thresholds": expected,
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Per-user step-detection threshold for the Unity app.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train", help="Evaluate and export the threshold model")
    p_train.add_argument("--data-dir", default="ml/data/raw")
    p_train.add_argument("--output", default="ml/models/threshold_model.json")
    p_train.add_argument("--reports-dir", default="ml/reports/threshold")
    p_train.add_argument("--config", help="Optional pipeline config JSON")
    p_cal = sub.add_parser("calibrate", help="Threshold for one recorded calibration walk")
    p_cal.add_argument("--model", default="ml/published/threshold_model.json")
    p_cal.add_argument("--input", required=True, help="Session folder or sensor_data.csv")
    p_cal.add_argument("--steps", type=int, required=True, help="Steps taken during the recording")
    p_gold = sub.add_parser("golden", help="Write the fixture that unity/Tests checks the plugin against")
    p_gold.add_argument("--output", default="unity/Tests/golden.json")
    args = parser.parse_args(argv)

    if args.command == "train":
        config = PipelineConfig.load(args.config) if args.config else PipelineConfig()
        try:
            train_threshold_model(args.data_dir, config, args.output, args.reports_dir)
        except TrainingDataError as exc:
            print(f"Not trained: {exc}", file=sys.stderr)
            return 2
        print(f"Wrote {args.output} and reports to {args.reports_dir}/")
    elif args.command == "calibrate":
        model = json.loads(Path(args.model).read_text())
        threshold, features = calibrate_recording(model, args.input, args.steps)
        print(json.dumps({"threshold_g": threshold, "method": model["recommended_method"], "features": features},
                         indent=2))
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(golden_fixture()) + "\n")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
