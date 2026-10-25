"""Run a trained model on a new recording.

Steps are found by automatic peak detection, then each candidate is classified left/right or
marked unknown when confidence is below the threshold. Step labels in the input folder are never
read for prediction; with --evaluate-against-labels they are read afterwards, only to score
detection and classification separately.

Usage:
    python -m ml.src.predict --model-path ml/models/pocketstep.joblib --input path/to/sensor_data.csv
    python -m ml.src.predict --model-path ... --input path/to/session_folder --evaluate-against-labels
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import schema
from .config import PipelineConfig
from .features import step_features
from .models import ModelBundle
from .preprocessing import Session, ValidationReport, load_labels, load_metadata, load_sensor, preprocess
from .segmentation import OK, detect_candidate_steps, label_exclusion_reason, match_steps

PREDICTION_COLUMNS = ["candidate_id", schema.TIME, "predicted_label", "confidence", "probability_left",
                      "probability_right", "abstained", "window_status"]


class RecordingError(RuntimeError):
    pass


def load_recording(input_path: str | Path, config: PipelineConfig, pocket_side: Optional[str] = None) -> Session:
    """Sensor data and metadata only; labels are deliberately left empty."""
    input_path = Path(input_path)
    sensor_path = input_path / schema.SENSOR_FILE if input_path.is_dir() else input_path
    folder = sensor_path.parent
    report = ValidationReport(session_dir=str(folder))
    metadata_path = folder / schema.METADATA_FILE
    metadata = load_metadata(metadata_path, report) if metadata_path.exists() else {}
    report.issues = [i for i in report.issues if i.code != "metadata_missing_key"]
    if pocket_side:
        metadata["pocket_side"] = pocket_side
    if metadata.get("pocket_side") not in schema.SIDES:
        raise RecordingError("The pocket side is unknown. Put session_metadata.json next to the sensor file "
                             "or pass --pocket-side left|right.")
    report.session_id = metadata.get("session_id") or folder.name
    sensor, gaps, median_dt = load_sensor(sensor_path, report, config.preprocessing, metadata)
    if not report.ok:
        raise RecordingError(f"The recording failed validation:\n{report.format()}")
    empty_labels = pd.DataFrame(columns=["step_id", schema.TIME, "foot_label", "label_source", "is_valid",
                                         "reviewed", "notes"])
    return Session(folder, metadata, sensor, empty_labels, report, gaps, median_dt)


def predict_recording(bundle: ModelBundle, session: Session, threshold: Optional[float] = None) -> pd.DataFrame:
    config = PipelineConfig.from_dict(bundle.config)
    processed = preprocess(session, config.preprocessing)
    candidates = detect_candidate_steps(processed, session.median_interval, config.segmentation)
    features = step_features(processed, candidates, session, config)
    out = pd.DataFrame({"candidate_id": np.arange(1, len(candidates) + 1), schema.TIME: candidates,
                        "predicted_label": "unknown", "confidence": np.nan, "probability_left": np.nan,
                        "probability_right": np.nan, "abstained": True, "window_status": features["window_status"]})
    ok = (features["window_status"] == OK).to_numpy()
    if ok.any():
        labels, confidence, abstained, p_right = bundle.predict(features[ok], threshold)
        out.loc[ok, "predicted_label"] = labels
        out.loc[ok, "confidence"] = confidence
        out.loc[ok, "probability_right"] = p_right
        out.loc[ok, "probability_left"] = 1 - p_right
        out.loc[ok, "abstained"] = abstained
    return out[PREDICTION_COLUMNS]


def evaluate_against_labels(predictions: pd.DataFrame, labels: pd.DataFrame, tolerance: float) -> dict:
    """Scores the two problems separately: (1) did detection find the annotated footfalls, and
    (2) for matched steps with a usable left/right label, was the foot right?"""
    annotated = labels[schema.TIME].to_numpy(dtype=float)
    detection = match_steps(predictions[schema.TIME].to_numpy(dtype=float), annotated, tolerance)
    pairs = detection.pop("matches")
    rows = []
    for label_idx, pred_idx in pairs:
        label = labels.iloc[label_idx]
        if label_exclusion_reason(label["foot_label"], bool(label["is_valid"]), bool(label["reviewed"]), False):
            continue
        rows.append((label["foot_label"], predictions.iloc[pred_idx]["predicted_label"]))
    truth = np.array([r[0] for r in rows])
    predicted = np.array([r[1] for r in rows])
    covered = predicted != "unknown"
    classification = {
        "matched_steps_with_usable_labels": int(len(rows)),
        "coverage": float(covered.mean()) if len(rows) else None,
        "accuracy_on_covered": float(np.mean(truth[covered] == predicted[covered])) if covered.any() else None,
    }
    return {"step_detection": detection, "classification_on_detected_steps": classification,
            "note": "Detection and left/right classification are separate problems and are scored separately."}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Predict left/right/unknown for steps in a new recording.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", required=True, help="sensor_data.csv or a session folder")
    parser.add_argument("--output", default="ml/reports/predictions.csv")
    parser.add_argument("--threshold", type=float, help="Confidence threshold (default: the model's)")
    parser.add_argument("--pocket-side", choices=schema.SIDES, help="Needed when there is no session_metadata.json")
    parser.add_argument("--evaluate-against-labels", action="store_true",
                        help="After predicting, score detection and classification against step_labels.csv")
    args = parser.parse_args(argv)

    bundle = ModelBundle.load(args.model_path)
    config = PipelineConfig.from_dict(bundle.config)
    try:
        session = load_recording(args.input, config, args.pocket_side)
    except RecordingError as exc:
        print(exc, file=sys.stderr)
        return 2
    for issue in session.report.issues:
        print(f"[{issue.level}] {issue.code}: {issue.message}")

    predictions = predict_recording(bundle, session, args.threshold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False, float_format="%.6f")
    counts = predictions["predicted_label"].value_counts().to_dict()
    print(f"{len(predictions)} candidate step(s) detected: {counts}. Wrote {output}")
    if bundle.training_metadata.get("synthetic_data"):
        print("NOTE: this model was trained on synthetic data.")

    if args.evaluate_against_labels:
        labels_path = session.path / schema.LABELS_FILE
        report = ValidationReport(session_dir=str(session.path))
        t = session.sensor[schema.TIME]
        labels = load_labels(labels_path, report, (float(t.iloc[0]), float(t.iloc[-1])))
        if not report.ok or labels.empty:
            print(f"Cannot evaluate: {report.format()}", file=sys.stderr)
            return 2
        result = evaluate_against_labels(predictions, labels, config.segmentation.match_tolerance_seconds)
        evaluation_path = output.with_suffix(".evaluation.json")
        evaluation_path.write_text(json.dumps(result, indent=2))
        d, c = result["step_detection"], result["classification_on_detected_steps"]
        print(f"Step detection: precision {d['precision']:.3f}, recall {d['recall']:.3f}, F1 {d['f1']:.3f} "
              f"(tolerance {d['tolerance_seconds']} s)")
        acc = c["accuracy_on_covered"]
        print(f"Left/right on detected steps: coverage {c['coverage'] or 0:.1%}, accuracy "
              f"{'n/a' if acc is None else f'{acc:.3f}'}. Wrote {evaluation_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
