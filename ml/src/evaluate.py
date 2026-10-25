"""Evaluate a trained model on its held-out test groups.

Usage:
    python -m ml.src.evaluate --model-path ml/models/pocketstep.joblib

Writes ml/reports/metrics.json, ml/reports/confusion_matrix.png and ml/reports/test_predictions.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from . import schema
from .config import PipelineConfig
from .features import build_feature_table
from .models import ModelBundle
from .preprocessing import load_sessions
from .train import brier, session_fingerprint

DEVELOPMENT_TARGET_MACRO_F1 = 0.85
FEET = ["left", "right"]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Binary left/right metrics. y_pred must not contain 'unknown'."""
    if len(y_true) == 0:
        return {"n": 0}
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=FEET, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=FEET, average="macro", zero_division=0)),
        "per_foot": {foot: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                     for foot, p, r, f, s in zip(FEET, precision, recall, f1, support)},
        "confusion_matrix": {"labels": FEET,
                             "rows_true_cols_predicted": confusion_matrix(y_true, y_pred, labels=FEET).tolist()},
    }


def thresholded_metrics(y_true: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    covered = labels != "unknown"
    matrix = confusion_matrix(y_true, labels, labels=[*FEET, "unknown"])[:2]
    return {
        "threshold": threshold,
        "coverage": float(covered.mean()) if len(labels) else 0.0,
        "abstention_rate": float(1 - covered.mean()) if len(labels) else 0.0,
        "abstained": int((~covered).sum()),
        "on_covered_steps": classification_metrics(y_true[covered], labels[covered]),
        "confusion_matrix_with_unknown": {"rows": FEET, "columns": [*FEET, "unknown"],
                                          "values": matrix.tolist()},
    }


def grouped_results(predictions: pd.DataFrame, column: str) -> dict:
    out = {}
    for key, part in predictions.groupby(column):
        y = part["foot_label"].to_numpy()
        covered = part["predicted_label"] != "unknown"
        out[str(key)] = {
            "steps": int(len(part)),
            "accuracy": float(accuracy_score(y, part["argmax_label"])),
            "macro_f1": float(f1_score(y, part["argmax_label"], labels=FEET, average="macro", zero_division=0)),
            "coverage": float(covered.mean()),
            "accuracy_on_covered": float(accuracy_score(y[covered], part.loc[covered, "predicted_label"]))
            if covered.any() else None,
        }
    return out


def limitations(bundle: ModelBundle, test_table: pd.DataFrame, predictions: pd.DataFrame,
                fingerprint_mismatches: list[str]) -> list[str]:
    meta = bundle.training_metadata
    split = meta["split"]
    notes = []
    if meta.get("synthetic_data"):
        notes.append("Trained and evaluated on SYNTHETIC data. These numbers only show that the pipeline runs; "
                     "they say nothing about real-world left/right classification.")
    n_test_groups = len(split["test"]["groups"])
    if n_test_groups < 5:
        notes.append(f"The test set has only {n_test_groups} {split['group_by']}(s); metrics may change a lot "
                     f"with different people. Report them with that caveat.")
    if split["group_by"] == "session":
        notes.append("Split by session: the same participant can appear in training and test, which usually "
                     "overstates performance on new people.")
    if bundle.config["features"]["feature_set"] == "offline":
        notes.append("Uses future-dependent features (time_until_next_step); not valid for real-time prediction.")
    if bundle.calibrator is None:
        notes.append("Probabilities are raw Random Forest vote fractions and are not calibrated; treat the "
                     "confidence threshold as a ranking cut-off rather than a true probability.")
    for side in ("train", "test"):
        sides = split[side]["pocket_sides"]
        if len(sides) < 2:
            notes.append(f"The {side} split only contains {sides or 'no'} pocket side(s).")
    counts = meta.get("step_status_counts", {})
    excluded = sum(v for k, v in counts.items() if k != "included")
    if excluded:
        notes.append(f"{excluded} annotated step(s) were excluded before training ({ {k: v for k, v in counts.items() if k != 'included'} }); "
                     "results only describe valid, reviewed left/right steps with complete windows.")
    left, right = (predictions["foot_label"] == "left").sum(), (predictions["foot_label"] == "right").sum()
    if min(left, right) < 0.4 * (left + right):
        notes.append(f"The test labels are imbalanced ({left} left / {right} right); prefer balanced accuracy.")
    notes.append("Steps are scored at annotated footfalls. Automatic step detection is a separate problem; "
                 "evaluate it with `python -m ml.src.predict ... --evaluate-against-labels`.")
    if fingerprint_mismatches:
        notes.append(f"Files changed since training for: {', '.join(fingerprint_mismatches)}.")
    return notes


def evaluate(model_path: str | Path, data_dir: Optional[str | Path] = None, reports_dir: str | Path = "ml/reports",
             threshold: Optional[float] = None) -> dict:
    bundle = ModelBundle.load(model_path)
    config = PipelineConfig.from_dict(bundle.config)
    meta = bundle.training_metadata
    data_dir = Path(data_dir or meta["data_dir"])
    threshold = bundle.confidence_threshold if threshold is None else threshold
    test_sessions = set(meta["split"]["test"]["sessions"])

    sessions = [s for s in load_sessions(data_dir, config) if s.session_id in test_sessions]
    missing = test_sessions - {s.session_id for s in sessions}
    if missing:
        raise FileNotFoundError(f"Held-out session(s) not found under {data_dir}: {sorted(missing)}")
    mismatches = [s.session_id for s in sessions
                  if meta["sessions"].get(s.session_id, {}).get("sha256") != session_fingerprint(s)]

    table, rejected = build_feature_table(sessions, config)
    if rejected:
        raise RuntimeError(f"Held-out session(s) now fail validation: {[s.session_id for s in rejected]}")
    test_rows = table[table["exclusion_reason"] == ""].reset_index(drop=True)
    if test_rows.empty:
        raise RuntimeError("No usable steps in the held-out sessions.")

    labels, confidence, abstained, p_right = bundle.predict(test_rows, threshold)
    predictions = test_rows[["session_id", "participant_id", "step_id", schema.TIME, "foot_label"]].copy()
    predictions["probability_right"] = p_right
    predictions["confidence"] = confidence
    predictions["argmax_label"] = np.where(p_right >= 0.5, "right", "left")
    predictions["predicted_label"] = labels
    predictions["abstained"] = abstained
    predictions["correct"] = predictions["argmax_label"] == predictions["foot_label"]

    y_true = predictions["foot_label"].to_numpy()
    y_binary = (y_true == "right").astype(int)
    macro_f1 = f1_score(y_true, predictions["argmax_label"], labels=FEET, average="macro", zero_division=0)
    report = {
        "model_path": str(model_path),
        "data_dir": str(data_dir),
        "synthetic_data": meta.get("synthetic_data", False),
        "dataset": {"split": meta["split"], "test_steps_scored": int(len(predictions)),
                    "test_step_status_counts": table["exclusion_reason"].replace("", "included").value_counts().to_dict()},
        "all_steps_no_abstention": classification_metrics(y_true, predictions["argmax_label"].to_numpy()),
        "with_confidence_threshold": thresholded_metrics(y_true, labels, threshold),
        "per_session": grouped_results(predictions, "session_id"),
        "per_participant": grouped_results(predictions, "participant_id"),
        "probability_quality": {
            "brier_score_test": brier(y_binary, p_right),
            "calibration": bundle.config["training"]["calibration"],
            "note": "Reported only. Calibration and thresholds are fitted on training groups, never on these labels.",
        },
        "development_target": {
            "metric": "macro_f1 (all held-out steps, no abstention)",
            "target": DEVELOPMENT_TARGET_MACRO_F1,
            "value": float(macro_f1),
            # Never report a target as met on synthetic data.
            "met": None if meta.get("synthetic_data") else bool(macro_f1 >= DEVELOPMENT_TARGET_MACRO_F1),
            "note": "A development target, not a claim of achieved performance."
                    + (" Not applicable: synthetic data." if meta.get("synthetic_data") else ""),
        },
        "limitations": limitations(bundle, table, predictions, mismatches),
    }

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    predictions.to_csv(reports_dir / "test_predictions.csv", index=False)
    plot_confusion(report, reports_dir / "confusion_matrix.png")
    return report


def plot_confusion(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    binary = np.array(report["all_steps_no_abstention"]["confusion_matrix"]["rows_true_cols_predicted"])
    with_unknown = np.array(report["with_confidence_threshold"]["confusion_matrix_with_unknown"]["values"])
    threshold = report["with_confidence_threshold"]["threshold"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [2, 3]})
    panels = [(axes[0], binary, FEET, "All steps (no abstention)"),
              (axes[1], with_unknown, [*FEET, "unknown"], f"Confidence threshold {threshold:.2f}")]
    for ax, matrix, columns, title in panels:
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        ax.set_xticks(range(len(columns)), columns)
        ax.set_yticks(range(2), FEET)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Annotated")
        ax.set_title(title)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > matrix.max() / 2 else "black", fontsize=12)
    if report.get("synthetic_data"):
        fig.suptitle("SYNTHETIC DATA: not a real-world result", color="firebrick")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_report(report: dict) -> None:
    overall = report["all_steps_no_abstention"]
    thresholded = report["with_confidence_threshold"]
    covered = thresholded["on_covered_steps"]
    print(f"Held-out steps: {overall['n']}")
    print(f"  accuracy {overall['accuracy']:.3f} | balanced accuracy {overall['balanced_accuracy']:.3f} | "
          f"macro F1 {overall['macro_f1']:.3f}")
    for foot, m in overall["per_foot"].items():
        print(f"  {foot:>5}: precision {m['precision']:.3f} recall {m['recall']:.3f} F1 {m['f1']:.3f} (n={m['support']})")
    print(f"  confusion (rows annotated, cols predicted {FEET}): {overall['confusion_matrix']['rows_true_cols_predicted']}")
    acc = covered.get("accuracy")
    print(f"Threshold {thresholded['threshold']:.2f}: coverage {thresholded['coverage']:.1%}, "
          f"abstention {thresholded['abstention_rate']:.1%}, accuracy on covered "
          f"{'n/a' if acc is None else f'{acc:.3f}'}")
    for name, values in report["per_session"].items():
        print(f"  session {name}: accuracy {values['accuracy']:.3f}, macro F1 {values['macro_f1']:.3f} (n={values['steps']})")
    target = report["development_target"]
    status = {True: "met", False: "not met", None: "not applicable (synthetic data)"}[target["met"]]
    print(f"Development target macro F1 >= {target['target']}: {status} ({target['value']:.3f}). "
          f"This is a target, not a claim.")
    print("Limitations:")
    for note in report["limitations"]:
        print(f"  - {note}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained PocketStep model on its held-out groups.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", help="Override the data directory recorded at training time")
    parser.add_argument("--reports-dir", default="ml/reports")
    parser.add_argument("--threshold", type=float,
                        help="Confidence threshold (default: the model's). Choose it from the training "
                             "out-of-fold sweep, not from test results.")
    args = parser.parse_args(argv)
    report = evaluate(args.model_path, args.data_dir, args.reports_dir, args.threshold)
    print_report(report)
    print(f"\nWrote {args.reports_dir}/metrics.json, confusion_matrix.png and test_predictions.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
