"""Train the baseline left/right step classifier with a grouped (leakage-free) split.

Usage:
    python -m ml.src.train --data-dir ml/data/raw
    python -m ml.src.train --data-dir ml/data/synthetic --model-path ml/models/synthetic.joblib

The test groups are only recorded here, never scored; run ml.src.evaluate for held-out metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict

from . import schema
from .config import PipelineConfig
from .features import build_feature_table, exclusion_summary, feature_names
from .models import ModelBundle, ProbabilityCalibrator, apply_threshold, build_estimator, class_mapping, encode_target
from .preprocessing import Session, load_sessions

DEFAULT_MODEL_PATH = "ml/models/pocketstep.joblib"
SPLIT_ATTEMPTS = 100


class TrainingDataError(RuntimeError):
    """The data cannot support a valid grouped train/test split."""


def group_column(group_by: str) -> str:
    if group_by not in ("participant", "session"):
        raise ValueError("group_by must be 'participant' or 'session'")
    return f"{group_by}_id"


def usable_rows(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["exclusion_reason"] == ""].reset_index(drop=True)


def grouped_split(rows: pd.DataFrame, group_by: str, test_fraction: float, min_groups: int,
                  seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Indices of train and test rows such that no group appears on both sides, both sides contain
    both feet, and the split is reproducible from `seed`. Returns (train_idx, test_idx, attempt).

    Raises TrainingDataError instead of falling back to a leakage-prone random split.
    """
    column = group_column(group_by)
    groups = rows[column].to_numpy()
    unique = sorted(set(groups))
    if len(unique) < min_groups:
        hint = ""
        if group_by == "participant" and rows["session_id"].nunique() >= min_groups:
            hint = (" You can split by session with --group-by session, but then one person's steps may be in "
                    "both training and test data, which tends to overstate performance on new people.")
        raise TrainingDataError(
            f"Only {len(unique)} {group_by}(s) have usable labeled steps ({', '.join(map(str, unique)) or 'none'}); "
            f"a grouped split needs at least {min_groups} so that the test set contains {group_by}s the model "
            f"never saw. Collect more data rather than splitting individual steps randomly, because neighbouring "
            f"steps are nearly identical and a random split would leak.{hint}")

    y = rows["foot_label"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=SPLIT_ATTEMPTS, test_size=test_fraction, random_state=seed)
    for attempt, (train_idx, test_idx) in enumerate(splitter.split(rows, y, groups)):
        if len(set(groups[train_idx])) < 2:
            continue
        if set(y[train_idx]) == {"left", "right"} and set(y[test_idx]) == {"left", "right"}:
            return train_idx, test_idx, attempt
    raise TrainingDataError(
        f"No grouped split out of {SPLIT_ATTEMPTS} attempts put both left and right steps on both sides. "
        f"Check the label balance per {group_by}.")


def session_fingerprint(session: Session) -> str:
    """SHA-256 over the session's three files, so reports can name the exact data used."""
    digest = hashlib.sha256()
    for name in (schema.SENSOR_FILE, schema.LABELS_FILE, schema.METADATA_FILE):
        path = session.path / name
        digest.update(name.encode())
        digest.update(path.read_bytes() if path.exists() else b"")
    return digest.hexdigest()


def threshold_sweep(y_true: np.ndarray, p_positive: np.ndarray,
                    thresholds=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9)) -> list[dict]:
    """Coverage and accuracy of confident predictions at each threshold (binary 0/1 classes; the
    mapping to left/right is one-to-one per step, so accuracy and coverage are unchanged by it)."""
    out = []
    for threshold in thresholds:
        labels, _, abstained = apply_threshold(p_positive, threshold)
        covered = ~abstained
        truth = np.where(y_true == 1, "right", "left")
        accuracy = float(np.mean(labels[covered] == truth[covered])) if covered.any() else None
        out.append({"threshold": threshold, "coverage": float(covered.mean()), "accuracy_on_covered": accuracy})
    return out


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def train(data_dir: str | Path, config: PipelineConfig, model_path: str | Path = DEFAULT_MODEL_PATH,
          log=print) -> ModelBundle:
    data_dir = Path(data_dir)
    sessions = load_sessions(data_dir, config)
    if not sessions:
        raise TrainingDataError(f"No sessions found under {data_dir}.")
    table, rejected = build_feature_table(sessions, config)
    for session in rejected:
        log(f"Rejected session (not used):\n{session.report.format()}")
    accepted = [s for s in sessions if s.report.ok]

    synthetic = {s.session_id for s in accepted if s.synthetic}
    if synthetic and len(synthetic) != len(accepted):
        raise TrainingDataError(
            "The data directory mixes synthetic and real sessions. Train on one kind only, so synthetic data "
            "can never inflate results reported for real recordings.")

    rows = usable_rows(table)
    exclusions = exclusion_summary(table)
    log(f"Steps: {len(table)} annotated, {len(rows)} usable for binary training. By status: {exclusions}")
    if rows.empty:
        raise TrainingDataError("No usable labeled steps (valid, reviewed, left/right, with a complete window).")

    tc = config.training
    train_idx, test_idx, attempt = grouped_split(rows, tc.group_by, tc.test_fraction, tc.min_groups, tc.random_seed)
    names = feature_names(config.features.feature_set)
    train_rows, test_rows = rows.iloc[train_idx], rows.iloc[test_idx]
    X_train = train_rows[names]
    y_train = encode_target(train_rows["foot_label"], train_rows["pocket_side"], config.model.target)
    groups_train = train_rows[group_column(tc.group_by)].to_numpy()

    estimator = build_estimator(config.model, tc.random_seed)
    estimator.fit(X_train, y_train)

    # Out-of-fold predictions on the training groups: used for calibration and threshold choice,
    # so neither ever looks at the held-out test labels.
    validation: dict = {}
    calibrator = None
    n_train_groups = len(set(groups_train))
    folds = min(tc.calibration_folds, n_train_groups)
    if folds >= 2:
        oof = cross_val_predict(clone(estimator), X_train, y_train, groups=groups_train,
                                cv=GroupKFold(n_splits=folds), method="predict_proba")[:, 1]
        validation = {"method": f"GroupKFold({folds}) out-of-fold on training groups",
                      "brier_raw": brier(y_train, oof), "threshold_sweep_raw": threshold_sweep(y_train, oof)}
        if tc.calibration != "none":
            calibrator = ProbabilityCalibrator(tc.calibration).fit(oof, y_train)
            calibrated = calibrator.transform(oof)
            validation["brier_calibrated_in_sample"] = brier(y_train, calibrated)
            validation["threshold_sweep_calibrated_in_sample"] = threshold_sweep(y_train, calibrated)
    elif tc.calibration != "none":
        raise TrainingDataError(f"Calibration needs at least 2 training {tc.group_by}s; found {n_train_groups}.")
    else:
        validation = {"note": f"Only {n_train_groups} training {tc.group_by}; no out-of-fold validation possible."}

    def split_summary(part: pd.DataFrame) -> dict:
        return {
            "groups": sorted(map(str, part[group_column(tc.group_by)].unique())),
            "sessions": sorted(map(str, part["session_id"].unique())),
            "participants": sorted(map(str, part["participant_id"].unique())),
            "steps": int(len(part)),
            "left": int((part["foot_label"] == "left").sum()),
            "right": int((part["foot_label"] == "right").sum()),
            "pocket_sides": sorted(map(str, part["pocket_side"].dropna().unique())),
        }

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "synthetic_data": bool(synthetic),
        "sessions": {s.session_id: {"participant_id": s.participant_id, "path": str(s.path),
                                    "sha256": session_fingerprint(s), "synthetic": s.synthetic} for s in accepted},
        "rejected_sessions": {s.report.session_id or str(s.path): s.report.codes("error") for s in rejected},
        "step_status_counts": exclusions,
        "split": {"group_by": tc.group_by, "test_fraction": tc.test_fraction, "seed": tc.random_seed,
                  "attempt": attempt, "train": split_summary(train_rows), "test": split_summary(test_rows)},
        "validation": validation,
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    bundle = ModelBundle(estimator=estimator, feature_names=names, config=config.to_dict(),
                         training_metadata=metadata, calibrator=calibrator,
                         label_mapping=class_mapping(config.model.target))
    bundle.save(model_path)
    return bundle


def print_summary(bundle: ModelBundle, model_path: str | Path) -> None:
    meta = bundle.training_metadata
    split = meta["split"]
    print(f"\nSaved model to {model_path}")
    print(f"Split by {split['group_by']} (seed {split['seed']}):")
    for side in ("train", "test"):
        part = split[side]
        print(f"  {side}: {len(part['groups'])} group(s) {part['groups']}, {part['steps']} steps "
              f"({part['left']} left / {part['right']} right)")
    sweep = meta["validation"].get("threshold_sweep_raw")
    if sweep:
        print(f"Out-of-fold validation on training groups (Brier {meta['validation']['brier_raw']:.3f}):")
        for row in sweep:
            acc = row["accuracy_on_covered"]
            print(f"  threshold {row['threshold']:.2f}: coverage {row['coverage']:.0%}, "
                  f"accuracy {'n/a' if acc is None else f'{acc:.3f}'}")
    if meta["synthetic_data"]:
        print("NOTE: trained on SYNTHETIC data. Results say nothing about real-world performance.")
    print(f"Next: python -m ml.src.evaluate --model-path {model_path}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the PocketStep left/right classifier.")
    parser.add_argument("--data-dir", required=True, help="Folder of exported session folders")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config", help="Pipeline config JSON; command-line options override it")
    parser.add_argument("--group-by", choices=["participant", "session"])
    parser.add_argument("--test-fraction", type=float)
    parser.add_argument("--min-groups", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--feature-set", choices=["live", "offline"])
    parser.add_argument("--target", choices=["pocket_relative", "absolute"])
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--min-samples-leaf", type=int)
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--calibration", choices=["none", "sigmoid", "isotonic"])
    parser.add_argument("--include-unreviewed", action="store_true")
    parser.add_argument("--save-config", help="Write the effective config to this JSON file")
    args = parser.parse_args(argv)

    config = PipelineConfig.load(args.config) if args.config else PipelineConfig()
    overrides = {
        ("training", "group_by"): args.group_by, ("training", "test_fraction"): args.test_fraction,
        ("training", "min_groups"): args.min_groups, ("training", "random_seed"): args.seed,
        ("features", "feature_set"): args.feature_set, ("model", "target"): args.target,
        ("model", "n_estimators"): args.n_estimators,
        ("model", "max_depth"): args.max_depth, ("model", "min_samples_leaf"): args.min_samples_leaf,
        ("training", "confidence_threshold"): args.confidence_threshold,
        ("training", "calibration"): args.calibration,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            setattr(getattr(config, section), key, value)
    if args.include_unreviewed:
        config.training.include_unreviewed = True
    if args.save_config:
        config.save(args.save_config)

    try:
        bundle = train(args.data_dir, config, args.model_path)
    except TrainingDataError as exc:
        print(f"Training stopped: {exc}", file=sys.stderr)
        return 2
    print_summary(bundle, args.model_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
