"""Model construction, probability calibration, abstention and persistence.

Models are looked up by name in MODEL_BUILDERS. The Random Forest baseline consumes the per-step
feature table. A future 1D CNN would register its own builder and consume raw step windows
(segmentation.extract_windows) instead of engineered features; ModelBundle only requires
`predict_proba`-style output for the right foot, so train/evaluate/predict stay unchanged apart
from how inputs are assembled.

Model files are joblib pickles: only load files you created or trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import ModelConfig

BUNDLE_FORMAT_VERSION = 1
TARGETS = ("pocket_relative", "absolute")


def encode_target(foot_labels, pocket_sides, target: str) -> np.ndarray:
    """1/0 class for each left/right step: 1 = right ("absolute") or 1 = pocket-side leg ("pocket_relative")."""
    foot_labels = np.asarray(foot_labels)
    if target == "absolute":
        return (foot_labels == "right").astype(int)
    if target == "pocket_relative":
        return (foot_labels == np.asarray(pocket_sides)).astype(int)
    raise ValueError(f"Unknown target {target!r}; choose from {TARGETS}")


def class_mapping(target: str) -> dict:
    if target == "absolute":
        return {"left": 0, "right": 1}
    return {"other_leg": 0, "pocket_side_leg": 1}


def _random_forest(config: ModelConfig, seed: int) -> Pipeline:
    # Missing context features (e.g. no previous step) are imputed with medians learned from the
    # training split only, because the imputer is fitted inside the pipeline.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            max_features=config.max_features,
            class_weight=config.class_weight,
            random_state=seed,
            n_jobs=-1,
        )),
    ])


MODEL_BUILDERS: dict[str, Callable[[ModelConfig, int], Pipeline]] = {
    "random_forest": _random_forest,
}


def build_estimator(config: ModelConfig, seed: int):
    if config.model_type not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model_type {config.model_type!r}; available: {sorted(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[config.model_type](config, seed)


class ProbabilityCalibrator:
    """Maps raw P(right) to calibrated P(right). Fit it on out-of-fold predictions from the training
    groups, never on held-out test labels."""

    def __init__(self, method: str):
        if method not in ("sigmoid", "isotonic"):
            raise ValueError(f"Unknown calibration method {method!r}")
        self.method = method
        self._model = None

    def fit(self, p_right: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        if self.method == "isotonic":
            self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_right, y)
        else:
            self._model = LogisticRegression().fit(_logit(p_right)[:, None], y)
        return self

    def transform(self, p_right: np.ndarray) -> np.ndarray:
        if self.method == "isotonic":
            return self._model.predict(p_right)
        return self._model.predict_proba(_logit(p_right)[:, None])[:, 1]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def apply_threshold(p_right: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (labels, confidence, abstained). Confidence is the larger class probability; below
    `threshold` the label is "unknown". A threshold of 0.5 never abstains."""
    if not 0.5 <= threshold <= 1.0:
        raise ValueError("confidence threshold must be between 0.5 and 1.0")
    p_right = np.asarray(p_right, dtype=float)
    confidence = np.maximum(p_right, 1 - p_right)
    labels = np.where(p_right >= 0.5, "right", "left").astype(object)
    abstained = confidence < threshold
    labels[abstained] = "unknown"
    return labels, confidence, abstained


@dataclass
class ModelBundle:
    estimator: object
    feature_names: list[str]
    config: dict
    training_metadata: dict
    calibrator: Optional[ProbabilityCalibrator] = None
    label_mapping: dict = field(default_factory=dict)
    format_version: int = BUNDLE_FORMAT_VERSION

    @property
    def confidence_threshold(self) -> float:
        return float(self.config["training"]["confidence_threshold"])

    @property
    def target(self) -> str:
        return self.config["model"]["target"]

    def raw_probability_positive(self, features: pd.DataFrame) -> np.ndarray:
        """Uncalibrated probability of class 1 (see class_mapping)."""
        X = features[self.feature_names]
        return self.estimator.predict_proba(X)[:, list(self.estimator.classes_).index(1)]

    def probability_right(self, features: pd.DataFrame) -> np.ndarray:
        p = self.raw_probability_positive(features)
        if self.calibrator is not None:
            p = self.calibrator.transform(p)
        if self.target == "pocket_relative":
            # P(pocket-side leg) -> P(right) using each step's recorded pocket side.
            pocket_left = features["pocket_side_is_left"].to_numpy() == 1.0
            p = np.where(pocket_left, 1 - p, p)
        return p

    def predict(self, features: pd.DataFrame, threshold: Optional[float] = None):
        """(labels, confidence, abstained, P(right)) for each row."""
        p = self.probability_right(features)
        labels, confidence, abstained = apply_threshold(p, self.confidence_threshold if threshold is None else threshold)
        return labels, confidence, abstained, p

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "ModelBundle":
        bundle = joblib.load(path)
        if not isinstance(bundle, ModelBundle):
            raise TypeError(f"{path} does not contain a PocketStep model bundle")
        if bundle.format_version != BUNDLE_FORMAT_VERSION:
            raise ValueError(f"{path} uses bundle format {bundle.format_version}; expected {BUNDLE_FORMAT_VERSION}")
        return bundle
