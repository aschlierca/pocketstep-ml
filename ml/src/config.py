"""Pipeline configuration. Every value is saved with a trained model so results can be reproduced."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class PreprocessingConfig:
    # An interval longer than gap_factor x the median interval is a gap. Samples are never invented.
    gap_factor: float = 3.0
    # Fraction of non-gap intervals allowed to deviate >25% from the median before warning.
    irregular_fraction_warning: float = 0.05
    # Cleaning (removing NaN rows / duplicate timestamps) may drop at most this fraction of rows;
    # beyond it the session is rejected instead of silently losing data.
    max_dropped_fraction: float = 0.02
    # Optional zero-phase Butterworth low-pass applied to the raw axes. None disables it.
    noise_filter_cutoff_hz: Optional[float] = None
    noise_filter_order: int = 4
    # Gravity is estimated per axis with this low-pass cutoff and subtracted to get linear acceleration.
    gravity_cutoff_hz: float = 0.5
    # Allowed |timestamp_seconds - elapsed_seconds| before the two columns are treated as disagreeing.
    time_column_tolerance: float = 1e-6


@dataclass
class SegmentationConfig:
    # Window around each footfall. 'after' is also the prediction latency for live use.
    window_before_seconds: float = 0.4
    window_after_seconds: float = 0.4
    # Windows with fewer than this fraction of the expected samples are rejected.
    min_sample_coverage: float = 0.8
    # Peak detection (exploration and inference only; never used to make ground truth).
    peak_smoothing_seconds: float = 0.05
    peak_min_height_g: float = 0.12
    peak_min_prominence_g: float = 0.1
    peak_min_distance_seconds: float = 0.3
    # Detected and annotated steps within this tolerance count as the same step.
    match_tolerance_seconds: float = 0.1


@dataclass
class FeatureConfig:
    # "live" uses only information available shortly after a footfall; "offline" adds features
    # that need future steps (time_until_next_step). Never use "offline" for a real-time model.
    feature_set: str = "live"
    # Prominence used when counting peaks inside a window.
    window_peak_prominence_g: float = 0.05
    # Gaps between consecutive steps longer than this are treated as "no previous/next step".
    max_step_interval_seconds: float = 2.0


@dataclass
class ModelConfig:
    model_type: str = "random_forest"
    # "pocket_relative": learn whether a step is by the leg carrying the phone, then convert to
    # left/right with the session's known pocket side (exact, since pocket side is recorded up front).
    # Without this, "left" means the phone's own leg in left-pocket sessions but the other leg in
    # right-pocket sessions, an XOR that greedy tree learners cannot pick up.
    # "absolute": learn left/right directly.
    target: str = "pocket_relative"
    n_estimators: int = 300
    max_depth: Optional[int] = None
    min_samples_leaf: int = 2
    max_features: str = "sqrt"
    class_weight: Optional[str] = "balanced"


@dataclass
class TrainingConfig:
    # "participant" (preferred) or "session". Splits never put steps of one group on both sides.
    group_by: str = "participant"
    test_fraction: float = 0.25
    min_groups: int = 3
    random_seed: int = 42
    # Assisted labels that nobody reviewed are excluded unless this is True.
    include_unreviewed: bool = False
    # Predictions with max class probability below this become "unknown".
    confidence_threshold: float = 0.6
    # Probability calibration fitted on out-of-fold predictions from the training groups only:
    # "none", "sigmoid" or "isotonic".
    calibration: str = "none"
    calibration_folds: int = 3


@dataclass
class ThresholdConfig:
    # The Unity app's step detector (unity/Plugins/iOS/PocketStepCounter.mm). These values must
    # match the plugin, or the calibrated threshold will not transfer.
    sample_rate_hz: float = 60.0
    # filtered = w * filtered + (1 - w) * |a|, once per sample, starting from filter_initial_g.
    filter_previous_weight: float = 0.6
    filter_initial_g: float = 1.0
    # A step is counted when filtered > threshold and more than this has passed since the last step.
    min_interval_seconds: float = 0.45
    # The threshold the app used before calibration; reported as a baseline.
    current_threshold_g: float = 1.45
    # Calibration walk: the first calibration_steps annotated steps of each session, plus a short tail.
    calibration_steps: int = 20
    calibration_tail_seconds: float = 0.5
    # Sessions need at least this many annotated steps after the calibration walk to be scored.
    min_evaluation_steps: int = 10
    # Thresholds searched, in g of total acceleration magnitude (gravity included, as in the app).
    grid_min_g: float = 1.02
    grid_max_g: float = 2.5
    grid_step_g: float = 0.01
    # Detected and annotated steps within this tolerance are the same step. Wider than the left/right
    # pipeline's because the app's filter delays detection by a frame or two.
    match_tolerance_seconds: float = 0.15
    # Every threshold within this F1 of the best counts as best; the median of them is used.
    best_f1_tolerance: float = 0.01
    ridge_alpha: float = 1.0
    min_participants: int = 3
    include_unreviewed: bool = False


@dataclass
class PipelineConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        return _build(cls, data)

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def _build(cls, data: dict[str, Any]):
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        default = getattr(cls(), name)
        kwargs[name] = _build(type(default), value) if is_dataclass(default) else value
    return cls(**kwargs)
