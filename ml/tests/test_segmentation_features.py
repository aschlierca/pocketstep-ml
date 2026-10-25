import numpy as np
import pandas as pd
import pytest

from ml.src import schema
from ml.src.config import FeatureConfig, PipelineConfig, SegmentationConfig
from ml.src.features import FEATURE_SETS, FUTURE_FEATURES, build_feature_table, feature_names, session_step_table, window_features
from ml.src.preprocessing import load_session, load_sessions, preprocess
from ml.src.segmentation import (INSUFFICIENT_SAMPLES, OK, OUT_OF_RANGE, OVERLAPS_GAP, detect_candidate_steps,
                                 extract_window, label_exclusion_reason, match_steps)

from .conftest import base_metadata, labels_frame, resting_sensor, write_dir


def _processed(tmp_path, sensor, labels=None):
    session = load_session(write_dir(tmp_path, sensor, labels if labels is not None else labels_frame([1.0], ["left"]),
                                     base_metadata()))
    return session, preprocess(session)


def test_window_selects_samples_by_time(tmp_path):
    session, processed = _processed(tmp_path, resting_sensor())
    window = extract_window(processed, 2.0, session.median_interval, session.gaps, SegmentationConfig())
    assert window.status == OK
    assert len(window.data) == 81  # 1.6 s to 2.4 s inclusive at 100 Hz
    assert window.data[schema.TIME].min() == pytest.approx(1.6)
    assert window.data[schema.TIME].max() == pytest.approx(2.4)


def test_window_duration_is_configurable(tmp_path):
    session, processed = _processed(tmp_path, resting_sensor())
    config = SegmentationConfig(window_before_seconds=0.25, window_after_seconds=0.5)
    window = extract_window(processed, 2.0, session.median_interval, session.gaps, config)
    assert window.data[schema.TIME].min() == pytest.approx(1.75)
    assert window.data[schema.TIME].max() == pytest.approx(2.5)


def test_window_rejections(tmp_path):
    sensor = resting_sensor()
    gap = (sensor[schema.TIME] > 3.0) & (sensor[schema.TIME] < 3.3)
    sparse = (sensor[schema.TIME] > 4.6) & (sensor[schema.TIME] < 5.4) & (np.arange(len(sensor)) % 2 == 1)
    session, processed = _processed(tmp_path, sensor[~gap & ~sparse])
    config = SegmentationConfig()
    status = lambda c: extract_window(processed, c, session.median_interval, session.gaps, config).status
    assert status(0.2) == OUT_OF_RANGE
    assert status(5.9) == OUT_OF_RANGE
    assert status(3.1) == OVERLAPS_GAP
    # Every other sample missing: no interval is long enough to be a gap, but coverage is ~50%.
    assert status(5.0) == INSUFFICIENT_SAMPLES
    assert status(2.0) == OK


def test_irregular_timestamps_do_not_shift_windows(tmp_path):
    sensor = resting_sensor()
    rng = np.random.default_rng(1)
    jitter = rng.uniform(-0.002, 0.002, len(sensor))
    sensor[schema.TIME] = sensor[schema.TIME] + jitter
    sensor[schema.ELAPSED] = sensor[schema.TIME]
    session, processed = _processed(tmp_path, sensor)
    window = extract_window(processed, 2.0, session.median_interval, session.gaps, SegmentationConfig())
    assert window.status == OK
    assert window.data[schema.TIME].min() >= 1.6
    assert window.data[schema.TIME].max() <= 2.4


@pytest.mark.parametrize("foot, valid, reviewed, include, expected", [
    ("left", True, True, False, ""),
    ("right", False, True, False, "invalid_label"),
    ("unknown", True, True, False, "unknown_label"),
    ("left", True, False, False, "unreviewed_label"),
    ("left", True, False, True, ""),
    ("unknown", False, True, False, "invalid_label"),
])
def test_label_exclusion_reason(foot, valid, reviewed, include, expected):
    assert label_exclusion_reason(foot, valid, reviewed, include) == expected


def test_step_table_handles_invalid_unknown_and_unreviewed(tmp_path):
    times = [1.0, 1.5, 2.0, 2.5, 3.0, 5.8]
    labels = labels_frame(times, ["left", "right", "unknown", "right", "left", "right"],
                          valid=[True, False, True, True, True, True],
                          reviewed=[True, True, True, False, True, True])
    sensor = resting_sensor(pulses=times)
    session = load_session(write_dir(tmp_path, sensor, labels, base_metadata()))
    table = session_step_table(session, PipelineConfig())
    assert table["exclusion_reason"].tolist() == [
        "", "invalid_label", "unknown_label", "unreviewed_label", "", "window_out_of_range"]
    assert table.loc[table["window_status"] == OK, "magnitude_max"].notna().all()
    assert np.isnan(table.loc[5, "magnitude_max"])
    # Context features use every annotated step, including excluded ones, as neighbours.
    assert table["time_since_previous_step"].tolist()[:5] == pytest.approx([np.nan, 0.5, 0.5, 0.5, 0.5], nan_ok=True)
    assert np.isnan(table.loc[5, "time_since_previous_step"])  # 2.8 s exceeds max_step_interval_seconds


def test_window_features_on_known_signal():
    t = np.round(np.arange(1.6, 2.4001, 0.01), 6)
    magnitude = np.ones_like(t)
    magnitude[t == 2.05] = 1.5
    window = pd.DataFrame({
        schema.TIME: t,
        "accel_x": np.linspace(-1, 1, len(t)), "accel_y": 0.0, "accel_z": -magnitude,
        "magnitude": magnitude, "dynamic_magnitude": magnitude - 1,
        "vertical": magnitude - 1, "horizontal": 0.0,
    })
    f = window_features(window, center=2.0, config=FeatureConfig())
    assert f["accel_x_max"] == pytest.approx(1)
    assert f["accel_x_min"] == pytest.approx(-1)
    assert f["accel_x_mean"] == pytest.approx(0)
    assert f["accel_y_std"] == 0
    assert f["magnitude_max"] == pytest.approx(1.5)
    assert f["magnitude_peak_to_peak"] == pytest.approx(0.5)
    assert f["peak_prominence"] == pytest.approx(0.5)
    assert f["peak_time_offset"] == pytest.approx(0.05)
    assert f["n_peaks"] == 1
    assert f["signal_energy"] == pytest.approx(0.25 / len(t))
    assert f["vertical_max"] == pytest.approx(0.5)


def test_live_feature_set_excludes_future_features():
    assert set(FUTURE_FEATURES).isdisjoint(feature_names("live"))
    assert set(FUTURE_FEATURES) <= set(feature_names("offline"))
    with pytest.raises(ValueError):
        feature_names("everything")


def test_peak_detection_and_matching(tmp_path):
    pulses = [1.0, 1.6, 2.2, 2.8, 3.4, 4.0]
    session, processed = _processed(tmp_path, resting_sensor(pulses=pulses))
    detected = detect_candidate_steps(processed, session.median_interval, SegmentationConfig())
    assert detected == pytest.approx(pulses, abs=0.02)
    result = match_steps(detected, np.array(pulses), tolerance=0.1)
    assert result["precision"] == result["recall"] == 1.0


def test_match_steps_counts_misses_and_extras():
    result = match_steps(np.array([1.02, 2.5, 3.0]), np.array([1.0, 2.0, 3.05]), tolerance=0.1)
    assert (result["true_positives"], result["false_positives"], result["false_negatives"]) == (2, 1, 1)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["mean_abs_timing_error_seconds"] == pytest.approx(0.035)


def test_feature_table_over_synthetic_dataset(synthetic_dataset):
    table, rejected = build_feature_table(load_sessions(synthetic_dataset), PipelineConfig())
    assert not rejected
    assert table["session_id"].nunique() == 8
    included = table[table["exclusion_reason"] == ""]
    assert included[list(FEATURE_SETS["live"])].drop(columns=["time_since_previous_step"]).notna().all().all()
    assert set(included["foot_label"]) == {"left", "right"}
