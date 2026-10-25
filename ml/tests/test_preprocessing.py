import json

import numpy as np
import pandas as pd
import pytest

from ml.src import schema
from ml.src.config import PreprocessingConfig
from ml.src.preprocessing import (acceleration_magnitude, contiguous_segments, find_gaps, find_session_dirs,
                                  load_session, preprocess)

from .conftest import base_metadata, labels_frame, resting_sensor, write_dir


def test_synthetic_session_loads_cleanly(synthetic_session_dir):
    session = load_session(synthetic_session_dir)
    assert session.report.ok, session.report.format()
    assert session.synthetic
    assert session.report.stats["sampling_rate_hz"] == pytest.approx(100, rel=0.02)
    assert len(session.labels) == 20


def test_raw_files_are_not_modified(synthetic_session_dir):
    before = {p.name: p.read_bytes() for p in synthetic_session_dir.iterdir()}
    load_session(synthetic_session_dir)
    assert before == {p.name: p.read_bytes() for p in synthetic_session_dir.iterdir()}


def test_missing_sensor_column_is_an_error(tmp_path):
    sensor = resting_sensor().drop(columns=["accel_y"])
    session = load_session(write_dir(tmp_path, sensor, labels_frame([1.0], ["left"]), base_metadata()))
    assert not session.report.ok
    assert "sensor_columns" in session.report.codes("error")


def test_non_numeric_rows_are_counted_and_removed(tmp_path):
    sensor = resting_sensor().astype({"accel_x": object})
    sensor.loc[10, "accel_x"] = "oops"
    session = load_session(write_dir(tmp_path, sensor, labels_frame([1.0], ["left"]), base_metadata()))
    assert session.report.ok
    assert "missing_values" in session.report.codes("warning")
    assert session.report.stats["dropped_rows"] == 1
    assert len(session.sensor) == len(sensor) - 1


def test_dropping_too_many_rows_rejects_session(tmp_path):
    sensor = resting_sensor()
    sensor.loc[sensor.index[::10], "accel_z"] = np.nan  # 10% missing
    session = load_session(write_dir(tmp_path, sensor, labels_frame([1.0], ["left"]), base_metadata()))
    assert "too_many_dropped_rows" in session.report.codes("error")


def test_out_of_order_and_duplicate_timestamps(tmp_path):
    sensor = resting_sensor()
    sensor.iloc[[100, 101]] = sensor.iloc[[101, 100]].to_numpy()
    sensor = pd.concat([sensor, sensor.iloc[[200]]], ignore_index=True)
    session = load_session(write_dir(tmp_path, sensor, labels_frame([1.0], ["left"]), base_metadata()))
    codes = session.report.codes("warning")
    assert "timestamp_order" in codes
    assert "duplicate_timestamps" in codes
    assert np.all(np.diff(session.sensor[schema.TIME]) > 0)


def test_disagreeing_time_columns_are_an_error(tmp_path):
    sensor = resting_sensor()
    sensor[schema.ELAPSED] = sensor[schema.ELAPSED] + 0.5
    session = load_session(write_dir(tmp_path, sensor, labels_frame([1.0], ["left"]), base_metadata()))
    assert "time_columns_disagree" in session.report.codes("error")


def test_gaps_are_reported_not_filled(tmp_path):
    sensor = resting_sensor()
    sensor = sensor[(sensor[schema.TIME] < 2.0) | (sensor[schema.TIME] > 2.5)]
    session = load_session(write_dir(tmp_path, sensor, labels_frame([1.0], ["left"]), base_metadata()))
    assert "sensor_gaps" in session.report.codes("warning")
    assert len(session.gaps) == 1
    assert session.gaps[0][0] == pytest.approx(1.99)
    assert session.gaps[0][1] == pytest.approx(2.51)
    assert len(session.sensor) == len(sensor)


def test_find_gaps_and_segments():
    t = np.array([0.0, 0.01, 0.02, 0.03, 0.5, 0.51, 0.52])
    gaps, median_dt = find_gaps(t, gap_factor=3)
    assert median_dt == pytest.approx(0.01)
    assert gaps == [(0.03, 0.5)]
    assert contiguous_segments(t, gaps) == [slice(0, 4), slice(4, 7)]


def test_values_in_m_per_s2_are_rejected(tmp_path):
    sensor = resting_sensor()
    sensor[list(schema.AXES)] *= 9.81
    session = load_session(write_dir(tmp_path, sensor, labels_frame([1.0], ["left"]), base_metadata()))
    assert "units" in session.report.codes("error")


def test_acceleration_magnitude():
    df = pd.DataFrame({"accel_x": [3.0, 0.0], "accel_y": [4.0, 0.0], "accel_z": [0.0, -1.0]})
    assert acceleration_magnitude(df) == pytest.approx([5.0, 1.0])


def test_label_validation_errors(tmp_path):
    labels = labels_frame([1.0, 2.0], ["left", "middle"])
    labels.loc[1, "is_valid"] = "maybe"
    session = load_session(write_dir(tmp_path, resting_sensor(), labels, base_metadata()))
    messages = " ".join(i.message for i in session.report.issues if i.level == "error")
    assert "foot_label outside" in messages
    assert "is_valid that is not true/false" in messages


def test_labels_without_reviewed_column_treat_assisted_as_unreviewed(tmp_path):
    labels = labels_frame([1.0, 2.0], ["left", "right"]).drop(columns=["reviewed"])
    labels.loc[1, "label_source"] = "assisted"
    session = load_session(write_dir(tmp_path, resting_sensor(), labels, base_metadata()))
    assert session.report.ok
    assert "labels_no_reviewed_column" in session.report.codes("warning")
    assert session.labels["reviewed"].tolist() == [True, False]


def test_missing_labels_file_is_a_warning(tmp_path):
    session = load_session(write_dir(tmp_path, resting_sensor(), None, base_metadata()))
    assert session.report.ok
    assert "labels_missing" in session.report.codes("warning")
    assert session.labels.empty


def test_metadata_problems(tmp_path):
    session = load_session(write_dir(tmp_path / "a", resting_sensor(), labels_frame([1.0], ["left"]), None))
    assert "metadata_missing" in session.report.codes("error")
    session = load_session(write_dir(tmp_path / "b", resting_sensor(), labels_frame([1.0], ["left"]),
                                     base_metadata(schema_version="9.0")))
    assert "schema_version" in session.report.codes("error")


def test_find_session_dirs(tmp_path, synthetic_session_dir):
    (tmp_path / "not_a_session").mkdir()
    assert find_session_dirs(tmp_path) == [synthetic_session_dir]


def test_preprocess_separates_gravity_from_motion(tmp_path):
    sensor = resting_sensor(duration=8, pulses=[4.0], pulse_height=0.5)
    session = load_session(write_dir(tmp_path, sensor, labels_frame([4.0], ["left"]), base_metadata()))
    out = preprocess(session)
    quiet = out[out[schema.TIME].between(1.0, 2.0)]
    assert quiet["magnitude_raw"].mean() == pytest.approx(1.0, abs=1e-6)
    assert quiet["dynamic_magnitude"].abs().max() < 0.01
    peak = out.loc[out["magnitude"].idxmax()]
    assert peak[schema.TIME] == pytest.approx(4.0, abs=0.011)
    # The pulse pushes along gravity, so the gravity-aligned component carries it, not the horizontal one.
    assert peak["vertical"] > 0.3
    assert peak["horizontal"] < 0.05
    assert peak["dynamic_magnitude"] > 0.3


def test_filters_do_not_cross_gaps(tmp_path):
    sensor = resting_sensor(duration=8)
    sensor.loc[sensor[schema.TIME] >= 4.0, "accel_z"] = -2.0
    sensor = sensor[(sensor[schema.TIME] < 3.5) | (sensor[schema.TIME] >= 4.0)]
    session = load_session(write_dir(tmp_path, sensor, labels_frame([1.0], ["left"]), base_metadata()))
    out = preprocess(session)
    before = out[out[schema.TIME] < 3.5]
    after = out[out[schema.TIME] >= 4.0]
    assert before["gravity_z"].to_numpy() == pytest.approx(-1.0, abs=1e-6)
    assert after["gravity_z"].to_numpy() == pytest.approx(-2.0, abs=1e-6)
    assert sorted(out["segment"].unique()) == [0, 1]


def test_noise_filter_rejects_cutoff_above_nyquist(synthetic_session_dir):
    session = load_session(synthetic_session_dir)
    with pytest.raises(ValueError):
        preprocess(session, PreprocessingConfig(noise_filter_cutoff_hz=60))
    filtered = preprocess(session, PreprocessingConfig(noise_filter_cutoff_hz=15))
    assert filtered["accel_x"].std() < preprocess(session)["accel_x"].std()
    assert filtered["magnitude_raw"].equals(preprocess(session)["magnitude_raw"])
