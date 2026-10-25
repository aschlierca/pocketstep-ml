import json
import shutil

import numpy as np
import pandas as pd
import pytest

from ml.src import schema
from ml.src.config import PipelineConfig
from ml.src.evaluate import evaluate
from ml.src.features import build_feature_table
from ml.src.models import ModelBundle, apply_threshold, encode_target
from ml.src.predict import PREDICTION_COLUMNS, evaluate_against_labels, load_recording, main as predict_main, predict_recording
from ml.src.preprocessing import load_sessions
from ml.src.train import TrainingDataError, grouped_split, train, usable_rows

from .conftest import base_metadata, labels_frame, resting_sensor, write_dir


@pytest.fixture(scope="module")
def trained(synthetic_dataset, tmp_path_factory):
    model_path = tmp_path_factory.mktemp("model") / "model.joblib"
    config = PipelineConfig()
    config.model.n_estimators = 60
    bundle = train(synthetic_dataset, config, model_path, log=lambda *_: None)
    return bundle, model_path


def _usable(dataset):
    table, _ = build_feature_table(load_sessions(dataset), PipelineConfig())
    return usable_rows(table)


def test_grouped_split_never_shares_groups(synthetic_dataset):
    rows = _usable(synthetic_dataset)
    for group_by in ("participant", "session"):
        train_idx, test_idx, _ = grouped_split(rows, group_by, 0.25, 3, seed=1)
        column = f"{group_by}_id"
        assert set(rows.iloc[train_idx][column]).isdisjoint(rows.iloc[test_idx][column])
        assert set(rows.iloc[test_idx]["foot_label"]) == {"left", "right"}


def test_grouped_split_is_reproducible(synthetic_dataset):
    rows = _usable(synthetic_dataset)
    a = grouped_split(rows, "participant", 0.25, 3, seed=5)
    b = grouped_split(rows, "participant", 0.25, 3, seed=5)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_too_few_groups_stops_with_explanation(synthetic_dataset):
    rows = _usable(synthetic_dataset)
    two_people = rows[rows["participant_id"].isin(["SYN-P01", "SYN-P02"])]
    with pytest.raises(TrainingDataError, match="--group-by session"):
        grouped_split(two_people, "participant", 0.25, 3, seed=0)
    with pytest.raises(TrainingDataError, match="at least 3"):
        grouped_split(two_people[two_people["session_id"] == "SYN-P01-S1"], "session", 0.25, 3, seed=0)


def test_training_records_split_and_exclusions(trained):
    bundle, _ = trained
    meta = bundle.training_metadata
    split = meta["split"]
    assert set(split["train"]["groups"]).isdisjoint(split["test"]["groups"])
    assert meta["synthetic_data"] is True
    assert "included" in meta["step_status_counts"]
    assert {"invalid_label", "unknown_label"} & set(meta["step_status_counts"])
    assert all(len(info["sha256"]) == 64 for info in meta["sessions"].values())
    assert bundle.feature_names and "time_until_next_step" not in bundle.feature_names


def test_mixed_synthetic_and_real_data_is_refused(synthetic_dataset, tmp_path):
    shutil.copytree(synthetic_dataset, tmp_path / "data")
    times = [1.0, 1.5, 2.0, 2.5]
    write_dir(tmp_path / "data" / "real", resting_sensor(pulses=times), labels_frame(times, ["left", "right"] * 2),
              base_metadata(synthetic=False, session_id="REAL-1"))
    with pytest.raises(TrainingDataError, match="mixes synthetic and real"):
        train(tmp_path / "data", PipelineConfig(), tmp_path / "m.joblib", log=lambda *_: None)


def test_model_save_load_consistency(trained, synthetic_dataset):
    bundle, path = trained
    loaded = ModelBundle.load(path)
    rows = _usable(synthetic_dataset)
    assert loaded.feature_names == bundle.feature_names
    assert loaded.config == bundle.config
    assert loaded.label_mapping == {"other_leg": 0, "pocket_side_leg": 1}
    np.testing.assert_allclose(loaded.probability_right(rows), bundle.probability_right(rows))


def test_threshold_behaviour():
    labels, confidence, abstained = apply_threshold(np.array([0.95, 0.55, 0.45, 0.1]), 0.6)
    assert labels.tolist() == ["right", "unknown", "unknown", "left"]
    assert confidence == pytest.approx([0.95, 0.55, 0.55, 0.9])
    assert abstained.tolist() == [False, True, True, False]
    labels, _, abstained = apply_threshold(np.array([0.51, 0.49]), 0.5)
    assert labels.tolist() == ["right", "left"] and not abstained.any()
    with pytest.raises(ValueError):
        apply_threshold(np.array([0.5]), 0.4)


def test_pocket_relative_target_round_trip():
    feet = np.array(["left", "right", "left", "right"])
    pockets = np.array(["left", "left", "right", "right"])
    assert encode_target(feet, pockets, "pocket_relative").tolist() == [1, 0, 0, 1]
    assert encode_target(feet, pockets, "absolute").tolist() == [0, 1, 0, 1]


def test_evaluate_writes_reports(trained, tmp_path):
    _, model_path = trained
    report = evaluate(model_path, reports_dir=tmp_path)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "confusion_matrix.png").stat().st_size > 0
    saved = json.loads((tmp_path / "metrics.json").read_text())
    overall = saved["all_steps_no_abstention"]
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "per_foot", "confusion_matrix"):
        assert key in overall
    assert set(saved["per_session"]) == set(report["dataset"]["split"]["test"]["sessions"])
    assert saved["per_participant"]
    thresholded = saved["with_confidence_threshold"]
    assert thresholded["coverage"] + thresholded["abstention_rate"] == pytest.approx(1)
    assert saved["development_target"]["met"] is None  # never "met" on synthetic data
    assert any("SYNTHETIC" in note for note in saved["limitations"])
    predictions = pd.read_csv(tmp_path / "test_predictions.csv")
    assert set(predictions["session_id"]) == set(report["dataset"]["split"]["test"]["sessions"])


def test_prediction_ignores_labels(trained, synthetic_dataset, tmp_path):
    bundle, _ = trained
    session_dir = shutil.copytree(synthetic_dataset / "SYN-P02-S1", tmp_path / "s")
    first = predict_recording(bundle, load_recording(session_dir, PipelineConfig.from_dict(bundle.config)))
    # Scrambling the labels must not change any prediction.
    labels = pd.read_csv(session_dir / schema.LABELS_FILE)
    labels["foot_label"] = "left"
    labels.to_csv(session_dir / schema.LABELS_FILE, index=False)
    second = predict_recording(bundle, load_recording(session_dir, PipelineConfig.from_dict(bundle.config)))
    pd.testing.assert_frame_equal(first, second)


def test_prediction_csv_output(trained, synthetic_dataset, tmp_path):
    _, model_path = trained
    output = tmp_path / "pred.csv"
    code = predict_main(["--model-path", str(model_path), "--input", str(synthetic_dataset / "SYN-P02-S1" / schema.SENSOR_FILE),
                         "--output", str(output), "--threshold", "0.7", "--evaluate-against-labels"])
    assert code == 0
    predictions = pd.read_csv(output)
    assert list(predictions.columns) == PREDICTION_COLUMNS
    assert len(predictions) > 0
    assert set(predictions["predicted_label"]) <= {"left", "right", "unknown"}
    scored = predictions[predictions["window_status"] == "ok"]
    assert (scored["abstained"] == (scored["confidence"] < 0.7)).all()
    assert (predictions.loc[predictions["abstained"], "predicted_label"] == "unknown").all()
    assert scored["probability_left"].add(scored["probability_right"]).to_numpy() == pytest.approx(1)
    evaluation = json.loads(output.with_suffix(".evaluation.json").read_text())
    assert evaluation["step_detection"]["recall"] > 0.9
    assert "classification_on_detected_steps" in evaluation


def test_prediction_requires_pocket_side(trained, tmp_path):
    _, model_path = trained
    write_dir(tmp_path, resting_sensor())
    assert predict_main(["--model-path", str(model_path), "--input", str(tmp_path / schema.SENSOR_FILE),
                         "--output", str(tmp_path / "p.csv")]) == 2
    assert predict_main(["--model-path", str(model_path), "--input", str(tmp_path / schema.SENSOR_FILE),
                         "--output", str(tmp_path / "p.csv"), "--pocket-side", "left"]) == 0


def test_evaluate_against_labels_separates_detection_and_classification():
    predictions = pd.DataFrame({schema.TIME: [1.0, 2.0, 3.5], "predicted_label": ["left", "unknown", "left"]})
    labels = labels_frame([1.02, 2.0, 3.0], ["left", "right", "left"])
    labels["is_valid"] = True
    labels["reviewed"] = True
    result = evaluate_against_labels(predictions, labels, tolerance=0.1)
    assert result["step_detection"]["true_positives"] == 2
    assert result["step_detection"]["false_negatives"] == 1
    c = result["classification_on_detected_steps"]
    assert c["matched_steps_with_usable_labels"] == 2
    assert c["coverage"] == 0.5
    assert c["accuracy_on_covered"] == 1.0
