import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.src import threshold as th
from ml.src.config import PipelineConfig, ThresholdConfig
from ml.src.synthetic import generate_dataset
from ml.src.train import TrainingDataError


def plugin_loop(times, x, y, z, threshold, cfg):
    """The Unity plugin's handler, transcribed sample by sample."""
    filtered, last, count, values = cfg.filter_initial_g, -math.inf, 0, []
    for t, a, b, c in zip(times, x, y, z):
        filtered = cfg.filter_previous_weight * filtered + (1.0 - cfg.filter_previous_weight) * math.sqrt(a * a + b * b + c * c)
        values.append(filtered)
        if filtered > threshold and (t - last) > cfg.min_interval_seconds:
            count += 1
            last = t
    return np.array(values), count


@pytest.fixture(scope="module")
def varied_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("varied")
    generate_dataset(root, participants=5, sessions_per_participant=2, steps=45, seed=5, varied_walkers=True)
    return root


def test_detector_matches_the_plugin_loop():
    cfg = ThresholdConfig()
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.uniform(0.012, 0.022, 3000))
    x, y = rng.normal(0, 0.3, 3000), rng.normal(0, 0.3, 3000)
    z = 1 + rng.normal(0, 0.4, 3000)
    filtered = th.filtered_magnitude(x, y, z, cfg)
    for threshold in (1.05, 1.3, 1.45, 1.8, 2.4):
        expected_filtered, expected_count = plugin_loop(t, x, y, z, threshold, cfg)
        assert np.array_equal(filtered, expected_filtered)
        assert len(th.detect_steps(t, filtered, threshold, cfg.min_interval_seconds)) == expected_count


def test_resample_uses_app_rate_and_never_fills_gaps():
    t = np.concatenate([np.arange(0, 2, 0.01), np.arange(3, 4, 0.01)])
    sensor = pd.DataFrame({"timestamp_seconds": t, "accel_x": t, "accel_y": 0.0, "accel_z": 1.0})
    rt, rx, _, _ = th.resample(sensor, [(1.99, 3.0)], 60.0)
    assert np.allclose(np.diff(rt[rt < 1.99]), 1 / 60)
    assert not ((rt > 1.99) & (rt < 3.0)).any()
    assert np.allclose(rx, rt)


def test_calibration_peaks_are_spaced_and_sorted():
    times = np.arange(0, 10, 1 / 60)
    filtered = np.ones_like(times)
    for k, center in enumerate(np.arange(1, 9, 0.6)):
        filtered += (0.3 + 0.02 * k) * np.exp(-0.5 * ((times - center) / 0.03) ** 2)
        filtered += 0.25 * np.exp(-0.5 * ((times - center - 0.1) / 0.03) ** 2)  # a bounce inside the cooldown
    heights = th.calibration_peaks(times, filtered, 5, 0.45)
    assert heights == sorted(heights, reverse=True) and len(heights) == 5
    assert heights[-1] > 1.3  # bounces never displace footfalls


def test_count_match_finds_threshold_between_noise_and_steps():
    cfg = ThresholdConfig()
    times = np.arange(0, 12, 1 / 60)
    filtered = np.ones_like(times)
    for center in np.arange(1, 11, 0.5):  # 20 footfalls of 1.6 g, bounces of 1.3 g in between
        filtered += 0.6 * np.exp(-0.5 * ((times - center) / 0.03) ** 2)
        filtered += 0.3 * np.exp(-0.5 * ((times - center - 0.25) / 0.05) ** 2)
    value = th.count_match_threshold(times, filtered, 20, cfg)
    assert 1.3 < value < 1.6
    assert len(th.detect_steps(times, filtered, value, cfg.min_interval_seconds)) == 20


def test_training_writes_model_and_beats_built_in_threshold_on_varied_walkers(varied_dataset, tmp_path):
    out = tmp_path / "threshold_model.json"
    report = th.train_threshold_model(varied_dataset, PipelineConfig(), out, tmp_path / "reports", log=lambda *_: None)
    model = json.loads(out.read_text())
    assert model["format"] == th.MODEL_FORMAT and model["recommended_method"] in th.SELECTABLE
    assert model["model"]["features"] == list(th.FEATURES)
    for name in ("threshold_metrics.json", "threshold_cv_results.csv", "threshold_sessions.csv"):
        assert (tmp_path / "reports" / name).exists()
    s = report["summary"]
    assert s["oracle"]["mean_f1"] >= max(s[m]["mean_f1"] for m in th.METHODS)
    assert s[report["recommended_method"]]["mean_f1"] > s["current"]["mean_f1"]
    assert report["limitations"][0].startswith("SYNTHETIC DATA")


def test_cross_validation_never_sees_held_out_evaluation_labels(varied_dataset):
    cfg = ThresholdConfig()
    prepared = [th.prepare_session(s, cfg)[0] for s in th.load_sessions(varied_dataset)]
    held_out = prepared[0].participant_id
    before = th.cross_validate(prepared, cfg)
    for s in prepared:
        if s.participant_id == held_out:  # pretend this participant's later steps look completely different
            s.grid_f1 = np.roll(s.grid_f1, 40)
            s.oracle_g = 2.4
    after = th.cross_validate(prepared, cfg)
    pick = lambda rows: rows[(rows.participant_id == held_out) & rows.method.isin(["fixed", "count_match", "model"])]
    assert pick(before)["threshold_g"].tolist() == pick(after)["threshold_g"].tolist()


def test_too_few_participants_is_an_error(varied_dataset, tmp_path):
    config = PipelineConfig()
    config.threshold.min_participants = 6
    with pytest.raises(TrainingDataError, match="Only 5 participant"):
        th.train_threshold_model(varied_dataset, config, tmp_path / "m.json", tmp_path / "r", log=lambda *_: None)


def test_short_sessions_are_skipped_with_a_reason(synthetic_dataset):
    session = th.load_sessions(synthetic_dataset)[0]
    cs, reason = th.prepare_session(session, ThresholdConfig(calibration_steps=40))
    assert cs is None and "needs 50" in reason


def test_calibrate_recording_applies_recommended_method(varied_dataset, tmp_path):
    out = tmp_path / "model.json"
    th.train_threshold_model(varied_dataset, PipelineConfig(), out, tmp_path / "r", log=lambda *_: None)
    model = json.loads(out.read_text())
    session_dir = sorted(p for p in varied_dataset.iterdir() if p.is_dir())[0]
    for method in th.SELECTABLE:
        value, features = th.calibrate_recording({**model, "recommended_method": method}, session_dir, 45)
        assert model["calibration"]["grid_min_g"] <= value <= model["calibration"]["grid_max_g"]
        assert set(features) == set(th.FEATURES)
    fixed, _ = th.calibrate_recording({**model, "recommended_method": "fixed"}, session_dir, 45)
    assert fixed == model["fixed_threshold_g"]


UNITY = Path(__file__).resolve().parents[2] / "unity"


def test_plugin_fixture_is_up_to_date():
    committed = json.loads((UNITY / "Tests" / "golden.json").read_text())
    assert committed == json.loads(json.dumps(th.golden_fixture())), \
        "Regenerate with: python -m ml.src.threshold golden"


@pytest.mark.skipif(sys.platform != "darwin" or shutil.which("xcrun") is None, reason="needs macOS with Xcode")
def test_unity_plugin_matches_python():
    result = subprocess.run([str(UNITY / "Tests" / "run_plugin_tests.sh")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
