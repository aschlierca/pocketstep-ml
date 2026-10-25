import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.src import schema
from ml.src.synthetic import generate_dataset, simulate_session, write_session


def base_metadata(**overrides) -> dict:
    metadata = {
        "schema_version": "1.0",
        "participant_id": "P-TEST",
        "session_id": "S-TEST",
        "phone_model": "test",
        "ios_version": "18.0",
        "pocket_side": "right",
        "starting_foot": "left",
        "target_steps": 4,
        "walking_condition": "comfortable",
        "sampling_interval_seconds": 0.01,
        "session_start": "2026-01-01T00:00:00Z",
        "session_end": "2026-01-01T00:00:10Z",
        "notes": "",
        "synthetic": True,
    }
    metadata.update(overrides)
    return metadata


def resting_sensor(duration: float = 6.0, fs: float = 100.0, pulses=(), pulse_height: float = 0.5) -> pd.DataFrame:
    """Phone lying still (reads -1 g on z) with optional magnitude pulses at the given times."""
    t = np.round(np.arange(0, duration, 1 / fs), 6)
    bump = sum((pulse_height * np.exp(-0.5 * ((t - p) / 0.02) ** 2) for p in pulses), np.zeros_like(t))
    return pd.DataFrame({schema.TIME: t, schema.ELAPSED: t, "accel_x": 0.0, "accel_y": 0.0, "accel_z": -(1.0 + bump)})


def labels_frame(times, feet, valid=None, reviewed=None, source="manual") -> pd.DataFrame:
    n = len(times)
    return pd.DataFrame({
        "step_id": range(1, n + 1),
        schema.TIME: times,
        "foot_label": feet,
        "label_source": source,
        "is_valid": ["true" if v else "false" for v in (valid if valid is not None else [True] * n)],
        "reviewed": ["true" if v else "false" for v in (reviewed if reviewed is not None else [True] * n)],
        "notes": "",
    })


def write_dir(path: Path, sensor: pd.DataFrame = None, labels: pd.DataFrame = None, metadata: dict = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if sensor is not None:
        sensor.to_csv(path / schema.SENSOR_FILE, index=False)
    if labels is not None:
        labels.to_csv(path / schema.LABELS_FILE, index=False)
    if metadata is not None:
        (path / schema.METADATA_FILE).write_text(json.dumps(metadata))
    return path


@pytest.fixture
def synthetic_session_dir(tmp_path) -> Path:
    rng = np.random.default_rng(0)
    sensor, labels, metadata = simulate_session(
        rng, participant_id="SYN-P01", session_id="SYN-P01-S1", pocket_side="right",
        starting_foot="left", n_steps=20, walking_condition="comfortable")
    return write_session(tmp_path / "SYN-P01-S1", sensor, labels, metadata)


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synthetic")
    generate_dataset(root, participants=4, sessions_per_participant=2, steps=30, seed=3)
    return root
