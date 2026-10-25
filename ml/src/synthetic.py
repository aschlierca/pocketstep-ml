"""Synthetic sessions for developing and testing the pipeline.

The signals are a simple physical caricature of walking in place with a phone in a trouser pocket:
an impact pulse at each footfall, lateral sway toward the stance foot, and a thigh lift (which
rotates the phone relative to gravity) before steps of the pocket-side leg. Per-session phone
orientation is randomised.

Synthetic data is NOT a participant recording and must never be used to claim real-world
performance. Every generated session has "synthetic": true in its metadata, participant IDs start
with "SYN-", and the training/evaluation reports flag it.

Usage:
    python -m ml.src.synthetic --output-dir ml/data/synthetic --participants 4 --sessions-per-participant 2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import schema

CADENCE_HZ = {"slow": 1.4, "comfortable": 1.8, "brisk": 2.2}
STILL_SECONDS = 2.0


def _gauss(t: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - center) / width) ** 2)


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def simulate_session(rng: np.random.Generator, *, participant_id: str, session_id: str, pocket_side: str,
                     starting_foot: str, n_steps: int, walking_condition: str, fs: float = 100.0,
                     unexpected_gap: bool = False, unknown_fraction: float = 0.03,
                     invalid_fraction: float = 0.03, assisted: bool = False,
                     traits: Optional[dict] = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (sensor_data, step_labels, metadata) in the exported schema."""
    traits = traits or {}
    impact = traits.get("impact_g", rng.uniform(0.5, 0.9))
    sway = traits.get("sway_g", rng.uniform(0.06, 0.12))
    lift = np.radians(traits.get("thigh_lift_deg", rng.uniform(15, 30)))

    cadence = CADENCE_HZ[walking_condition] * rng.uniform(0.92, 1.08)
    intervals = (1 / cadence) * (1 + 0.05 * rng.standard_normal(n_steps - 1))
    step_times = STILL_SECONDS + np.concatenate([[0.0], np.cumsum(intervals)])
    duration = step_times[-1] + STILL_SECONDS
    t = np.arange(0.0, duration, 1.0 / fs)
    t = np.sort(t + rng.uniform(-0.0005, 0.0005, len(t)))
    t[0] = max(t[0], 0.0)

    feet = [starting_foot if k % 2 == 0 else ("left" if starting_foot == "right" else "right") for k in range(n_steps)]

    # Body frame: [vertical (up), anterior, lateral (+ toward the left)]. Specific force in g; 1 g up at rest.
    body = np.zeros((len(t), 3))
    body[:, 0] = 1.0
    pitch = np.zeros(len(t))
    for s, foot in zip(step_times, feet):
        ipsilateral = foot == pocket_side
        amp = impact * rng.uniform(0.8, 1.2) * (1.0 if ipsilateral else 0.85)
        body[:, 0] += amp * _gauss(t, s, 0.025) - 0.3 * amp * _gauss(t, s + 0.08, 0.04)
        body[:, 2] += (1.0 if foot == "left" else -1.0) * sway * _gauss(t, s + 0.12, 0.10)
        body[:, 1] += (0.15 if ipsilateral else 0.05) * amp * _gauss(t, s - 0.05, 0.03)
        if ipsilateral:
            pitch += lift * rng.uniform(0.8, 1.2) * _gauss(t, s - 0.2, 0.1)

    # Phone frame: top of the phone pointing down the thigh, screen facing the leg (mirrored per side),
    # plus a random per-session tilt so axes never line up exactly with the body.
    lateral_sign = 1.0 if pocket_side == "right" else -1.0
    mount = np.array([[0.0, 1.0, 0.0],          # device x ~ anterior
                      [-1.0, 0.0, 0.0],         # device y ~ down the thigh
                      [0.0, 0.0, lateral_sign]])  # device z ~ lateral
    tilt = _rotation(rng.standard_normal(3), np.radians(rng.uniform(5, 25)))
    lateral_axis = np.array([0.0, 0.0, 1.0])
    readings = np.empty_like(body)
    for i in range(len(t)):
        thigh = _rotation(lateral_axis, -pitch[i]) @ body[i]
        readings[i] = tilt @ mount @ thigh
    readings += 0.015 * rng.standard_normal(readings.shape)

    if unexpected_gap:
        gap_start = step_times[n_steps // 2] + 0.1
        keep = (t < gap_start) | (t > gap_start + 0.4)
        t, readings = t[keep], readings[keep]

    sensor = pd.DataFrame({
        schema.TIME: t, schema.ELAPSED: t,
        "accel_x": readings[:, 0], "accel_y": readings[:, 1], "accel_z": readings[:, 2],
    })

    annotated = step_times + rng.normal(0, 0.008, n_steps)
    foot_labels = np.array(feet, dtype=object)
    foot_labels[rng.random(n_steps) < unknown_fraction] = "unknown"
    labels = pd.DataFrame({
        "step_id": np.arange(1, n_steps + 1),
        schema.TIME: annotated,
        "foot_label": foot_labels,
        "label_source": "assisted" if assisted else "manual",
        "is_valid": np.where(rng.random(n_steps) < invalid_fraction, "false", "true"),
        "reviewed": "true",
        "notes": "synthetic",
    })

    intervals_measured = np.diff(t)
    median_dt = float(np.median(intervals_measured))
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=int(rng.integers(0, 24 * 300)))
    metadata = {
        "schema_version": schema.SUPPORTED_SCHEMA_VERSIONS[-1],
        "participant_id": participant_id,
        "session_id": session_id,
        "phone_model": "synthetic",
        "ios_version": "n/a",
        "pocket_side": pocket_side,
        "starting_foot": starting_foot,
        "target_steps": n_steps,
        "walking_condition": walking_condition,
        "sampling_interval_seconds": median_dt,
        "requested_sampling_interval_seconds": 1.0 / fs,
        "sampling_rate_hz_estimate": 1.0 / median_dt,
        "session_start": start.isoformat().replace("+00:00", "Z"),
        "session_end": (start + timedelta(seconds=float(t[-1]))).isoformat().replace("+00:00", "Z"),
        "notes": "Synthetic session generated by ml.src.synthetic. Not a real recording.",
        "recording_status": "completed",
        "sample_count": int(len(t)),
        "unexpected_gap_count": int(unexpected_gap),
        "pauses": [],
        "interruptions": [],
        "pedometer_steps": None,
        "time_basis": "Seconds since session start (simulated).",
        "app_version": "synthetic",
        "synthetic": True,
    }
    return sensor, labels, metadata


def write_session(directory: Path, sensor: pd.DataFrame, labels: pd.DataFrame, metadata: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    sensor.to_csv(directory / schema.SENSOR_FILE, index=False, float_format="%.6f")
    labels.to_csv(directory / schema.LABELS_FILE, index=False, float_format="%.6f")
    (directory / schema.METADATA_FILE).write_text(json.dumps(metadata, indent=2))
    return directory


def generate_dataset(output_dir: str | Path, participants: int = 4, sessions_per_participant: int = 2,
                     steps: int = 40, seed: int = 7, fs: float = 100.0) -> list[Path]:
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    conditions = list(CADENCE_HZ)
    written = []
    for p in range(1, participants + 1):
        participant_id = f"SYN-P{p:02d}"
        traits = {"impact_g": rng.uniform(0.5, 0.9), "sway_g": rng.uniform(0.06, 0.12),
                  "thigh_lift_deg": rng.uniform(15, 30)}
        for s in range(1, sessions_per_participant + 1):
            session_id = f"{participant_id}-S{s}"
            sensor, labels, metadata = simulate_session(
                rng,
                participant_id=participant_id,
                session_id=session_id,
                pocket_side="right" if (p + s) % 2 == 0 else "left",
                starting_foot=str(rng.choice(schema.SIDES)),
                n_steps=steps,
                walking_condition=conditions[(p + s) % len(conditions)],
                fs=fs,
                unexpected_gap=(p == 1 and s == 1),
                assisted=(s == 2),
                traits=traits,
            )
            written.append(write_session(output_dir / session_id, sensor, labels, metadata))
    return written


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic PocketStep sessions (for testing only).")
    parser.add_argument("--output-dir", default="ml/data/synthetic")
    parser.add_argument("--participants", type=int, default=4)
    parser.add_argument("--sessions-per-participant", type=int, default=2)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    paths = generate_dataset(args.output_dir, args.participants, args.sessions_per_participant, args.steps, args.seed)
    print(f"Wrote {len(paths)} synthetic session(s) to {args.output_dir}. They are not real recordings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
