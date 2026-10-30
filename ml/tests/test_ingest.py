import json
import shutil
import subprocess
import zipfile

import numpy as np
import pandas as pd
import pytest

from ml.src import ingest, schema
from ml.src.ingest import IngestError, extract_zip, main as ingest_main
from ml.src.synthetic import simulate_session, write_session


def run(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A working clone with a local bare 'origin', as a researcher would have."""
    remote = tmp_path / "remote.git"
    run(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    work = tmp_path / "work"
    run(tmp_path, "clone", "-q", str(remote), str(work))
    for key, value in (("user.name", "Researcher"), ("user.email", "r@example.org"), ("pull.rebase", "true")):
        run(work, "config", key, value)
    (work / "ml" / "data" / "raw").mkdir(parents=True)
    (work / "ml" / "data" / "raw" / "README.md").write_text("dataset\n")
    run(work, "add", ".")
    run(work, "commit", "-q", "-m", "init")
    run(work, "push", "-q", "-u", "origin", "main")
    monkeypatch.chdir(work)
    return work, remote


def make_export(directory, session_id="S-20261030-101500-ABCD", participant="P-7QK2MX", seed=0,
                synthetic=False, mutate_labels=False):
    """A ZIP shaped like the app's export: <session_id>/<three files>."""
    rng = np.random.default_rng(seed)
    sensor, labels, metadata = simulate_session(rng, participant_id=participant, session_id=session_id,
                                                pocket_side="right", starting_foot="left", n_steps=12,
                                                walking_condition="comfortable")
    metadata["synthetic"] = synthetic
    metadata["notes"] = ""
    if mutate_labels:
        labels.loc[0, "foot_label"] = "unknown"
    folder = write_session(directory / "build" / session_id, sensor, labels, metadata)
    zip_path = directory / f"{session_id}.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in folder.iterdir():
            archive.write(path, f"{session_id}/{path.name}")
    return zip_path


def remote_files(remote):
    return run(remote, "ls-tree", "-r", "--name-only", "main").splitlines()


def test_new_session_is_committed_and_pushed(repo, tmp_path):
    work, remote = repo
    zip_path = make_export(tmp_path)
    assert ingest_main([str(zip_path), "--yes"]) == 0
    files = remote_files(remote)
    for name in (schema.SENSOR_FILE, schema.LABELS_FILE, schema.METADATA_FILE):
        assert f"ml/data/raw/S-20261030-101500-ABCD/{name}" in files
    message = run(remote, "log", "-1", "--format=%an|%s", "main")
    assert message.startswith("Researcher|Add 1 session(s) (P-7QK2MX)")


def test_declining_confirmation_publishes_nothing(repo, tmp_path):
    work, remote = repo
    assert ingest_main([str(make_export(tmp_path))], input_fn=lambda _: "n") == 1
    assert not (work / "ml/data/raw/S-20261030-101500-ABCD").exists()
    assert run(work, "status", "--porcelain") == ""


def test_reingesting_is_a_no_op_and_label_fixes_update(repo, tmp_path):
    work, remote = repo
    zip_path = make_export(tmp_path)
    assert ingest_main([str(zip_path), "--yes"]) == 0
    head = run(remote, "rev-parse", "main")
    assert ingest_main([str(zip_path), "--yes"]) == 0
    assert run(remote, "rev-parse", "main") == head

    fixed = make_export(tmp_path / "fixed", mutate_labels=True)
    assert ingest_main([str(fixed), "--yes"]) == 0
    assert run(remote, "log", "-1", "--format=%s", "main").startswith("Update labels of 1 session(s)")
    labels = pd.read_csv(work / "ml/data/raw/S-20261030-101500-ABCD" / schema.LABELS_FILE)
    assert labels.loc[0, "foot_label"] == "unknown"


def test_different_recording_with_same_id_is_refused(repo, tmp_path):
    work, remote = repo
    assert ingest_main([str(make_export(tmp_path, seed=0)), "--yes"]) == 0
    head = run(remote, "rev-parse", "main")
    assert ingest_main([str(make_export(tmp_path / "other", seed=1)), "--yes"]) == 1
    assert run(remote, "rev-parse", "main") == head


@pytest.mark.parametrize("kwargs", [{"synthetic": True}, {"participant": "Jane Doe"}])
def test_unsuitable_sessions_are_rejected(repo, tmp_path, kwargs):
    work, remote = repo
    assert ingest_main([str(make_export(tmp_path, **kwargs)), "--yes"]) == 1
    assert len(remote_files(remote)) == 1


def test_invalid_session_is_rejected(repo, tmp_path):
    work, remote = repo
    zip_path = tmp_path / "S-BAD.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("S-BAD/sensor_data.csv", "timestamp_seconds,accel_x\n0,1\n")
        archive.writestr("S-BAD/step_labels.csv", "step_id\n")
        archive.writestr("S-BAD/session_metadata.json", json.dumps({"session_id": "S-BAD"}))
    assert ingest_main([str(zip_path), "--yes"]) == 1
    assert len(remote_files(remote)) == 1


def test_zip_with_unsafe_paths_is_refused(tmp_path):
    zip_path = tmp_path / "S-EVIL.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../sensor_data.csv", "x")
    with pytest.raises(IngestError, match="unexpected path"):
        extract_zip(zip_path, tmp_path / "out")


def test_push_race_is_resolved_by_rebasing(repo, tmp_path, monkeypatch):
    work, remote = repo
    other = tmp_path / "other_clone"
    run(tmp_path, "clone", "-q", str(remote), str(other))
    run(other, "config", "user.name", "Other")
    run(other, "config", "user.email", "o@example.org")
    theirs = make_export(tmp_path / "a", session_id="S-20261030-090000-0001", participant="P-AAAAAA")
    ours = make_export(tmp_path / "b", session_id="S-20261030-090000-0002", participant="P-BBBBBB", seed=2)

    real_collect = ingest.collect

    def collect_while_someone_else_pushes(*args, **kwargs):
        # Another researcher publishes after this run pulled, so this run's first push is rejected.
        folder = other / "ml/data/raw/S-20261030-090000-0001"
        folder.mkdir(parents=True)
        with zipfile.ZipFile(theirs) as archive:
            for name in (schema.SENSOR_FILE, schema.LABELS_FILE, schema.METADATA_FILE):
                (folder / name).write_bytes(archive.read(f"S-20261030-090000-0001/{name}"))
        run(other, "add", ".")
        run(other, "commit", "-q", "-m", "other researcher")
        run(other, "push", "-q")
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(ingest, "collect", collect_while_someone_else_pushes)
    assert ingest.main([str(ours), "--yes"]) == 0
    files = remote_files(remote)
    assert any("S-20261030-090000-0001" in f for f in files)
    assert any("S-20261030-090000-0002" in f for f in files)
