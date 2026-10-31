import json
import shutil

from ml.src import schema
from ml.src.retrain import retrain
from ml.src.synthetic import generate_dataset


def _real_dataset(source, target):
    """Copies the synthetic dataset but marks it as real so it behaves like researcher data."""
    shutil.copytree(source, target)
    for path in target.rglob(schema.METADATA_FILE):
        metadata = json.loads(path.read_text())
        metadata["synthetic"] = False
        path.write_text(json.dumps(metadata))
    return target


def test_retrain_without_data_explains_status(tmp_path):
    (tmp_path / "raw").mkdir()
    assert retrain(tmp_path / "raw", tmp_path / "published", log=lambda *_: None) is False
    status = (tmp_path / "published" / "README.md").read_text()
    assert "Not trained: no valid sessions yet" in status
    assert not (tmp_path / "published" / "pocketstep.joblib").exists()


def test_retrain_with_too_few_participants_keeps_going(synthetic_dataset, tmp_path):
    data = _real_dataset(synthetic_dataset, tmp_path / "raw")
    for folder in data.iterdir():
        if folder.is_dir() and not folder.name.startswith("SYN-P01"):
            shutil.rmtree(folder)
    assert retrain(data, tmp_path / "published", log=lambda *_: None) is False
    assert "Only 1 participant(s)" in (tmp_path / "published" / "README.md").read_text()


def test_retrain_publishes_model_and_reports(synthetic_dataset, tmp_path):
    data = _real_dataset(synthetic_dataset, tmp_path / "raw")
    out = tmp_path / "published"
    assert retrain(data, out, log=lambda *_: None) is True
    assert (out / "pocketstep.joblib").exists()
    for name in ("metrics.json", "confusion_matrix.png", "test_predictions.csv"):
        assert (out / "reports" / name).exists()
    status = (out / "README.md").read_text()
    assert "| Macro F1 |" in status
    assert "Development target" in status


def test_retrain_publishes_threshold_model_for_unity(tmp_path):
    source = tmp_path / "varied"
    generate_dataset(source, participants=4, sessions_per_participant=2, steps=45, seed=8, varied_walkers=True)
    out = tmp_path / "published"
    assert retrain(_real_dataset(source, tmp_path / "raw"), out, log=lambda *_: None) is True
    model = json.loads((out / "threshold_model.json").read_text())
    assert model["recommended_method"] in ("fixed", "count_match", "model")
    for name in ("threshold_metrics.json", "threshold_cv_results.csv", "threshold_sessions.csv"):
        assert (out / "reports" / name).exists()
    status = (out / "README.md").read_text()
    assert status.index("## Step threshold model (Unity)") < status.index("## Left/right model")
    assert "| `oracle` |" in status


def test_short_sessions_explain_why_no_threshold_model(synthetic_dataset, tmp_path):
    out = tmp_path / "published"
    retrain(_real_dataset(synthetic_dataset, tmp_path / "raw"), out, log=lambda *_: None)
    assert not (out / "threshold_model.json").exists()
    assert "usable annotated steps; needs 30" in (out / "README.md").read_text()
