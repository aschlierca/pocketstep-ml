import json
import shutil

from ml.src import schema
from ml.src.retrain import retrain


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
