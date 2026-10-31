"""Retrain the published models from every session in ml/data/raw.

The GitHub Actions workflow runs this after new data is pushed and commits the results, so
researchers never retrain by hand. It can also be run locally:

    python -m ml.src.retrain

Writes to ml/published/:
    threshold_model.json               per-user step threshold for the Unity app (see ml/src/threshold.py)
    reports/threshold_metrics.json, threshold_cv_results.csv, threshold_sessions.csv
    pocketstep.joblib                  left/right classifier
    reports/metrics.json, confusion_matrix.png, test_predictions.csv
    README.md                          human-readable status: data, evaluation, limitations

Each model is trained only when there is enough data for it. Otherwise README.md explains why, any
previous version is left untouched, and the command still exits 0 so the workflow stays green.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import PipelineConfig
from .evaluate import evaluate
from .features import build_feature_table, exclusion_summary
from .preprocessing import find_session_dirs, load_sessions
from .threshold import format_summary, train_threshold_model
from .train import TrainingDataError, train

DEFAULT_DATA_DIR = Path("ml/data/raw")
DEFAULT_OUTPUT_DIR = Path("ml/published")
MODEL_FILE = "pocketstep.joblib"
THRESHOLD_MODEL_FILE = "threshold_model.json"


def dataset_summary(data_dir: Path, config: PipelineConfig) -> tuple[list[str], dict]:
    sessions = load_sessions(data_dir, config) if data_dir.exists() and find_session_dirs(data_dir) else []
    accepted = [s for s in sessions if s.report.ok]
    table, rejected = build_feature_table(sessions, config)
    lines = [
        f"- Sessions: {len(sessions)} ({len(rejected)} rejected by validation)",
        f"- Participants: {len({s.participant_id for s in accepted})}",
        f"- Pocket sides: {', '.join(sorted({str(s.pocket_side) for s in accepted})) or 'none'}",
    ]
    if len(table):
        lines.append(f"- Annotated steps: {len(table)}; by status: {exclusion_summary(table)}")
    for session in rejected:
        lines.append(f"- Rejected `{session.session_id}`: {', '.join(session.report.codes('error'))}")
    return lines, {"sessions": len(sessions), "accepted": len(accepted)}


def metrics_lines(report: dict) -> list[str]:
    overall = report["all_steps_no_abstention"]
    thresholded = report["with_confidence_threshold"]
    covered = thresholded["on_covered_steps"]
    split = report["dataset"]["split"]
    target = report["development_target"]
    status = {True: "met", False: "not met", None: "not applicable"}[target["met"]]
    lines = [
        f"Split by **{split['group_by']}** (seed {split['seed']}): "
        f"train {split['train']['groups']}, test {split['test']['groups']}.",
        "",
        "| Held-out metric | Value |",
        "|---|---|",
        f"| Steps scored | {overall['n']} |",
        f"| Accuracy | {overall['accuracy']:.3f} |",
        f"| Balanced accuracy | {overall['balanced_accuracy']:.3f} |",
        f"| Macro F1 | {overall['macro_f1']:.3f} |",
        f"| Coverage at threshold {thresholded['threshold']:.2f} | {thresholded['coverage']:.1%} |",
        f"| Accuracy on covered steps | "
        f"{'n/a' if covered.get('accuracy') is None else format(covered['accuracy'], '.3f')} |",
        "",
        f"Development target (macro F1 ≥ {target['target']}): **{status}**. This is a target, not a claim.",
        "",
        "Limitations:",
        *[f"- {note}" for note in report["limitations"]],
    ]
    return lines


def threshold_lines(report: dict, path: Path) -> list[str]:
    data = report["dataset"]
    lines = [
        f"Model: `{path.as_posix()}` (copy into the Unity project; see `unity/README.md`). "
        f"The app uses **`{report['recommended_method']}`**, the best of fixed / count_match / model below.",
        "",
        f"Leave-one-participant-out over {data['participants']} participant(s), {data['sessions']} session(s): "
        "each session's first steps are the calibration walk and the steps after it are scored.",
        "",
        format_summary(report),
        "",
    ]
    if data["skipped_sessions"]:
        lines += ["Sessions not used:", *[f"- `{k}`: {v}" for k, v in data["skipped_sessions"].items()], ""]
    lines += ["Limitations:", *[f"- {note}" for note in report["limitations"]]]
    return lines


def retrain_threshold(data_dir: Path, output_dir: Path, config: PipelineConfig, log=print) -> tuple[bool, list[str]]:
    path = output_dir / THRESHOLD_MODEL_FILE
    try:
        report = train_threshold_model(data_dir, config, path, output_dir / "reports", log=log)
    except TrainingDataError as exc:
        kept = path.exists()
        return False, [f"Not trained: {exc}", "",
                       "The previous threshold model was kept." if kept else "No threshold model has been published yet."]
    return True, threshold_lines(report, path)


def retrain(data_dir: Path = DEFAULT_DATA_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR,
            config: Optional[PipelineConfig] = None, log=print) -> bool:
    """Returns True when at least one model was trained and evaluated."""
    config = config or PipelineConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_lines, counts = dataset_summary(data_dir, config)
    commit = os.environ.get("GITHUB_SHA", "")
    header = [
        "# Published model status",
        "",
        "Generated automatically by `python -m ml.src.retrain` from every session in `ml/data/raw/`. "
        "Do not edit by hand.",
        "",
        f"- Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        + (f" (data commit `{commit[:7]}`)" if commit else ""),
        "",
        "## Data",
        "",
        *data_lines,
        "",
    ]

    no_data = ["Not trained: no valid sessions yet. Collect data and publish it with `python -m ml.src.ingest`."]
    threshold_trained, threshold_body = (False, no_data) if counts["accepted"] == 0 else \
        retrain_threshold(data_dir, output_dir, config, log)

    trained = False
    if counts["accepted"] == 0:
        body = no_data
    else:
        try:
            model_path = output_dir / MODEL_FILE
            train(data_dir, config, model_path, log=log)
            report = evaluate(model_path, data_dir, output_dir / "reports")
            body = [f"Model: `{model_path.as_posix()}`. Reports: `{(output_dir / 'reports').as_posix()}/`.", "",
                    *metrics_lines(report)]
            trained = True
        except TrainingDataError as exc:
            kept = (output_dir / MODEL_FILE).exists()
            body = [f"Not trained: {exc}", "",
                    "The previous model was kept." if kept else "No model has been published yet."]
    sections = ["## Step threshold model (Unity)", "", *threshold_body, "", "## Left/right model", "", *body]
    (output_dir / "README.md").write_text("\n".join(header + sections) + "\n")
    log("\n".join(sections))
    return trained or threshold_trained


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Retrain and evaluate the published models from ml/data/raw.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", help="Optional pipeline config JSON")
    args = parser.parse_args(argv)
    config = PipelineConfig.load(args.config) if args.config else PipelineConfig()
    retrain(args.data_dir, args.output_dir, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
