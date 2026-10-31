# PocketStep ML

A research prototype for **calibrating step detection to each person** who walks in place with
an iPhone in their pocket, so that an iOS Unity app counts steps reliably for light and heavy
walkers alike. It also studies whether the same accelerometer data can tell **left-foot steps from
right-foot steps**.

## Why this project exists

An iOS Unity app detects walking-in-place steps from the accelerometer by comparing the signal
with a threshold. One fixed threshold does not suit everyone: a light, soft walker may never cross
it (missed steps), while a heavy or bouncy walker crosses it more than once per step (extra steps).
The right threshold depends on the person, their pocket and phone, and how they walk.

The aim of the ML work is to choose that threshold automatically:

1. The user walks in place for a known number of steps (a short calibration walk).
2. A model trained on recordings of many different people walking in place estimates the
   threshold that detects that user's steps best.
3. The app uses that threshold from then on, instead of one value tuned for an "average" walker.

This repository provides what that requires: an app that records accelerometer data during a walk
with a target number of steps, labeled footfall times that serve as ground truth for which steps
should be detected, a shared dataset of different people, and a pipeline that evaluates step
detection against those labels. **The threshold-calibration model itself is not built yet.** See
[docs/project_goal.md](docs/project_goal.md) for the plan and what is already in place.

## Components

It has two independent parts that communicate only through exported CSV files:

1. **PocketStep**, an iOS app (Swift/SwiftUI) that records raw accelerometer data, lets a
   researcher label each step's foot, and exports a documented session dataset.
2. **A Python pipeline** that validates and preprocesses the recordings, trains a baseline Random
   Forest without session or participant leakage, evaluates it on held-out recordings, and runs
   inference on new recordings with a `left` / `right` / `unknown` output.

This is not a fitness app. The priorities are data quality, trustworthy labels, reproducibility
and honest evaluation.

> **Status.** The Python pipeline is tested end to end on synthetic data, and newly published
> sessions retrain the model automatically through GitHub Actions. The iOS app builds and
> its logic tests pass in the iOS Simulator, but recording has not yet been verified on a
> physical iPhone (the Simulator has no motion sensors). **No real-world accuracy has been
> measured.** The macro-F1 ≥ 0.85 figure below is a development target, not a result.

## Researcher quick start

Collect data with your own iPhone and add it to the shared dataset in a few steps. The model then
retrains automatically.

**Once, after cloning** (you need write access to this repository):

```bash
git clone https://github.com/aschlierca/pocketstep-ml.git
cd pocketstep-ml
ios/setup_signing.sh                      # finds your Apple team; see "iOS app" if it can't
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r ml/requirements.txt
```

Open `ios/PocketStep.xcodeproj`, select your iPhone and press **Run** (see
[Running on an iPhone](#running-on-an-iphone) for first-run prompts).

**For every session:**

1. Record and label it in the app ([protocol](docs/data_collection_protocol.md)).
2. In **Session Review**, tap **Export ZIP** and AirDrop it to your Mac (it lands in `~/Downloads`).
3. From the repository folder, run:

   ```bash
   source .venv/bin/activate
   python -m ml.src.ingest
   ```

   This validates every new `S-*.zip` in `~/Downloads`, shows what will be published, asks you
   to confirm consent for public release, and then commits and pushes the session.
4. GitHub Actions retrains the model on all published sessions and commits the result to
   [`ml/published/`](ml/published/README.md) within a few minutes. There is nothing else to run.

To fix labels later, correct them in the app, export again and re-run `python -m ml.src.ingest`.
It replaces the labels and keeps the recording.

## Contents

- [Why this project exists](#why-this-project-exists)
- [Researcher quick start](#researcher-quick-start)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [iOS app](#ios-app): Xcode setup, testing on an iPhone, permissions
- [Collecting and labeling data](#collecting-and-labeling-data)
- [Data format](#data-format)
- [Python pipeline](#python-pipeline): setup, commands, outputs
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Privacy and research ethics](#privacy-and-research-ethics)

Detailed documents:
[project goal](docs/project_goal.md) ·
[data format](docs/data_format.md) ·
[collection and labeling protocol](docs/data_collection_protocol.md) ·
[ML pipeline design](docs/ml_pipeline.md)

## Architecture

```
iPhone (PocketStep app)                              Computer (Python, ml/)
┌────────────────────────────────────┐              ┌─────────────────────────────────────────────┐
│ Session setup ─▶ Recording         │   ZIP of     │ preprocessing: validate, gravity separation │
│   CMMotionManager @100 Hz          │   3 files    │ segmentation: windows at labeled footfalls  │
│   CMPedometer (reference only)     │ ───────────▶ │ features: one row per step                  │
│ Step labeling (manual / assisted)  │  sensor_data │ train: grouped split, Random Forest         │
│ Review & export                    │  step_labels │ evaluate: held-out metrics + reports        │
│ Data stays on device until export  │  metadata    │ predict: detect steps, left/right/unknown   │
└────────────────────────────────────┘              └──────────────────────┬──────────────────────┘
                                                      python -m ml.src.ingest │ validate, commit, push
                                                                              ▼
                                       GitHub: ml/data/raw/ ──▶ Actions: retrain ──▶ ml/published/
```

```
.github/workflows/
  retrain.yml                retrains the published model when data or pipeline code changes
ios/
  project.yml                XcodeGen spec (the generated PocketStep.xcodeproj is committed)
  setup_signing.sh           writes Config/Signing.xcconfig for your Apple team (git-ignored)
  Config/                    build settings shared by everyone
  PocketStep/
    App/                     entry point, navigation, Info.plist
    Models/                  session metadata, samples, step labels + CSV codec
    Services/                recorder, CSV writer, session store, candidate detection,
                             assisted labeling, annotation checks, ZIP export
    Views/                   setup, recording, labeling, review screens
    Tests/                   XCTest unit tests
ml/
  src/                       schema, config, preprocessing, segmentation, features,
                             models, train, evaluate, predict, synthetic, ingest, retrain
  tests/                     pytest suite
  data/raw/                  the published dataset, one folder per session (added by ingest)
  data/processed/            generated feature tables (git-ignored)
  data/synthetic/            8 synthetic sample sessions (committed, flagged synthetic)
  published/                 model + reports trained on data/raw by GitHub Actions
  models/, reports/          your local outputs (git-ignored)
  requirements.txt
docs/                        data format, protocol, pipeline design
```

## Prerequisites

| For | You need |
|---|---|
| iOS app | A Mac with Xcode 15 or later (built with Xcode 26.2), an iPhone with iOS 17 or later, and an Apple ID for signing (a free personal team works for testing on your own phone) |
| Regenerating the Xcode project (optional) | [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`) |
| Python pipeline | Python 3.11 or later (tested with 3.12) |

## iOS app

### Xcode setup

1. Run `ios/setup_signing.sh` once. It reads your team ID from your Apple Development certificate
   and writes `ios/Config/Signing.xcconfig`, which is git-ignored. The bundle identifier becomes
   `org.pocketstep.PocketStep.<TEAM_ID>`, so it never clashes with other researchers'.
   - If it finds no certificate: in Xcode open **Settings → Accounts**, add your Apple ID
     (a free one works), then **Manage Certificates… → + → Apple Development**, and run it again.
   - With several teams it asks which to use, or you can pass one: `ios/setup_signing.sh ABCDE12345`.
2. Open `ios/PocketStep.xcodeproj` in Xcode. **Do not change the team in the Signing tab**: that
   edits the shared project file. The script's settings are already applied.
3. If you add or remove source files, regenerate the project: `cd ios && xcodegen generate`.

### Running on an iPhone

1. Connect the iPhone and unlock it. On iOS 16 and later, turn on
   **Settings → Privacy & Security → Developer Mode** and restart when asked.
2. Choose the iPhone as the run destination and press **Run** (⌘R).
3. With a free personal team, the first launch may be blocked. Trust the developer under
   **Settings → General → VPN & Device Management**.

### Running the unit tests

In Xcode press ⌘U, or from the command line:

```bash
cd ios
xcodebuild test -project PocketStep.xcodeproj -scheme PocketStep \
  -destination 'platform=iOS Simulator,name=iPhone 16'
```

The tests cover CSV writing and incremental flushing, sampling statistics, the metadata JSON keys,
step-label CSV round trips, candidate step detection, assisted labeling, annotation warnings,
reading sensor files, and ZIP export. The Simulator can run the app and these tests, but it cannot
record. Starting a recording there shows "No accelerometer is available".

### Sensor permissions

- **Raw accelerometer data** (CMMotionManager) needs no permission prompt.
- **Pedometer counts** (CMPedometer) need *Motion & Fitness* access. The app asks when the
  recording screen opens. If access is denied, recording still works and the app shows how to
  enable it: **Settings → Privacy & Security → Motion & Fitness**. Pedometer counts are shown as a
  reference only and are never used as labels.
- The app does not request location, camera, microphone, contacts or network access.

During recording the app keeps the screen from auto-locking and turns on the proximity sensor, so
the screen switches off in the pocket and cannot be pressed. If the app is backgrounded (for
example, by the side button), recording pauses and the interruption is logged.

## Collecting and labeling data

The full protocol, including consent, phone placement, interruptions, labeling and optional video
sync, is in [docs/data_collection_protocol.md](docs/data_collection_protocol.md). In short:

1. **New Session**: anonymous participant ID, pocket side, starting foot, target steps, walking
   condition, notes.
2. **Record**: press Start, pocket the phone during the countdown (screen toward the leg, top
   pointing down), stand still for 2 s, walk in place, stand still for 2 s, then Stop.
3. **Label**: on the chart, choose each footfall and add Left/Right/Unknown (**manual**), or detect
   candidate steps and let **assisted labeling** propose alternating feet. Proposals are marked
   *Needs review* until a person confirms or corrects each one. Mark unclear steps *Unknown* and
   unusable ones *Invalid*.
4. **Review & export**: check the counts and warnings, then tap **Export ZIP** and AirDrop it to
   your Mac.
5. **Publish**: run `python -m ml.src.ingest` from the repository. It validates the session and
   pushes it to `ml/data/raw/`, and the model retrains automatically
   (see [Researcher quick start](#researcher-quick-start)).

## Data format

Each session folder contains three files. All times are **seconds since session start on the
monotonic clock**; acceleration is in **g**, raw from Core Motion (gravity included, device axes).

| File | Columns / keys |
|---|---|
| `sensor_data.csv` | `timestamp_seconds`, `elapsed_seconds` (identical; validated), `accel_x`, `accel_y`, `accel_z` |
| `step_labels.csv` | `step_id`, `timestamp_seconds` (footfall), `foot_label` (`left`/`right`/`unknown`), `label_source` (`manual`/`assisted`), `is_valid`, `reviewed`, `notes` |
| `session_metadata.json` | `schema_version` (`"1.0"`), `participant_id`, `session_id`, `phone_model`, `ios_version`, `pocket_side`, `starting_foot`, `target_steps`, `walking_condition`, `sampling_interval_seconds` (measured), `session_start`, `session_end`, `notes`, plus pauses, interruptions, sample count, measured rate and other details |

See [docs/data_format.md](docs/data_format.md) for types, examples and every metadata key.

## Python pipeline

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r ml/requirements.txt
```

Run every command below **from the repository root** (the modules are run as `ml.src.*`).

### Try it on the synthetic sample data

```bash
python -m ml.src.preprocessing --data-dir ml/data/synthetic
python -m ml.src.train --data-dir ml/data/synthetic --model-path ml/models/synthetic.joblib
python -m ml.src.evaluate --model-path ml/models/synthetic.joblib
python -m ml.src.predict --model-path ml/models/synthetic.joblib \
  --input ml/data/synthetic/SYN-P02-S1 --evaluate-against-labels
```

Synthetic results only show that the code runs; every report marks them as synthetic. Regenerate or
enlarge the sample set with `python -m ml.src.synthetic --output-dir ml/data/synthetic --participants 4`.

### Commands for real data

| Step | Command |
|---|---|
| Publish exported sessions (validate, commit, push) | `python -m ml.src.ingest [ZIPs or folders]` |
| Retrain the published model locally (normally done by GitHub Actions) | `python -m ml.src.retrain` |
| Validate recordings | `python -m ml.src.preprocessing --data-dir ml/data/raw` |
| Export the per-step feature table (optional) | `python -m ml.src.features --data-dir ml/data/raw --output ml/data/processed/step_features.csv` |
| Train | `python -m ml.src.train --data-dir ml/data/raw` |
| Evaluate on held-out data | `python -m ml.src.evaluate --model-path ml/models/pocketstep.joblib` |
| Predict on a new recording | `python -m ml.src.predict --model-path ml/models/pocketstep.joblib --input path/to/sensor_data.csv` |
| Run tests | `python -m pytest` |

Useful options:

- `ingest`: with no arguments it takes every `S-*.zip` in `~/Downloads`. `--dry-run` validates
  only, `--no-push` commits without pushing, `--yes` skips the confirmation (only when consent is
  already confirmed).
- `train`: `--group-by participant|session` (default participant), `--seed`, `--test-fraction`,
  `--min-groups`, `--feature-set live|offline`, `--target pocket_relative|absolute`,
  `--n-estimators`, `--max-depth`, `--min-samples-leaf`, `--confidence-threshold`,
  `--calibration none|sigmoid|isotonic`, `--include-unreviewed`, `--config config.json`, and
  `--save-config` to write the effective configuration.
- `evaluate`: `--threshold` (choose it from the training sweep, not from test results),
  `--data-dir` if the data has moved, `--reports-dir`.
- `predict`: `--output` (default `ml/reports/predictions.csv`), `--threshold`, `--pocket-side`
  when there is no `session_metadata.json` next to the sensor file, and `--evaluate-against-labels`
  to score detection and classification separately afterwards.

`predict` needs the pocket side. It reads `session_metadata.json` from the same folder as the
sensor file, or takes `--pocket-side left|right`.

### Expected outputs

- **Ingest**: one line per session (`NEW`, `UPDATED LABELS`, `skipped` or `REJECTED` with the
  reason), label counts, warnings and the notes that will become public; then a consent prompt,
  a commit and a push. A different recording that reuses an existing session ID is refused.
- **Published model** (`ml/published/`, written by GitHub Actions): `pocketstep.joblib`,
  `reports/metrics.json`, `reports/confusion_matrix.png`, `reports/test_predictions.csv` and
  `README.md` with the dataset summary, split, held-out metrics and limitations. Until at least
  3 participants have usable data, the README explains why no model was trained.
- **Validate**: one block per session, `OK` or `REJECTED`, with errors and warnings (gaps, duplicate
  or out-of-order timestamps, units, sampling rate, label problems) and basic stats. Exit code 1
  if any session is rejected.
- **Train**: the number of usable and excluded steps by reason, the exact train/test groups, an
  out-of-fold coverage/accuracy table for choosing a threshold, and the saved model path. With too
  few participants or sessions it stops and explains why (exit code 2) instead of splitting
  individual steps.
- **Evaluate**: prints accuracy, balanced accuracy, per-foot precision/recall/F1, macro F1, the
  confusion matrix, coverage and abstention at the threshold, per-session results and limitations.
  Writes `ml/reports/metrics.json`, `ml/reports/confusion_matrix.png` and
  `ml/reports/test_predictions.csv`.
- **Predict**: writes one row per detected step (`candidate_id, timestamp_seconds, predicted_label,
  confidence, probability_left, probability_right, abstained, window_status`). With
  `--evaluate-against-labels` it also writes `<output>.evaluation.json` with detection precision and
  recall, and left/right accuracy on the matched steps.

### How the pipeline avoids common mistakes

- Splits by participant or session and **never randomly across neighbouring steps**. It stops if
  there are fewer than 3 groups.
- Excludes `unknown`, invalid and unreviewed labels from training and reports how many.
- Fits learned preprocessing, calibration and threshold choices on training data only.
- Separates live features from those that need future steps (`time_until_next_step` is
  offline-only).
- Scores step detection separately from left/right classification.
- Refuses to mix synthetic and real sessions, and labels synthetic results everywhere.

See [docs/ml_pipeline.md](docs/ml_pipeline.md) for the reasoning, including why the model learns
"pocket-side leg vs other leg" by default and converts the result to left/right.

## Known limitations

- **Automatic retraining needs data from at least 3 participants** before it publishes a model,
  because it refuses to split one person's steps between training and test.
- **Not validated on real recordings yet.** Recording on a physical iPhone, the end-to-end export,
  and model performance on real people are all unverified.
- **Small-sample evaluation.** With a handful of participants, held-out metrics have wide
  uncertainty. Reports say so.
- **Fixed pocket and posture.** The protocol covers walking in place with a snug front pocket.
  Other pockets, loose clothing, walking forward or turning are not covered.
- **Labels depend on the annotator.** Footfall times come from the chart (peaks) or from video
  synced by hand. Assisted labels assume alternation and must be reviewed.
- **Step detection is a simple peak detector** with fixed thresholds. It has not been tuned on
  real data.
- **Offline signal processing.** Gravity separation uses zero-phase filtering, and live
  predictions need about 0.4 s after the footfall. An on-device real-time version would need
  causal filters and a model running on the phone (for example through Core ML). This prototype
  keeps inference in Python.
- **Sampling.** iPhones may deliver fewer samples than requested and with jitter. The measured
  rate is recorded and gaps are flagged, but the pipeline does not resample.
- **Probabilities.** Random Forest probabilities are uncalibrated unless `--calibration` is used.
  The confidence threshold is a tuning knob, not a guarantee.
- **The synthetic data is a caricature** designed to exercise the code; it has an easy,
  deliberately planted signal.

## Troubleshooting

| Problem | What to do |
|---|---|
| "No accelerometer is available" | You are in the Simulator. Run on a physical iPhone. |
| Pedometer shows "Motion & Fitness access is off" | Enable it in **Settings → Privacy & Security → Motion & Fitness**. Recording works without it. |
| Recording paused by itself | The app went to the background (side button, call, app switch). Check the phone position, then Resume or Stop. Check the interruption list in Session Review. |
| Measured rate well below 100 Hz | Close other apps and keep PocketStep in the foreground. Low-power mode can reduce rates. The pipeline uses the measured rate, but gaps are flagged. |
| Xcode signing errors | Choose a Team and make the bundle identifier unique. |
| "Untrusted Developer" on the iPhone | **Settings → General → VPN & Device Management** → trust your profile. |
| `No sessions found under ...` | Unzip exports so each session is a folder containing `sensor_data.csv`, for example `ml/data/raw/S-.../sensor_data.csv`. |
| A session is `REJECTED` | Read the error codes from `python -m ml.src.preprocessing`. Fix labels in the app and re-export. Never edit `sensor_data.csv` by hand. |
| `Training stopped: Only N participant(s)...` | Collect more participants, or use `--group-by session` if you accept that results then overstate performance on new people. |
| `mixes synthetic and real sessions` | Keep `ml/data/synthetic` separate from `ml/data/raw`. |
| Many steps excluded as `unreviewed_label` | Review assisted proposals in the app (or use `--include-unreviewed` for exploration only). |
| Many `window_out_of_range` / `window_overlaps_gap` | Steps too close to the start/end or to a pause. Stand still for 2 s at the start and end, and avoid pausing mid-walk. |
| `ModuleNotFoundError: No module named 'ml'` | Run commands from the repository root. |
| `ios/setup_signing.sh` finds no certificate | Add your Apple ID in Xcode → Settings → Accounts and create an Apple Development certificate (Manage Certificates… → +), then rerun. |
| Xcode shows a modified `project.pbxproj` | You changed signing in the Signing tab. Discard it (`git checkout ios/PocketStep.xcodeproj`) and use `ios/setup_signing.sh`. |
| Ingest: `No session ZIPs found` | AirDrop the export to this Mac (it lands in `~/Downloads`) or pass the ZIP path. |
| Ingest: `git push failed` / permission denied | Ask the repository owner to add you as a collaborator, or run with `--no-push` and open a pull request. |
| Ingest: `a different recording with session ID ... is already published` | Each recording keeps its own session ID. Don't rename exports; record a new session instead. |
| `ml/published/README.md` did not update | Check the **Actions** tab. If the commit step fails with a permission error, set **Settings → Actions → General → Workflow permissions** to *Read and write*. |

## Privacy and research ethics

**The dataset is public.** Every session published with `python -m ml.src.ingest` goes into
`ml/data/raw/` in this public repository: accelerometer samples, step labels and metadata, which
includes the anonymous participant ID, phone model, iOS version, recording date and time, and the
notes field. Anyone can download it, and removing it later does not remove it from clones,
forks or Git history.

- Only publish sessions from participants who **consented to public release** of their
  anonymized motion data. `ingest` asks for this confirmation every time unless you pass `--yes`.
- Participants are identified only by anonymous codes; `ingest` rejects IDs that are not of that
  form. Keep any table linking codes to people private, never in this repository.
- Never put names or identifying details in notes; `ingest` shows the notes before publishing.
- The app collects no names, accounts, location, contacts, audio or video, and uploads nothing
  itself. Data leaves the phone only when a researcher exports it.
- Exported ZIPs, videos and local pipeline outputs stay git-ignored.
- Research with human participants may require institutional (IRB or ethics committee) approval and
  an approved informed-consent procedure that covers public data sharing. Obtain both before
  collecting data from anyone other than yourself.
