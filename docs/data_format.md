# Session data format (schema 1.0)

Each recording session is exported from the iPhone as `<session_id>.zip`, which contains one folder:

```
<session_id>/
├── sensor_data.csv
├── step_labels.csv
└── session_metadata.json
```

`python -m ml.src.ingest` validates the ZIP and publishes it as `ml/data/raw/<session_id>/`. The
Python pipeline searches the data directory recursively for folders that contain a
`sensor_data.csv`, so an unzipped folder anywhere under `--data-dir` also works for local analysis.

Raw sensor samples and step annotations live in separate files, so labels can be corrected or
replaced without touching the recording.

## Time basis

There is one clock: **seconds since the session start, on the device's monotonic clock**.
The app takes Core Motion's `CMAccelerometerData.timestamp` (seconds since boot, which does not jump
when the wall clock changes) and subtracts the uptime at which recording started.

The same basis is used for `timestamp_seconds` and `elapsed_seconds` in `sensor_data.csv`, the
footfall times in `step_labels.csv`, and the pause/interruption times in the metadata. Wall-clock
dates appear only as `session_start` and `session_end`.

## `sensor_data.csv`

One row per accelerometer reading, in time order.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `timestamp_seconds` | float | s | Time since session start (monotonic). |
| `elapsed_seconds` | float | s | The same value as `timestamp_seconds`. It is kept because the schema requires it, and the pipeline rejects a file where the two differ by more than 1 µs. |
| `accel_x` | float | g | Raw acceleration along the device x-axis, including gravity. |
| `accel_y` | float | g | Raw acceleration along the device y-axis, including gravity. |
| `accel_z` | float | g | Raw acceleration along the device z-axis, including gravity. |

Notes:

- Values are exactly what Core Motion reports: no filtering, no gravity removal and no axis
  rotation. A phone lying still reads about 1 g in total, split across the axes.
- Missing samples are never filled in. Pauses and dropped samples appear as gaps in
  `timestamp_seconds`.
- The phone axes are device axes, not body axes. Their relation to the leg depends on how the phone
  sits in the pocket.

Example:

```
timestamp_seconds,elapsed_seconds,accel_x,accel_y,accel_z
0.000412,0.000412,-0.051233,0.982114,-0.170551
0.010398,0.010398,-0.049876,0.979932,-0.168204
```

## `step_labels.csv`

One row per annotated step, in time order.

| Column | Type | Allowed values | Meaning |
|---|---|---|---|
| `step_id` | int | 1, 2, … | Position of the step in time order (renumbered on every save). |
| `timestamp_seconds` | float | s | The moment the foot lands, in the session time basis. It is not an arbitrary time within the step. |
| `foot_label` | string | `left`, `right`, `unknown` | Which foot landed. `unknown` means the annotator could not tell. |
| `label_source` | string | `manual`, `assisted` | `manual`: chosen by a person. `assisted`: proposed by the alternation pattern. |
| `is_valid` | bool | `true`, `false` | `false` marks a real step whose signal is unusable (for example, the phone shifted). It stays in the sequence. |
| `reviewed` | bool | `true`, `false` | Whether a person has checked the label. Assisted proposals start as `false`. |
| `notes` | string | free text | Optional, quoted per RFC 4180 when it contains commas, quotes or newlines. |

`reviewed` is an addition to the minimum required columns. If a label file has no `reviewed`
column, the pipeline treats `manual` labels as reviewed and `assisted` labels as unreviewed.

By default only steps that are **valid, reviewed, and `left` or `right`** are used as training targets.
All other steps are kept and reported with an exclusion reason.

Example:

```
step_id,timestamp_seconds,foot_label,label_source,is_valid,reviewed,notes
1,2.013000,left,manual,true,true,
2,2.571000,right,assisted,true,true,
3,3.118000,unknown,manual,true,true,"heel scuff, unclear"
```

## `session_metadata.json`

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | string | `"1.0"`. The pipeline rejects versions it does not know. |
| `participant_id` | string | Anonymous code (letters, digits, `-`, `_`). Never a name. |
| `session_id` | string | `S-<UTC yyyyMMdd-HHmmss>-<random hex>`; also the folder name. |
| `phone_model` | string | Hardware identifier (for example `iPhone15,2`) or a name the researcher typed in. |
| `ios_version` | string | iOS version at recording time. |
| `pocket_side` | string | `left` or `right`. |
| `starting_foot` | string | `left` or `right`, as instructed. |
| `target_steps` | int | Steps the participant was asked to take. |
| `walking_condition` | string | `slow`, `comfortable` or `brisk`. |
| `sampling_interval_seconds` | float or null | **Measured** median interval between samples. |
| `requested_sampling_interval_seconds` | float | Interval requested from Core Motion (0.01 = 100 Hz). |
| `sampling_rate_hz_estimate` | float or null | `1 / sampling_interval_seconds`. |
| `session_start`, `session_end` | ISO 8601 string | Wall-clock start and end (UTC). |
| `notes` | string | Researcher notes (no personal information). |
| `recording_status` | string | `completed`, `interrupted` (app terminated) or `failed` (sensor or file error). `in_progress` only appears in a file left behind by a crash; the app relabels it at the next launch. |
| `sample_count` | int | Rows written to `sensor_data.csv`. |
| `unexpected_gap_count` | int | Gaps longer than 3 requested intervals that were not pauses. |
| `pauses` | list | `{start_seconds, end_seconds, reason}` for each pause, in session time. |
| `interruptions` | list | `{at_seconds, reason}`, for example the app moving to the background. `at_seconds` is -1 when the time is unknown. |
| `pedometer_steps` | int or null | CMPedometer count. A reference only, never a label. |
| `time_basis` | string | Human-readable description of the time basis. |
| `app_version` | string | PocketStep version (build). |
| `synthetic` | bool | `true` only for generated test data. |

The pipeline treats missing `schema_version`, `participant_id`, `session_id` or `pocket_side` as an
error. Other missing keys from the required list produce warnings.

## Pipeline outputs

| File | Written by | Contents |
|---|---|---|
| `ml/data/processed/step_features.csv` | `ml.src.features` | One row per annotated step: identifiers, label info, window status, exclusion reason, all features. |
| `ml/models/*.joblib` | `ml.src.train` | Model, feature names, config, class mapping, split and data fingerprints (local, git-ignored). |
| `ml/published/` | `ml.src.retrain` (GitHub Actions) | `pocketstep.joblib`, `reports/` and a `README.md` status page for the model trained on `ml/data/raw/`. Committed. |
| `ml/reports/metrics.json` | `ml.src.evaluate` | Held-out metrics and limitations. |
| `ml/reports/confusion_matrix.png` | `ml.src.evaluate` | Confusion matrices without and with abstention. |
| `ml/reports/test_predictions.csv` | `ml.src.evaluate` | Per-step held-out predictions for error analysis. |
| `predictions.csv` (path set by `--output`) | `ml.src.predict` | `candidate_id, timestamp_seconds, predicted_label, confidence, probability_left, probability_right, abstained, window_status`. |
| `predictions.evaluation.json` | `ml.src.predict --evaluate-against-labels` | Detection and classification scores, reported separately. |
