# ML pipeline design

This document explains how the Python pipeline turns exported sessions into a left/right step
classifier, and why it is built this way. Commands are in the [README](../README.md#python-pipeline).

```
ml/data/raw/<session>/                 validate        preprocess          segment            features
  sensor_data.csv     ─┐            ┌───────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
  step_labels.csv     ─┼──────────▶ │ report    │──▶│ gravity,     │──▶│ windows around│──▶│ 1 row / step │
  session_metadata.json┘            │ issues    │   │ linear accel │   │ annotated     │   │ + exclusion  │
                                    └───────────┘   └──────────────┘   │ footfalls     │   │   reason     │
                                                                       └───────────────┘   └──────┬───────┘
                                                                                                  ▼
                               predict ◀── model bundle ◀── train (grouped split) ──▶ evaluate (held-out groups)
```

## Modules

| Module | Responsibility |
|---|---|
| `schema.py` | File names, column names, allowed values. |
| `config.py` | All tunable settings as dataclasses; saved inside every model. |
| `preprocessing.py` | Loading, validation reports, cleaning, gravity separation, filtering. CLI validates data. |
| `segmentation.py` | Time-based step windows, label exclusion rules, peak detection, detection matching. |
| `features.py` | Per-window and context features; live/offline feature sets. CLI writes the step table. |
| `models.py` | Model registry, calibration, abstention threshold, model bundle save/load. |
| `train.py` | Grouped split, fitting, out-of-fold validation, bundle metadata. |
| `evaluate.py` | Held-out metrics, plots, limitations. |
| `predict.py` | Detection + classification of a new recording; optional separate scoring. |
| `synthetic.py` | Synthetic sessions for tests and demos only. |
| `ingest.py` | Validates exported sessions and publishes them to `ml/data/raw/` with git. |
| `retrain.py` | Trains and evaluates on `ml/data/raw/` and writes `ml/published/`; run by CI. |

## Validation

The raw files are never modified. Problems become *errors* (the session is rejected) or *warnings*
(the session is used and the problem is reported):

- **Errors**: missing files or columns, time columns that disagree, more than 2% of rows unusable,
  more than 1% of timestamps out of order, values that look like m/s² instead of g, invalid label
  values, an unsupported schema version, or a missing pocket side.
- **Warnings**: a few removed rows, duplicate timestamps, gaps, irregular sampling, sampling rate
  or sample count different from the metadata, a median |a| far from 1 g, saturation, no labels,
  steps closer than 50 ms, or steps outside the recording.

Gaps are reported and kept. No samples are interpolated, and every later step respects the gaps.

## Gravity and orientation

A phone in a pocket is tilted in a way that differs between people, trousers and sessions, so
the device axes do not line up with the body. The pipeline therefore computes two kinds of signal:

- **Raw axes** (`accel_x/y/z`) and **raw magnitude** `sqrt(x² + y² + z²)`, which includes about 1 g
  of gravity.
- **Gravity-separated signals**: a 0.5 Hz zero-phase low-pass of each axis estimates gravity.
  Subtracting it gives linear acceleration, which is split into a **vertical** component (along
  gravity) and a **horizontal** magnitude (perpendicular to it). Both are independent of how the
  phone is rotated about gravity. `dynamic_magnitude = |a| − |gravity|` is the gravity-adjusted
  magnitude used for peak detection.

Filters run separately on each gap-free segment, so they never blend data across a pause.
Zero-phase filtering uses future samples. That is fine offline; an on-device version would need
causal filters.

An extra noise filter on the raw axes is available (`preprocessing.noise_filter_cutoff_hz`) and is
off by default.

## Windows and labels

During training and evaluation, step boundaries come **only from the annotated footfall times**.
Each step gets a window from 0.4 s before to 0.4 s after the footfall (configurable). Samples are
selected by time, not by count, so irregular sampling cannot shift a window. A window is rejected
if it runs past the recording, overlaps a gap, or has fewer than 80% of the expected samples. At
normal cadence neighbouring windows overlap; this is recorded in `overlaps_neighbor`.

A step is a binary training target only if it is valid, reviewed, labeled left or right, and has a
complete window. Every other step stays in the step table with its `exclusion_reason`, and training
prints the counts.

## Features

Per window: max/min/mean/std of each raw axis; magnitude mean/std/max and peak-to-peak; prominence and
timing of the main peak; energy (mean squared dynamic magnitude); number of peaks; vertical
max/min/std; horizontal max/mean and horizontal mean before and after the footfall. Context:
time since the previous step and the pocket side.

Peak amplitude is only one feature among many. The pipeline does not assume that a larger peak
means a particular foot.

### Live vs offline features

A real-time model has to decide a step's foot shortly after it lands. The window reaches
`window_after_seconds` (0.4 s) past the footfall, so live predictions arrive with that delay. Two
feature sets are defined:

- `live` (default): everything above. Time since the previous step uses earlier steps only.
- `offline`: adds `time_until_next_step`, which needs the next step. Use it only for offline
  analysis, never for a model meant to run in real time.

## Training target: pocket-relative by default

The phone mostly senses its own leg. In a left-pocket session "left" is the phone's leg; in a
right-pocket session it is the other leg. If the model learns left/right directly from mixed
sessions, the mapping from motion to label flips with the pocket side. That is an XOR, and greedy
tree learners cannot learn it because neither variable is informative on its own. On the synthetic
data a directly trained Random Forest stayed at chance (out-of-fold accuracy 0.45–0.49) even
though the cue was clearly present.

The default `model.target = "pocket_relative"` instead learns *pocket-side leg vs other leg* and
converts to left/right with the pocket side recorded at setup. Because the pocket side is known
before recording, the conversion is exact and the outputs are still `left`, `right` or `unknown`.
`--target absolute` trains on left/right directly, for comparison.

## Splitting without leakage

Neighbouring steps of one person are nearly identical, so a random step-level split would leak
and overstate accuracy. Training therefore:

- splits by **participant** (default) or **session**, so no group is on both sides;
- uses a fixed seed and records the exact groups, sessions, step counts and file fingerprints;
- requires both feet in train and test;
- **stops with an explanation** if there are fewer than `min_groups` (3) groups, instead of falling
  back to a random split;
- refuses to mix synthetic and real sessions in one training set.

Learned preprocessing (the median imputer for missing context features) is fitted inside the
model pipeline on the training split only.

## Confidence, calibration and abstention

A prediction's confidence is its larger class probability. Below `confidence_threshold` (default
0.6) the prediction becomes `unknown`. Evaluation reports coverage (the share of steps given a foot),
the abstention rate, and accuracy on the covered steps.

Random Forest vote fractions are not calibrated probabilities. Training computes out-of-fold
predictions on the **training groups** (GroupKFold) and uses them to:

- report a threshold sweep (coverage vs accuracy), so you can choose a threshold without looking
  at test results;
- optionally fit sigmoid or isotonic calibration (`--calibration sigmoid|isotonic`).

The held-out test labels are never used for calibration or threshold choice. The test Brier score
is reported for information only.

## Evaluation

`evaluate.py` reloads the held-out sessions recorded in the model, checks their fingerprints, and
reports:

- accuracy, balanced accuracy, precision/recall/F1 per foot, macro F1 and the confusion matrix,
  all without abstention;
- the same on covered steps at the confidence threshold, with a confusion matrix that includes
  `unknown`;
- per-session and per-participant results;
- the development target (macro F1 ≥ 0.85), clearly labeled as a target and never marked met on
  synthetic data;
- automatically generated limitations, such as a small test set, a session-level split,
  synthetic data, uncalibrated probabilities, a single pocket side or class imbalance.

`ml/reports/test_predictions.csv` lists every held-out step with its probability and correctness,
for error analysis.

## Step detection is a separate problem

At inference time there are no annotations, so steps come from peak detection on the smoothed
dynamic magnitude (`scipy.signal.find_peaks` with height, prominence and minimum distance).
Detection errors and left/right errors are different failures, so they are measured separately:

- `evaluate.py` scores classification at the **annotated** footfalls.
- `predict.py --evaluate-against-labels` matches detected to annotated steps one-to-one within
  0.1 s and reports detection precision, recall, F1 and timing error. It then reports left/right
  accuracy and coverage on the matched steps only.

Context features at inference use previously **detected** steps, while training uses annotated
ones. Missed or extra detections therefore also affect `time_since_previous_step`.

## Continuous retraining

`.github/workflows/retrain.yml` runs on every push to `main` that changes `ml/data/raw/`,
`ml/src/`, `ml/requirements.txt` or the workflow file. It runs the tests, then
`python -m ml.src.retrain`, and commits `ml/published/` as `github-actions[bot]`.

- Only the workflow writes the published model, so researchers publishing at the same time never
  conflict over it. Runs are serialized and each one checks out the newest `main`.
- Training uses a fixed seed, so any update can be re-run from the data commit named in
  `ml/published/README.md` (results may differ slightly across operating systems).
- With too few participants the run records why it did not train and keeps any previous model.
- Model files are scikit-learn pickles tied to the versions recorded in the training metadata. Load
  them with the same `ml/requirements.txt` environment the workflow uses.

## Adding a 1D CNN later

`models.MODEL_BUILDERS` maps a `model.model_type` name to a builder. A CNN would:

1. take raw or gravity-separated windows from `segmentation.extract_windows` (resampled to a fixed
   length) instead of the feature table;
2. register a builder that returns an object with `fit` and `predict_proba` (for example a
   scikit-learn-compatible wrapper);
3. reuse `grouped_split`, `encode_target`, calibration, thresholds, `ModelBundle` and the
   evaluation code unchanged.

The input assembly in `train.py` (`X_train = train_rows[names]`) is the only place that would branch
on the model type.

## Synthetic data

`ml/data/synthetic/` holds 8 generated sessions (4 participants × 2 sessions) for trying the
pipeline. They model an impact at each footfall, sway toward the stance foot, and a thigh lift
before pocket-side steps, with random phone tilt per session. Results on them only show that the
code works. They are flagged as synthetic in every report and must not be presented as evidence
about real recordings.
