# Project goal: per-user step-detection thresholds

## The problem

An iOS Unity app detects steps while a user walks in place with an iPhone in their pocket. Its
step detection compares the accelerometer signal with a **threshold**: a step is counted when
the signal rises above it.

One fixed threshold does not work for everyone:

| Walker | What a fixed threshold does |
|---|---|
| Light, soft footfalls, loose pocket | Peaks stay below the threshold, so **steps are missed**. |
| Heavy, bouncy footfalls, tight pocket | One footfall produces several peaks above the threshold, so **extra steps are counted**. |
| Different cadence (slow / brisk) | Peak heights and spacing change, so the best threshold shifts too. |

A value tuned on the developers themselves can be wrong for the next user, and asking users to
tune a number by hand is not practical.

## The goal

Let the app **choose the threshold for each user automatically**:

1. **Calibration walk.** The user walks in place for a known number of steps (for example 20).
2. **Estimate.** A model, trained on recordings of many different people, takes that walk's
   accelerometer signal and estimates the threshold that would detect this user's steps best.
3. **Apply and adjust.** The app uses the estimated threshold. It can repeat the calibration, or
   refine the value as more of the user's walking is observed.

Success means step counts that match the steps actually taken across many kinds of walkers.
The model is judged against **a single fixed threshold** (today's behavior) and against each
person's **best possible threshold**, the upper limit for any per-user method.

## What already exists in this repository

| Need | Where it is |
|---|---|
| Recordings of people walking in place for a known number of steps | iOS app: session setup with *target steps*, recording at 100 Hz (`ios/`) |
| Ground truth: when each step actually happened | Labeled footfall times in `step_labels.csv`, with validity and review state (labeling screen) |
| Many different people | Shared public dataset in `ml/data/raw/`, added with `python -m ml.src.ingest` |
| Varied conditions | Session metadata: pocket side, walking condition (slow / comfortable / brisk), phone model |
| A threshold-based step detector to tune | `segmentation.detect_candidate_steps` (peak height, prominence, minimum spacing) |
| Measuring how well a threshold detects steps | `segmentation.match_steps` (precision, recall, F1, timing error) and `python -m ml.src.predict --evaluate-against-labels` |
| Fair evaluation across people | Participant-grouped train/test splits in `ml/src/train.py`; automatic retraining in `ml/published/` |

The existing left/right classifier is a related research question. Its training and evaluation
setup carries over to the threshold model.

## What is not built yet

- **Per-session best threshold.** Sweep the detector's threshold on each labeled session and
  record the value with the best step-detection F1 (the target the model learns).
- **Threshold model.** A regressor from calibration-walk features (for example peak-height
  percentiles, cadence, signal energy, pocket side) to the best threshold, trained and evaluated
  with participant-grouped splits.
- **Evaluation against the baselines.** Step-count error and detection F1 with the predicted
  threshold, compared with the best single fixed threshold and with each person's best threshold.
- **Unity hand-off.** A format the Unity app can use, such as an exported threshold
  formula or lookup table, or a small on-device model.

## Open questions about the Unity app

The calibration only transfers to the app if it tunes **the same detector the app uses**. The
details below need to be confirmed and mirrored in the Python detector:

- Which signal is thresholded: raw acceleration magnitude, gravity-removed magnitude, or one axis?
  In what units (g or m/s²)? Unity's `Input.acceleration` reports g.
- Is the threshold a peak height, a rise above a baseline, or a prominence?
- Is there a minimum time between steps, smoothing, or a sampling rate the app runs at?
- How does the app use the calibration walk: a fixed number of steps, a fixed duration, or both?

## Data collection implications

- Record **many different people**, with light and heavy walkers, several cadences, and both pocket
  sides. Variety between people is what the threshold model learns from.
- Label **every** footfall, including soft ones that are easy to miss. Missing labels make a
  threshold look worse than it is.
- Keep *target steps* honest: the count the participant was asked to walk is the reference for
  step-count error.
