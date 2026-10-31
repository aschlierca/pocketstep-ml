# Project goal: per-user step-detection thresholds

## The problem

An iOS Unity app detects steps while a user walks in place with an iPhone in their pocket. Its
step-counter plugin reads Core Motion's accelerometer at 60 Hz and counts a step when the
smoothed acceleration magnitude rises above a **threshold**:

```
filtered = 0.6 * filtered + 0.4 * |a|          |a| in g, gravity included (1 g at rest)
step     when filtered > 1.45 g and more than 0.45 s since the previous step
```

One fixed threshold does not work for everyone:

| Walker | What a fixed threshold does |
|---|---|
| Light, soft footfalls, loose pocket | Peaks stay below the threshold, so **steps are missed**. |
| Heavy, bouncy footfalls, tight pocket | The bounce between footfalls also crosses it, so **extra steps are counted**. |
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

## How it works

1. **Data.** Researchers record people walking in place with the PocketStep app and label every
   footfall. Sessions go into the shared dataset in `ml/data/raw/`.
2. **Same detector.** `ml/src/threshold.py` resamples each recording to 60 Hz and runs a line-by-line
   copy of the Unity plugin's detector. `unity/Tests` checks that the plugin and the Python copy
   produce identical results.
3. **Calibration walk.** In each session, the first 20 labeled steps play the role of the
   calibration walk. From it the pipeline measures four features:
   - the threshold at which the walk counts exactly 20 steps (*count match*);
   - the 20th-highest peak (the softest footfall);
   - the median peak;
   - the 21st-highest peak (the largest peak that is probably not a footfall).
4. **Target.** The steps after the walk are scored for every threshold from 1.02 to 2.5 g. The
   threshold with the best detection F1 is that session's *best threshold*.
5. **Model.** Ridge regression learns to predict the best threshold from the walk features.
6. **Honest comparison.** With leave-one-participant-out evaluation, the pipeline compares five
   methods on the steps after each walk:
   - today's 1.45 g;
   - the best single value for the other participants;
   - count match alone;
   - the model;
   - each session's best threshold (an upper limit).
7. **Hand-off.** `ml/published/threshold_model.json` names whichever usable method scored best and
   holds the model's coefficients. GitHub Actions retrains it whenever data is published.
8. **On the phone.** `unity/Plugins/iOS/PocketStepCounter.mm` replaces the step-counter plugin. After
   `StartStepCalibration()` and a 20-step walk, `FinishStepCalibration(20)` computes the same
   features, applies the recommended method and starts using the new threshold. See
   [unity/README.md](../unity/README.md).

## Still to do

- **Real data.** Every result so far is on synthetic walkers. Collect sessions from many different
  people (40+ steps each) and check the comparison in `ml/published/README.md`.
- **Verify in Unity.** Build the plugin into the Unity project and compare its live step counts with
  a recorded calibration session on the same phone.
- **Cooldown.** The plugin's 0.45 s cooldown caps counting at 2.2 steps/s. If brisk walkers are
  undercounted, calibrate the cooldown too (it is a single setting in the model file and in
  `ThresholdConfig`).
- **Ongoing adjustment.** The threshold is set once per calibration. Refining it from the user's
  later walking would need a signal of which detections were right, which the game does not have
  yet.

## Data collection implications

- Record **many different people**, with light and heavy walkers, several cadences, and both pocket
  sides. Variety between people is what the threshold model learns from.
- Label **every** footfall, including soft ones that are easy to miss. Missing labels make a
  threshold look worse than it is.
- Keep *target steps* honest: the count the participant was asked to walk is the reference for
  step-count error.
- Walk **at least 40 steps** per session: the first 20 are the calibration walk and at least 10
  must remain to score the threshold.
