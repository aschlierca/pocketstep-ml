# Data collection and labeling protocol

This protocol is for collecting walking-in-place recordings with PocketStep and labeling each step
as left or right. Consistent collection matters more than volume: a model can only be evaluated
fairly if every session follows the same procedure.

## Before collecting any participant data

- **Ethics approval.** Recording people for research usually requires approval from an
  institutional review board or ethics committee, and a consent procedure they have approved. Get
  both before recording anyone other than yourself.
- **Consent, including public release.** Published sessions go into a public GitHub repository,
  where anyone can download them. Use an approved consent form that covers this. Explain what is
  recorded (phone motion only; no location, audio, camera or contacts), that the anonymized
  recording, labels and session details (including the recording date and time) will be public,
  and that published data cannot be fully withdrawn once others have copied it.
- **Anonymous IDs.** Give each participant a code such as `P-7QK2MX` (the app's
  *Generate Anonymous ID* button). Keep any table linking codes to people separate from the data,
  under your institution's rules. Never type names into the app's ID or notes fields.
- **Health and safety.** Walking in place is low risk, but stop if a participant feels unsteady.

## Equipment

- An iPhone running iOS 17 or later with PocketStep installed.
- Trousers with a front pocket that holds the phone snugly. Note the trouser type in the session notes.
- Optional: a second device to film the participant's feet, if your consent procedure covers
  video (see [Optional video for ground truth](#optional-video-for-ground-truth)).

## Session procedure

1. Open PocketStep and tap **+** (New Session).
2. Enter or generate the participant ID. Check the phone model and choose:
   - **Pocket side**: the pocket the phone will go in.
   - **Starting foot**: the foot the participant will step with first.
   - **Target steps**: for example 40.
   - **Walking condition**: slow, comfortable or brisk.
   - **Sampling rate**: keep 100 Hz unless you have a reason to change it.
   - **Countdown**: 5 s gives time to pocket the phone.
3. Read the on-screen checklist with the participant and tap **Continue to Recording**.
4. Tap **Start**. During the countdown, put the phone in the chosen front pocket:
   **screen facing the leg, top of the phone pointing down**. If you have to use another
   orientation, write it in the notes and use the same orientation for the whole session.
   The screen turns off in the pocket, so the pocket cannot press buttons.
5. The participant stands still for about 2 seconds, then walks in place starting with the agreed
   foot until the target count, then stands still again for about 2 seconds.
6. Take the phone out and tap **Stop** → **Stop and Save**.

If something goes wrong (the phone slips, the participant stumbles or talks and turns), either stop
and start a new session, or keep going and later mark the affected steps **invalid**.

### Interruptions

- **Pause/Resume**: no samples are recorded while paused. The pause is saved in the metadata and
  appears as a gap in the data.
- **App leaves the foreground** (side button pressed, a call): recording pauses automatically and
  an interruption is logged. Check the phone position before resuming.
- **App terminated**: everything written up to about one second before is kept, and the session
  is marked `interrupted`.
- **Sensor or file error**: recording stops, the session is marked `failed`, and the data saved so
  far is kept.

### Recommended dataset design

- Several sessions per participant, covering **both pocket sides**, and ideally all three walking
  conditions.
- As many different participants as possible. Training splits by participant, so at least
  3 participants are needed to train at all; 10 or more give a much more trustworthy estimate.
- Alternate the starting foot between sessions.

## Labeling steps

Open a session from the list, then tap **Label Steps**. The chart shows the acceleration magnitude
(about 1 g at rest). Footfalls usually appear as sharp peaks.

Markers show each step's label:

| Marker | Meaning |
|---|---|
| L / R / ? | left / right / unknown |
| dashed line, "Needs review" | the label has not been checked by a person |
| × after the letter, grey | invalid step |
| grey band | pause or gap with no data |

### Manual annotation

1. Press and drag on the chart to choose the moment a foot lands. Use **−10 ms / +10 ms** to adjust,
   and keep **Snap to nearest peak** on unless the peak is clearly not the footfall.
2. Tap **Left**, **Right** or **Unknown** under "Add a step at the selected time".
3. Repeat for every step. Use the step list's target icon to jump to a step on the chart, and the
   sliders icon to edit its time, foot, validity, review state or notes.

To speed this up, choose **Actions → Detect Candidate Steps**. The app adds a step at each
magnitude peak with foot *unknown* and *Needs review*. Delete false detections, add missed steps,
then assign each foot by swiping right (Left/Right) or by editing the step.

### Assisted annotation

Walking in place almost always alternates feet, so the app can propose labels:

1. Make sure every step exists (detect candidates, then fix missing or extra ones).
2. Choose **Actions → Assisted Labeling…**, pick the foot of the first step (defaults to the
   session's starting foot) and, if needed, a later step to restart the pattern from (for example
   after a double step).
3. Tap **Propose**. Unreviewed steps get alternating feet, marked `assisted` and **Needs review**.
   Reviewed steps are never overwritten.
4. Check every proposal against what happened. Confirm it (edit the step and turn on
   *Reviewed*, or select steps and tap **Reviewed**), or choose a different foot. A changed foot
   becomes a `manual` label.

Proposals are not ground truth until a person reviews them. Unreviewed labels are exported with
`reviewed=false`, and the training pipeline ignores them unless explicitly told otherwise.
**Mark All Reviewed** asks for confirmation; only use it after checking every step.

### Uncertain and invalid steps

- **Unknown**: you cannot tell which foot it was. The step stays in the data and is excluded from
  binary training.
- **Invalid**: it was a real step but the signal is unusable (for example, the phone shifted). It
  stays in the sequence, so later steps keep their position, and is excluded from training.
- To mark a run of steps, tap **Select**, select the steps, then use **Set Foot** or
  **Validity** in the bottom bar.
- Delete a marker only if it is not a step at all (a false detection).

### Review and export

The **Session Review** screen lists counts (left, right, unknown, invalid, not yet reviewed, usable for
training) and plain-language warnings, such as labels not reviewed, a count different from the
target, the same foot twice in a row, steps closer than 200 ms, steps outside the recording or
inside pauses, a first step that does not match the starting foot, sensor gaps, and sampling-rate
drift. Warnings never block export, but resolve them where you can.

Tap **Export ZIP** and send the file to your Mac with AirDrop (it lands in `~/Downloads`) or Save
to Files. Session folders are also visible in the Files app under
*On My iPhone → PocketStep → Sessions*. The app itself uploads nothing.

### Publish

From the repository folder on the Mac:

```bash
source .venv/bin/activate
python -m ml.src.ingest            # every S-*.zip in ~/Downloads, or pass ZIP paths
```

For each session, `ingest`:

1. validates it with the same checks as the training pipeline, and rejects failed sessions,
   synthetic data, and participant IDs that are not anonymous codes;
2. shows the label counts, warnings, unreviewed labels and any notes that will become public;
3. asks you to confirm that the participant consented to public release;
4. copies it to `ml/data/raw/<session_id>/`, commits and pushes.

GitHub Actions then retrains the model on all published sessions and updates
`ml/published/README.md`.

Re-exporting a session after fixing labels and running `ingest` again replaces its labels. The
recording itself can never be overwritten. Unreviewed labels are published but not used for
training until they are reviewed and re-exported.

## Optional video for ground truth

PocketStep does not record video. If your approved protocol allows filming feet, video can make
foot labels more reliable:

1. Film the feet only, on a separate device.
2. Synchronise the clocks: after the countdown, have the participant do one sharp heel stomp
   before walking. It shows up as a distinct peak in the chart and is easy to see in the video.
3. While labeling, find each footfall in the video and convert its time to session time using the
   stomp as the reference. Mark the stomp itself as an invalid step, or leave it unlabeled.
4. Store videos separately from the sensor export, under your data-handling plan. They are never
   part of the ZIP and are git-ignored if placed in the repository.
