# Unity step counter with per-user threshold

The Unity app counts steps with Core Motion and a threshold. These files keep that detector
unchanged but let each user's threshold be set by a short calibration walk. The trained model from
this repository picks the threshold.

| File | Copy to (Unity project) | What it is |
|---|---|---|
| `Plugins/iOS/PocketStepCounter.mm` | `Assets/Plugins/iOS/` | Replaces the existing step-counter plugin. Same `StartStepCounter()` / `GetStepCount()`, plus threshold and calibration functions. |
| `Scripts/PocketStepCounter.cs` | `Assets/Scripts/` | C# wrapper (`PocketStepCounter.Start()`, `.StepCount`, `.Threshold`, calibration). No-ops in the Editor. |
| `ml/published/threshold_model.json` (repo root) | `Assets/Resources/` (or anywhere a `TextAsset` can be loaded) | The trained model. Retrained automatically when new data is published. |

Delete the old step-counter `.mm` when you add this one: both define `StartStepCounter` and
`GetStepCount`, so the build fails to link if both are present. Existing C# code that imports
those two functions keeps working. The device-motion plugin (pitch/roll/yaw) is unrelated and stays.

## Detector

Unchanged from the original plugin, and copied exactly in `ml/src/threshold.py`:

```
filtered = 0.6 * filtered + 0.4 * |a|     every accelerometer sample (60 Hz), |a| in g including gravity
step     when filtered > threshold and more than 0.45 s since the previous step
```

One change: the cooldown uses each sample's own timestamp (`CMAccelerometerData.timestamp`) instead
of `CACurrentMediaTime()` when the handler runs. Both use the same clock; the sample timestamp just
isn't affected by queue delays.

## Calibration flow

```csharp
public TextAsset thresholdModel;           // threshold_model.json

void Start()
{
    PocketStepCounter.Start();
    PocketStepCounter.LoadModel(thresholdModel.text);
    if (PlayerPrefs.HasKey("stepThreshold"))
        PocketStepCounter.Threshold = PlayerPrefs.GetFloat("stepThreshold");
}

// Calibration screen: "Walk in place for 20 steps, then tap Done."
public void OnCalibrationStart() => PocketStepCounter.BeginCalibration();

public void OnCalibrationDone()
{
    double g = PocketStepCounter.FinishCalibration(PocketStepCounter.CalibrationSteps);
    if (g > 0) PlayerPrefs.SetFloat("stepThreshold", (float)g);
}
```

- Ask for exactly `PocketStepCounter.CalibrationSteps` steps (20 by default). The model was trained
  on walks of that length.
- Start calibration just before the user starts walking and finish right after the last step. Long
  idle time before or after has little effect, but any extra movement (picking the phone up, sitting
  down) counts as steps.
- `FinishCalibration` applies the threshold immediately and returns it. It returns -1 and keeps the
  old threshold if almost no data was recorded.
- Without `LoadModel` (or if the JSON is rejected), calibration still works. It picks the threshold
  at which the walk counts exactly the requested number of steps (`count_match`).

The model file names the method that scored best in evaluation (`recommended_method`): `fixed` (one
threshold for everyone), `count_match`, or `model` (a linear formula over four calibration
features). The plugin applies whichever it names, so updating the JSON is enough when the model is
retrained.

## Checking the plugin against Python

```bash
unity/Tests/run_plugin_tests.sh
```

This builds the plugin for macOS without Core Motion, feeds it a recorded-looking walk from
`Tests/golden.json`, and checks that the filtered signal, step count, calibration features and chosen
threshold match `ml/src/threshold.py` for every method. `pytest` runs it automatically on macOS.
After changing the Python detector, regenerate the fixture with `python -m ml.src.threshold golden`.
