// C# access to Plugins/iOS/PocketStepCounter.mm. Copy into Assets/Scripts (any folder works).
//
//   PocketStepCounter.Start();
//   PocketStepCounter.LoadModel(thresholdModelAsset.text);   // threshold_model.json as a TextAsset
//   PocketStepCounter.BeginCalibration();                    // user walks in place CalibrationSteps steps
//   double g = PocketStepCounter.FinishCalibration(PocketStepCounter.CalibrationSteps);
//   PlayerPrefs.SetFloat("stepThreshold", (float)g);          // reuse next launch via Threshold = ...
//
// Outside an iOS device build every call is a no-op so the project still runs in the Editor.

using System.Runtime.InteropServices;

public static class PocketStepCounter
{
#if UNITY_IOS && !UNITY_EDITOR
    [DllImport("__Internal")] private static extern void StartStepCounter();
    [DllImport("__Internal")] private static extern void StopStepCounter();
    [DllImport("__Internal")] private static extern int GetStepCount();
    [DllImport("__Internal")] private static extern void ResetStepCount();
    [DllImport("__Internal")] private static extern void SetStepThreshold(double thresholdG);
    [DllImport("__Internal")] private static extern double GetStepThreshold();
    [DllImport("__Internal")] private static extern int ConfigureStepCalibration(string json);
    [DllImport("__Internal")] private static extern int GetCalibrationSteps();
    [DllImport("__Internal")] private static extern void StartStepCalibration();
    [DllImport("__Internal")] private static extern double FinishStepCalibration(int steps);
#else
    private static double editorThreshold = 1.45;
    private static void StartStepCounter() { }
    private static void StopStepCounter() { }
    private static int GetStepCount() { return 0; }
    private static void ResetStepCount() { }
    private static void SetStepThreshold(double thresholdG) { editorThreshold = thresholdG; }
    private static double GetStepThreshold() { return editorThreshold; }
    private static int ConfigureStepCalibration(string json) { return 0; }
    private static int GetCalibrationSteps() { return 20; }
    private static void StartStepCalibration() { }
    private static double FinishStepCalibration(int steps) { return -1; }
#endif

    public static void Start() { StartStepCounter(); }

    public static void Stop() { StopStepCounter(); }

    public static int StepCount { get { return GetStepCount(); } }

    public static void ResetCount() { ResetStepCount(); }

    /// Threshold in g of filtered acceleration magnitude (gravity included). 1.45 until calibrated.
    public static double Threshold
    {
        get { return GetStepThreshold(); }
        set { SetStepThreshold(value); }
    }

    /// Loads threshold_model.json. Returns false if the text is not a supported model; calibration
    /// then falls back to matching the known step count.
    public static bool LoadModel(string json) { return ConfigureStepCalibration(json) == 1; }

    /// Steps the loaded model was trained with; ask the user to walk this many.
    public static int CalibrationSteps { get { return GetCalibrationSteps(); } }

    public static void BeginCalibration() { StartStepCalibration(); }

    /// Applies and returns the user's threshold, or -1 (threshold unchanged) if the walk was too short.
    public static double FinishCalibration(int steps) { return FinishStepCalibration(steps); }
}
