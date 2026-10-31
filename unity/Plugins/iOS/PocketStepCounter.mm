// PocketStep step counter for Unity (iOS). Replaces the original step-counter plugin: it keeps
// StartStepCounter() and GetStepCount() unchanged and adds a per-user threshold.
//
// Detector (unchanged from the original, and mirrored exactly by ml/src/threshold.py):
//   filtered = 0.6 * filtered + 0.4 * |a|      per 60 Hz accelerometer sample, |a| in g with gravity
//   step when filtered > threshold and more than 0.45 s since the previous step
//
// Calibration: call StartStepCalibration(), have the user walk in place for a known number of steps,
// then call FinishStepCalibration(steps). The plugin picks the threshold with the method named in
// threshold_model.json (pass its text to ConfigureStepCalibration first) and starts using it.
// Without a model it uses "count_match": the threshold at which the walk counts exactly `steps`.
//
// Keep only one copy of this file in Assets/Plugins/iOS: the original plugin defines the same
// functions and the two cannot be linked together.

#import <Foundation/Foundation.h>
#if !POCKETSTEP_NO_COREMOTION
#import <CoreMotion/CoreMotion.h>
#endif

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace {

struct Settings {
    double sampleRateHz = 60.0;
    double previousWeight = 0.6;
    double initialG = 1.0;
    double minIntervalSeconds = 0.45;
    int steps = 20;
    double gridMinG = 1.02;
    double gridMaxG = 2.5;
    double gridStepG = 0.01;
};

// Must match FEATURES in ml/src/threshold.py.
const char *const kFeatureNames[] = {"count_match_g", "peak_nth_g", "peak_median_g", "peak_next_g"};
const int kFeatureCount = 4;

struct Model {
    std::string method = "count_match";
    double fixedThresholdG = 1.45;
    double mean[kFeatureCount] = {0, 0, 0, 0};
    double scale[kFeatureCount] = {1, 1, 1, 1};
    double coefficients[kFeatureCount] = {0, 0, 0, 0};
    double intercept = 0;
};

#if !POCKETSTEP_NO_COREMOTION
CMMotionManager *motionManager;
#endif
Settings settings;
Model model;
int stepCount = 0;
double lastPeakTime = -INFINITY;
double filteredMagnitude = 1.0; // low-pass filtered value
double threshold = 1.45;        // the original fixed value until calibrated
bool calibrating = false;
std::vector<double> calibrationTimes;
std::vector<double> calibrationFiltered;
double lastFeatures[kFeatureCount] = {NAN, NAN, NAN, NAN};

double Median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    size_t n = values.size(), mid = n / 2;
    return n % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2.0;
}

int CountSteps(const std::vector<double> &t, const std::vector<double> &f, double thresholdG) {
    int count = 0;
    double last = -INFINITY;
    for (size_t i = 0; i < f.size(); i++) {
        if (f[i] > thresholdG && (t[i] - last) > settings.minIntervalSeconds) {
            count++;
            last = t[i];
        }
    }
    return count;
}

double CountMatchThreshold(const std::vector<double> &t, const std::vector<double> &f, int steps) {
    long n = std::lround((settings.gridMaxG - settings.gridMinG) / settings.gridStepG);
    std::vector<double> grid;
    std::vector<int> diffs;
    int best = INT32_MAX;
    for (long k = 0; k <= n; k++) {
        double g = settings.gridMinG + k * settings.gridStepG;
        int d = std::abs(CountSteps(t, f, g) - steps);
        grid.push_back(g);
        diffs.push_back(d);
        best = std::min(best, d);
    }
    std::vector<double> chosen;
    for (size_t i = 0; i < grid.size(); i++) {
        if (diffs[i] == best) chosen.push_back(grid[i]);
    }
    return Median(chosen);
}

// Heights of the `count` highest local maxima at least minIntervalSeconds apart, highest first.
std::vector<double> CalibrationPeaks(const std::vector<double> &t, const std::vector<double> &f, int count) {
    std::vector<size_t> maxima;
    for (size_t i = 1; i + 1 < f.size(); i++) {
        if (f[i] > f[i - 1] && f[i] >= f[i + 1]) maxima.push_back(i);
    }
    std::stable_sort(maxima.begin(), maxima.end(), [&](size_t a, size_t b) { return f[a] > f[b]; });
    std::vector<double> acceptedTimes, heights;
    for (size_t i : maxima) {
        if ((int)heights.size() == count) break;
        bool farEnough = true;
        for (double ta : acceptedTimes) {
            if (!(std::fabs(t[i] - ta) >= settings.minIntervalSeconds)) { farEnough = false; break; }
        }
        if (farEnough) {
            acceptedTimes.push_back(t[i]);
            heights.push_back(f[i]);
        }
    }
    return heights;
}

void ComputeFeatures(int steps, double out[kFeatureCount]) {
    const std::vector<double> &t = calibrationTimes, &f = calibrationFiltered;
    std::vector<double> heights = CalibrationPeaks(t, f, steps + 1);
    std::vector<double> top(heights.begin(), heights.begin() + std::min((size_t)steps, heights.size()));
    out[0] = CountMatchThreshold(t, f, steps);
    out[1] = top.empty() ? settings.initialG : top.back();
    out[2] = top.empty() ? settings.initialG : Median(top);
    out[3] = (int)heights.size() > steps ? heights[steps] : settings.initialG;
}

double ChooseThreshold(const double features[kFeatureCount]) {
    double value;
    if (model.method == "fixed") {
        value = model.fixedThresholdG;
    } else if (model.method == "model") {
        value = model.intercept;
        for (int j = 0; j < kFeatureCount; j++) {
            value += model.coefficients[j] * ((features[j] - model.mean[j]) / model.scale[j]);
        }
    } else {
        value = features[0];
    }
    return std::min(std::max(value, settings.gridMinG), settings.gridMaxG);
}

bool ReadNumbers(NSArray *array, double out[kFeatureCount]) {
    if (![array isKindOfClass:[NSArray class]] || array.count != kFeatureCount) return false;
    for (int j = 0; j < kFeatureCount; j++) out[j] = [array[j] doubleValue];
    return true;
}

} // namespace

extern "C" {

// Feeds one accelerometer sample (g, seconds). Called by Core Motion; exposed for tests.
void PocketStepProcessSample(double x, double y, double z, double timestamp) {
    double magnitude = sqrt(x * x + y * y + z * z);
    filteredMagnitude = settings.previousWeight * filteredMagnitude + (1.0 - settings.previousWeight) * magnitude;
    if (calibrating) {
        calibrationTimes.push_back(timestamp);
        calibrationFiltered.push_back(filteredMagnitude);
    }
    if (filteredMagnitude > threshold && (timestamp - lastPeakTime) > settings.minIntervalSeconds) {
        stepCount++;
        lastPeakTime = timestamp;
    }
}

void StartStepCounter() {
#if !POCKETSTEP_NO_COREMOTION
    if (!motionManager) {
        motionManager = [[CMMotionManager alloc] init];
    }
    if (motionManager.accelerometerActive) return;
    motionManager.accelerometerUpdateInterval = 1.0 / settings.sampleRateHz;
    [motionManager startAccelerometerUpdatesToQueue:[NSOperationQueue currentQueue]
                                        withHandler:^(CMAccelerometerData *data, NSError *error) {
        if (error) return;
        // data.timestamp is the sample's own time (same clock as CACurrentMediaTime), so queue
        // delays cannot shorten or stretch the cooldown.
        PocketStepProcessSample(data.acceleration.x, data.acceleration.y, data.acceleration.z, data.timestamp);
    }];
#endif
}

void StopStepCounter() {
#if !POCKETSTEP_NO_COREMOTION
    [motionManager stopAccelerometerUpdates];
#endif
}

int GetStepCount() { return stepCount; }

void ResetStepCount() {
    stepCount = 0;
}

// Back to the state at app launch, except the configured model and threshold.
void ResetStepCounterState() {
    stepCount = 0;
    lastPeakTime = -INFINITY;
    filteredMagnitude = settings.initialG;
    calibrating = false;
    calibrationTimes.clear();
    calibrationFiltered.clear();
}

double GetFilteredMagnitude() { return filteredMagnitude; }

void SetStepThreshold(double thresholdG) { threshold = thresholdG; }

double GetStepThreshold() { return threshold; }

// Loads threshold_model.json (its text). Returns 1 on success; on failure nothing changes.
int ConfigureStepCalibration(const char *json) {
    if (!json) return 0;
    NSData *data = [NSData dataWithBytes:json length:strlen(json)];
    NSDictionary *root = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    if (![root isKindOfClass:[NSDictionary class]] ||
        ![root[@"format"] isEqual:@"pocketstep-threshold-model"] || [root[@"version"] intValue] != 1) {
        return 0;
    }
    NSDictionary *detector = root[@"detector"], *calibration = root[@"calibration"], *linear = root[@"model"];
    NSString *method = root[@"recommended_method"];
    if (![detector isKindOfClass:[NSDictionary class]] || ![calibration isKindOfClass:[NSDictionary class]] ||
        ![linear isKindOfClass:[NSDictionary class]] || ![method isKindOfClass:[NSString class]]) {
        return 0;
    }
    NSArray *names = linear[@"features"];
    if (![names isKindOfClass:[NSArray class]] || names.count != kFeatureCount) return 0;
    for (int j = 0; j < kFeatureCount; j++) {
        if (![names[j] isEqual:@(kFeatureNames[j])]) return 0;
    }
    Model next;
    next.method = method.UTF8String;
    if (next.method != "fixed" && next.method != "count_match" && next.method != "model") return 0;
    next.fixedThresholdG = [root[@"fixed_threshold_g"] doubleValue];
    next.intercept = [linear[@"intercept"] doubleValue];
    if (!ReadNumbers(linear[@"feature_mean"], next.mean) || !ReadNumbers(linear[@"feature_scale"], next.scale) ||
        !ReadNumbers(linear[@"coefficients"], next.coefficients)) {
        return 0;
    }
    Settings s;
    s.sampleRateHz = [detector[@"sample_rate_hz"] doubleValue];
    s.previousWeight = [detector[@"filter_previous_weight"] doubleValue];
    s.initialG = [detector[@"filter_initial_g"] doubleValue];
    s.minIntervalSeconds = [detector[@"min_interval_seconds"] doubleValue];
    s.steps = [calibration[@"steps"] intValue];
    s.gridMinG = [calibration[@"grid_min_g"] doubleValue];
    s.gridMaxG = [calibration[@"grid_max_g"] doubleValue];
    s.gridStepG = [calibration[@"grid_step_g"] doubleValue];
    if (s.sampleRateHz <= 0 || s.gridStepG <= 0 || s.gridMaxG < s.gridMinG) return 0;
    model = next;
    settings = s;
    return 1;
}

// Steps the model expects the calibration walk to have (20 unless the model says otherwise).
int GetCalibrationSteps() { return settings.steps; }

void StartStepCalibration() {
    calibrationTimes.clear();
    calibrationFiltered.clear();
    calibrating = true;
}

// Ends the walk and applies the chosen threshold. Returns it, or -1 (threshold unchanged) when the
// walk recorded too little data.
double FinishStepCalibration(int steps) {
    calibrating = false;
    if (steps <= 0 || calibrationTimes.size() < 3) return -1;
    ComputeFeatures(steps, lastFeatures);
    threshold = ChooseThreshold(lastFeatures);
    return threshold;
}

// The last calibration's features, in the order of kFeatureNames. Returns how many were written.
int GetStepCalibrationFeatures(double *out, int capacity) {
    int n = std::min(capacity, kFeatureCount);
    for (int j = 0; j < n; j++) out[j] = lastFeatures[j];
    return n;
}

}
