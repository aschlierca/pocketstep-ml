// Checks unity/Plugins/iOS/PocketStepCounter.mm against values computed by ml/src/threshold.py.
// Run with unity/Tests/run_plugin_tests.sh (macOS, Xcode command line tools).

#import <Foundation/Foundation.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

extern "C" {
void PocketStepProcessSample(double x, double y, double z, double timestamp);
int GetStepCount();
void ResetStepCounterState();
double GetFilteredMagnitude();
void SetStepThreshold(double thresholdG);
int ConfigureStepCalibration(const char *json);
void StartStepCalibration();
double FinishStepCalibration(int steps);
int GetStepCalibrationFeatures(double *out, int capacity);
}

static int failures = 0;

static void Check(bool ok, NSString *what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what.UTF8String);
    if (!ok) failures++;
}

static bool Close(double a, double b) { return std::fabs(a - b) <= 1e-9 * std::max(1.0, std::fabs(b)); }

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s golden.json\n", argv[0]);
            return 2;
        }
        NSData *data = [NSData dataWithContentsOfFile:@(argv[1])];
        NSDictionary *golden = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
        NSArray *samples = golden[@"samples"];
        NSDictionary *expected = golden[@"expected"];
        int steps = [golden[@"steps"] intValue];
        NSArray *features = expected[@"features"];
        Check(samples.count > 0 && expected != nil, @"fixture loaded");

        Check(ConfigureStepCalibration("{\"format\": \"something-else\"}") == 0, @"rejects a foreign JSON file");

        NSDictionary *thresholds = expected[@"thresholds"];
        for (NSString *method in @[@"fixed", @"count_match", @"model"]) {
            NSMutableDictionary *model = [golden[@"model"] mutableCopy];
            model[@"recommended_method"] = method;
            NSData *json = [NSJSONSerialization dataWithJSONObject:model options:0 error:nil];
            NSString *text = [[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding];
            Check(ConfigureStepCalibration(text.UTF8String) == 1, [NSString stringWithFormat:@"%@: model accepted", method]);

            ResetStepCounterState();
            SetStepThreshold(1.45);
            StartStepCalibration();
            for (NSArray *s in samples) {
                PocketStepProcessSample([s[1] doubleValue], [s[2] doubleValue], [s[3] doubleValue], [s[0] doubleValue]);
            }
            Check(GetStepCount() == [expected[@"step_count_at_current_threshold"] intValue],
                  [NSString stringWithFormat:@"%@: step count at 1.45 g = %d", method, GetStepCount()]);
            Check(Close(GetFilteredMagnitude(), [expected[@"filtered_last"] doubleValue]),
                  [NSString stringWithFormat:@"%@: filtered magnitude matches", method]);

            double chosen = FinishStepCalibration(steps);
            double want = [thresholds[method] doubleValue];
            Check(Close(chosen, want), [NSString stringWithFormat:@"%@: threshold %.6f (Python %.6f)", method, chosen, want]);
            double got[4];
            GetStepCalibrationFeatures(got, 4);
            bool same = true;
            for (int j = 0; j < 4; j++) same = same && Close(got[j], [features[j] doubleValue]);
            Check(same, [NSString stringWithFormat:@"%@: calibration features match", method]);
        }

        ResetStepCounterState();
        StartStepCalibration();
        Check(FinishStepCalibration(steps) == -1, @"empty walk leaves the threshold unchanged");
    }
    printf("%s\n", failures ? "FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
}
