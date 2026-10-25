import Foundation

/// Proposes footfall times from acceleration-magnitude peaks to speed up annotation.
/// Candidates are suggestions only: they carry no foot and are marked unreviewed.
enum StepCandidateDetector {
    struct Parameters {
        /// Moving-average window used to estimate the slowly varying baseline (gravity plus posture).
        var baselineSeconds = 1.0
        /// Short moving-average window that suppresses sample-level noise.
        var smoothingSeconds = 0.05
        /// Minimum peak height above the baseline, in g.
        var minimumHeight = 0.12
        /// Minimum time between two candidates. Walking in place rarely exceeds 3 steps/s.
        var minimumSeparationSeconds = 0.3
    }

    /// Magnitude minus its moving-average baseline, lightly smoothed. Same length as `samples`.
    static func dynamicMagnitude(_ samples: [SensorSample], parameters: Parameters = Parameters()) -> [Double] {
        guard samples.count > 1 else { return samples.map { _ in 0 } }
        let dt = medianInterval(samples)
        let magnitude = samples.map(\.magnitude)
        let baseline = movingAverage(magnitude, window: max(1, Int((parameters.baselineSeconds / dt).rounded())))
        let dynamic = zip(magnitude, baseline).map { $0 - $1 }
        return movingAverage(dynamic, window: max(1, Int((parameters.smoothingSeconds / dt).rounded())))
    }

    static func detect(in samples: [SensorSample], parameters: Parameters = Parameters()) -> [Double] {
        guard samples.count >= 3 else { return [] }
        let signal = dynamicMagnitude(samples, parameters: parameters)

        var peaks: [Int] = []
        for i in 1..<(signal.count - 1)
        where signal[i] >= parameters.minimumHeight && signal[i] > signal[i - 1] && signal[i] >= signal[i + 1] {
            peaks.append(i)
        }

        // Keep the tallest peaks first, dropping any that fall too close to one already kept.
        var accepted: [Int] = []
        for i in peaks.sorted(by: { signal[$0] > signal[$1] }) {
            let t = samples[i].timestamp
            if accepted.allSatisfy({ abs(samples[$0].timestamp - t) >= parameters.minimumSeparationSeconds }) {
                accepted.append(i)
            }
        }
        return accepted.map { samples[$0].timestamp }.sorted()
    }

    /// Time of the highest dynamic-magnitude sample within `radius` seconds of `time`, or `time` if none.
    static func nearestPeak(to time: Double, in samples: [SensorSample], dynamic: [Double], radius: Double = 0.15) -> Double {
        var best: (time: Double, value: Double)?
        for (i, sample) in samples.enumerated() where abs(sample.timestamp - time) <= radius {
            if best == nil || dynamic[i] > best!.value {
                best = (sample.timestamp, dynamic[i])
            }
        }
        return best?.time ?? time
    }

    static func medianInterval(_ samples: [SensorSample]) -> Double {
        let intervals = zip(samples.dropFirst(), samples).map { $0.timestamp - $1.timestamp }.filter { $0 > 0 }.sorted()
        guard !intervals.isEmpty else { return 0.01 }
        return intervals[intervals.count / 2]
    }

    /// Centered moving average with shrinking windows at the edges.
    static func movingAverage(_ values: [Double], window: Int) -> [Double] {
        guard window > 1, !values.isEmpty else { return values }
        var prefix = [0.0]
        prefix.reserveCapacity(values.count + 1)
        for v in values { prefix.append(prefix.last! + v) }
        let half = window / 2
        return values.indices.map { i in
            let lo = max(0, i - half)
            let hi = min(values.count, i + half + 1)
            return (prefix[hi] - prefix[lo]) / Double(hi - lo)
        }
    }
}
