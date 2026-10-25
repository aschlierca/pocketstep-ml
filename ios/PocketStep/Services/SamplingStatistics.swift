import Foundation

/// Tracks the real spacing of incoming samples so the metadata reports the measured rate
/// instead of the requested one, and so dropped samples show up as gaps rather than being hidden.
struct SamplingStatistics {
    let requestedInterval: Double
    /// An interval longer than this multiple of the requested interval counts as a gap.
    let gapFactor: Double

    private(set) var sampleCount = 0
    private(set) var gapCount = 0
    private(set) var largestGapSeconds = 0.0
    private var intervals: [Double] = []
    private var lastTimestamp: Double?

    init(requestedInterval: Double, gapFactor: Double = 3) {
        self.requestedInterval = requestedInterval
        self.gapFactor = gapFactor
    }

    mutating func add(timestamp: Double) {
        sampleCount += 1
        defer { lastTimestamp = timestamp }
        guard let last = lastTimestamp else { return }
        let interval = timestamp - last
        guard interval > 0 else { return }
        if interval > requestedInterval * gapFactor {
            gapCount += 1
            largestGapSeconds = max(largestGapSeconds, interval)
        } else {
            intervals.append(interval)
        }
    }

    /// Call when recording pauses so the deliberate pause is not counted as a gap.
    mutating func breakContinuity() {
        lastTimestamp = nil
    }

    var medianInterval: Double? {
        guard !intervals.isEmpty else { return nil }
        let sorted = intervals.sorted()
        let mid = sorted.count / 2
        return sorted.count % 2 == 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
    }

    var estimatedRateHz: Double? {
        medianInterval.map { 1 / $0 }
    }
}
