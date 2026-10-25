import Foundation

struct ChartPoint: Identifiable {
    let id: Int
    let time: Double
    let magnitude: Double
}

struct TimeSpan: Identifiable {
    let id: Int
    let start: Double
    let end: Double
}

/// Loads one session's sensor data and labels, applies edits, and saves `step_labels.csv`
/// after every change so no annotation work is lost.
@MainActor
final class LabelingModel: ObservableObject {
    @Published private(set) var labels: [StepLabel] = []
    @Published private(set) var chartPoints: [ChartPoint] = []
    @Published private(set) var gaps: [TimeSpan] = []
    @Published private(set) var metadata: SessionMetadata?
    @Published private(set) var isLoaded = false
    @Published var errorMessage: String?

    let sessionId: String
    private let store: SessionStore
    private var samples: [SensorSample] = []
    private var dynamic: [Double] = []

    /// Snap radius when placing a step near a peak.
    static let snapRadius = 0.15
    /// Candidates closer than this to an existing step are treated as the same step.
    static let duplicateRadius = 0.15

    init(sessionId: String, store: SessionStore) {
        self.sessionId = sessionId
        self.store = store
    }

    private var labelsURL: URL { store.fileURL(SessionFiles.stepLabels, for: sessionId) }

    /// Height at which step letters are drawn: just above the tallest plotted sample.
    var markerRowY: Double {
        (chartPoints.map(\.magnitude).max() ?? 1) + 0.15
    }

    var timeRange: ClosedRange<Double>? {
        guard let first = samples.first?.timestamp, let last = samples.last?.timestamp else { return nil }
        return first...last
    }

    func load() async {
        defer { isLoaded = true }
        do {
            metadata = try store.loadMetadata(sessionId: sessionId)
            labels = StepLabelStore.normalized(try StepLabelStore.read(from: labelsURL))
        } catch {
            errorMessage = "Could not open this session: \(error.localizedDescription)"
            return
        }
        let sensorURL = store.fileURL(SessionFiles.sensorData, for: sessionId)
        let pauses = metadata?.pauses ?? []
        do {
            let loaded = try await Task.detached(priority: .userInitiated) {
                let result = try SensorDataReader.read(from: sensorURL)
                let dynamic = StepCandidateDetector.dynamicMagnitude(result.samples)
                return (result, dynamic, LabelingModel.downsample(result.samples),
                        LabelingModel.findGaps(result.samples, pauses: pauses))
            }.value
            samples = loaded.0.samples
            dynamic = loaded.1
            chartPoints = loaded.2
            gaps = loaded.3
            if samples.isEmpty {
                errorMessage = "This session has no sensor samples to label."
            } else if loaded.0.malformedRowCount > 0 {
                errorMessage = "\(loaded.0.malformedRowCount) unreadable row(s) in sensor_data.csv were skipped."
            }
        } catch {
            errorMessage = "Could not read sensor data: \(error.localizedDescription)"
        }
    }

    // MARK: Editing

    func snapped(_ time: Double) -> Double {
        StepCandidateDetector.nearestPeak(to: time, in: samples, dynamic: dynamic, radius: Self.snapRadius)
    }

    func addStep(at time: Double, foot: FootLabel, snapToPeak: Bool) {
        let t = snapToPeak ? snapped(time) : time
        commit(labels + [StepLabel(timestamp: t, footLabel: foot, labelSource: .manual, reviewed: true)])
    }

    /// Adds peak-detected candidates that do not duplicate existing steps. Returns how many were added.
    @discardableResult
    func detectCandidates() -> Int {
        let times = StepCandidateDetector.detect(in: samples)
        let new = times.filter { t in !labels.contains { abs($0.timestamp - t) < Self.duplicateRadius } }
        commit(labels + new.map { StepLabel(timestamp: $0, footLabel: .unknown, labelSource: .manual, reviewed: false) })
        return new.count
    }

    func applyAssisted(startingFoot: Side, fromIndex: Int) {
        commit(AssistedLabeler.propose(for: labels, startingFoot: startingFoot, fromIndex: fromIndex))
    }

    func replace(_ label: StepLabel) {
        commit(labels.map { $0.id == label.id ? label : $0 })
    }

    /// Choosing a foot is a human decision: it marks the label reviewed. Confirming an assisted
    /// proposal keeps the assisted source; overriding it makes the label manual.
    func setFoot(_ foot: FootLabel, for ids: Set<UUID>) {
        commit(labels.map { label in
            guard ids.contains(label.id) else { return label }
            var copy = label
            if !(label.labelSource == .assisted && label.footLabel == foot) {
                copy.footLabel = foot
                copy.labelSource = .manual
            }
            copy.reviewed = true
            return copy
        })
    }

    func setValid(_ isValid: Bool, for ids: Set<UUID>) {
        commit(labels.map { label in
            guard ids.contains(label.id) else { return label }
            var copy = label
            copy.isValid = isValid
            return copy
        })
    }

    func markReviewed(_ ids: Set<UUID>) {
        commit(labels.map { label in
            guard ids.contains(label.id) else { return label }
            var copy = label
            copy.reviewed = true
            return copy
        })
    }

    func delete(_ ids: Set<UUID>) {
        commit(labels.filter { !ids.contains($0.id) })
    }

    private func commit(_ newLabels: [StepLabel]) {
        labels = StepLabelStore.normalized(newLabels)
        do {
            try StepLabelStore.write(labels, to: labelsURL)
        } catch {
            errorMessage = "Could not save labels: \(error.localizedDescription)"
        }
    }

    // MARK: Chart preparation

    /// Keeps the minimum and maximum of each bucket so peaks survive downsampling.
    nonisolated static func downsample(_ samples: [SensorSample], maxBuckets: Int = 3000) -> [ChartPoint] {
        guard samples.count > maxBuckets * 2 else {
            return samples.enumerated().map { ChartPoint(id: $0.offset, time: $0.element.timestamp, magnitude: $0.element.magnitude) }
        }
        let bucketSize = Int((Double(samples.count) / Double(maxBuckets)).rounded(.up))
        var points: [ChartPoint] = []
        points.reserveCapacity(maxBuckets * 2)
        var start = 0
        while start < samples.count {
            let bucket = samples[start..<min(start + bucketSize, samples.count)]
            let low = bucket.min { $0.magnitude < $1.magnitude }!
            let high = bucket.max { $0.magnitude < $1.magnitude }!
            for s in [low, high].sorted(by: { $0.timestamp < $1.timestamp }) {
                points.append(ChartPoint(id: points.count, time: s.timestamp, magnitude: s.magnitude))
            }
            start += bucketSize
        }
        return points
    }

    /// Pauses plus any unexpected break longer than three median intervals.
    nonisolated static func findGaps(_ samples: [SensorSample], pauses: [PauseInterval]) -> [TimeSpan] {
        var spans = pauses.map { ($0.startSeconds, $0.endSeconds ?? $0.startSeconds) }
        let limit = StepCandidateDetector.medianInterval(samples) * 3
        for (a, b) in zip(samples, samples.dropFirst()) where b.timestamp - a.timestamp > limit {
            spans.append((a.timestamp, b.timestamp))
        }
        return spans.sorted { $0.0 < $1.0 }.enumerated().map { TimeSpan(id: $0.offset, start: $0.element.0, end: $0.element.1) }
    }
}
