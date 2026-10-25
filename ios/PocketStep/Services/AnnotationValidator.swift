import Foundation

struct AnnotationSummary: Equatable {
    var total = 0
    var left = 0
    var right = 0
    var unknown = 0
    var invalid = 0
    var unreviewed = 0
    var assisted = 0
    var manual = 0
    /// Valid, reviewed left/right labels: the ones the training pipeline uses by default.
    var usableForTraining = 0

    init(labels: [StepLabel]) {
        total = labels.count
        for label in labels {
            switch label.footLabel {
            case .left: left += 1
            case .right: right += 1
            case .unknown: unknown += 1
            }
            if !label.isValid { invalid += 1 }
            if !label.reviewed { unreviewed += 1 }
            if label.labelSource == .assisted { assisted += 1 } else { manual += 1 }
            if label.isValid && label.reviewed && label.footLabel != .unknown { usableForTraining += 1 }
        }
    }
}

/// Plain-language checks for incomplete or inconsistent annotations. Warnings never block export.
enum AnnotationValidator {
    static let minimumStepSpacing = 0.2

    static func warnings(metadata: SessionMetadata, labels: [StepLabel],
                         sampleTimeRange: ClosedRange<Double>?, malformedRows: Int = 0) -> [String] {
        var warnings: [String] = []
        let sorted = StepLabelStore.normalized(labels)
        let summary = AnnotationSummary(labels: sorted)

        if metadata.recordingStatus != .completed {
            warnings.append("The recording ended as \"\(metadata.recordingStatus.displayName.lowercased())\". Check that the data you need was saved.")
        }
        if sampleTimeRange == nil {
            warnings.append("No sensor samples could be read for this session.")
        }
        if malformedRows > 0 {
            warnings.append("\(malformedRows) row(s) of sensor_data.csv could not be read.")
        }
        if metadata.unexpectedGapCount > 0 {
            warnings.append("\(metadata.unexpectedGapCount) unexpected gap(s) in the sensor stream. Steps near gaps may be rejected during training.")
        }
        if let measured = metadata.samplingIntervalSeconds,
           abs(measured - metadata.requestedSamplingIntervalSeconds) / metadata.requestedSamplingIntervalSeconds > 0.1 {
            warnings.append(String(format: "Measured sampling rate (%.1f Hz) differs from the requested %.0f Hz by more than 10%%.",
                                   1 / measured, 1 / metadata.requestedSamplingIntervalSeconds))
        }

        if sorted.isEmpty {
            warnings.append("No steps are annotated yet.")
            return warnings
        }
        if summary.unreviewed > 0 {
            warnings.append("\(summary.unreviewed) label(s) have not been reviewed. They are exported with reviewed=false and the training pipeline skips them by default.")
        }
        let labeledSteps = summary.left + summary.right
        if labeledSteps != metadata.targetSteps {
            warnings.append("\(labeledSteps) step(s) are labeled left or right; the target was \(metadata.targetSteps).")
        }
        if summary.unknown > 0 {
            warnings.append("\(summary.unknown) step(s) are labeled unknown.")
        }

        let firstFooted = sorted.first { $0.footLabel != .unknown && $0.isValid }
        if let first = firstFooted, first.footLabel != FootLabel(metadata.startingFoot) {
            warnings.append("Step \(first.stepId) is the first labeled step but is \(first.footLabel.rawValue); the session's starting foot is \(metadata.startingFoot.rawValue).")
        }

        let repeats = zip(sorted, sorted.dropFirst()).filter { a, b in
            a.footLabel != .unknown && a.footLabel == b.footLabel
        }
        if !repeats.isEmpty {
            let ids = repeats.prefix(5).map { "\($0.1.stepId)" }.joined(separator: ", ")
            warnings.append("The same foot is labeled twice in a row \(repeats.count) time(s) (at step \(ids)\(repeats.count > 5 ? ", …" : "")). Walking in place normally alternates; check these steps.")
        }

        let crowded = zip(sorted, sorted.dropFirst()).filter { $1.timestamp - $0.timestamp < minimumStepSpacing }
        if !crowded.isEmpty {
            warnings.append("\(crowded.count) pair(s) of steps are less than \(Int(minimumStepSpacing * 1000)) ms apart, which may be duplicates.")
        }

        if let range = sampleTimeRange {
            let outside = sorted.filter { !range.contains($0.timestamp) }
            if !outside.isEmpty {
                warnings.append("\(outside.count) step(s) fall outside the recorded time range.")
            }
        }

        let inPause = sorted.filter { label in
            metadata.pauses.contains { pause in
                label.timestamp >= pause.startSeconds && label.timestamp <= (pause.endSeconds ?? .infinity)
            }
        }
        if !inPause.isEmpty {
            warnings.append("\(inPause.count) step(s) fall inside a pause, where there is no sensor data.")
        }
        return warnings
    }
}
