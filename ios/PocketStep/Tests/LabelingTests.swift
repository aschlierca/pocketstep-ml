import XCTest
@testable import PocketStep

/// Builds a 100 Hz signal resting at 1 g with short bumps at the given times.
private func syntheticSamples(duration: Double, bumps: [Double], height: Double = 0.5) -> [SensorSample] {
    stride(from: 0.0, to: duration, by: 0.01).map { t in
        let bump = bumps.reduce(0.0) { $0 + height * exp(-pow((t - $1) / 0.02, 2)) }
        return SensorSample(timestamp: t, x: 0, y: 0, z: -(1 + bump))
    }
}

final class StepCandidateDetectorTests: XCTestCase {
    func testFindsBumpsAtTheirTimes() {
        let bumps = [1.0, 1.55, 2.1, 2.65, 3.2]
        let detected = StepCandidateDetector.detect(in: syntheticSamples(duration: 4.5, bumps: bumps))
        XCTAssertEqual(detected.count, bumps.count)
        for (found, expected) in zip(detected, bumps) {
            XCTAssertEqual(found, expected, accuracy: 0.03)
        }
    }

    func testIgnoresFlatSignalAndEnforcesSeparation() {
        XCTAssertTrue(StepCandidateDetector.detect(in: syntheticSamples(duration: 3, bumps: [])).isEmpty)
        let close = StepCandidateDetector.detect(in: syntheticSamples(duration: 3, bumps: [1.0, 1.1]))
        XCTAssertEqual(close.count, 1)
    }

    func testNearestPeakSnapsWithinRadius() {
        let samples = syntheticSamples(duration: 3, bumps: [1.5])
        let dynamic = StepCandidateDetector.dynamicMagnitude(samples)
        XCTAssertEqual(StepCandidateDetector.nearestPeak(to: 1.42, in: samples, dynamic: dynamic), 1.5, accuracy: 0.011)
        // With nothing to snap to, the requested time is kept.
        XCTAssertEqual(StepCandidateDetector.nearestPeak(to: 2.9, in: samples, dynamic: dynamic, radius: 0), 2.9, accuracy: 0.011)
    }
}

final class AssistedLabelerTests: XCTestCase {
    func testAlternatesFromStartingFootAndMarksUnreviewed() {
        let labels = (0..<4).map { StepLabel(timestamp: Double($0), reviewed: false) }
        let proposed = AssistedLabeler.propose(for: labels, startingFoot: .right)
        XCTAssertEqual(proposed.map(\.footLabel), [.right, .left, .right, .left])
        XCTAssertTrue(proposed.allSatisfy { $0.labelSource == .assisted && !$0.reviewed })
    }

    func testNeverOverwritesReviewedLabelsAndCanRestartPattern() {
        var labels = (0..<5).map { StepLabel(timestamp: Double($0), reviewed: false) }
        labels[1] = StepLabel(timestamp: 1, footLabel: .left, labelSource: .manual, reviewed: true)
        let proposed = AssistedLabeler.propose(for: labels, startingFoot: .left, fromIndex: 1)
        XCTAssertEqual(proposed[0].footLabel, .unknown, "steps before fromIndex are unchanged")
        XCTAssertEqual(proposed[1].labelSource, .manual)
        XCTAssertEqual(proposed.map(\.footLabel), [.unknown, .left, .right, .left, .right])
    }
}

final class AnnotationValidatorTests: XCTestCase {
    private func metadata(target: Int = 4) -> SessionMetadata {
        var m = SessionMetadata(
            participantId: "P1", sessionId: "S", phoneModel: "iPhone", iosVersion: "18",
            pocketSide: .right, startingFoot: .left, targetSteps: target, walkingCondition: .comfortable,
            samplingIntervalSeconds: 0.01, requestedSamplingIntervalSeconds: 0.01, notes: "")
        m.recordingStatus = .completed
        return m
    }

    func testCleanAnnotationHasNoWarnings() {
        let labels = [Side.left, .right, .left, .right].enumerated().map {
            StepLabel(timestamp: 1 + Double($0.offset) * 0.5, footLabel: FootLabel($0.element))
        }
        XCTAssertEqual(AnnotationValidator.warnings(metadata: metadata(), labels: labels, sampleTimeRange: 0...10), [])
    }

    func testReportsInconsistencies() {
        let labels = [
            StepLabel(timestamp: 1.0, footLabel: .right),
            StepLabel(timestamp: 1.1, footLabel: .right),
            StepLabel(timestamp: 2.0, footLabel: .left, labelSource: .assisted, reviewed: false),
            StepLabel(timestamp: 20.0, footLabel: .unknown),
        ]
        let warnings = AnnotationValidator.warnings(metadata: metadata(), labels: labels, sampleTimeRange: 0...10)
        let text = warnings.joined(separator: "\n")
        XCTAssertTrue(text.contains("not been reviewed"))
        XCTAssertTrue(text.contains("the target was 4"))
        XCTAssertTrue(text.contains("starting foot is left"))
        XCTAssertTrue(text.contains("twice in a row"))
        XCTAssertTrue(text.contains("less than 200 ms apart"))
        XCTAssertTrue(text.contains("outside the recorded time range"))
        XCTAssertTrue(text.contains("labeled unknown"))
    }

    func testSummaryCounts() {
        let summary = AnnotationSummary(labels: [
            StepLabel(timestamp: 0, footLabel: .left),
            StepLabel(timestamp: 1, footLabel: .right, isValid: false),
            StepLabel(timestamp: 2, footLabel: .right, labelSource: .assisted, reviewed: false),
            StepLabel(timestamp: 3, footLabel: .unknown),
        ])
        XCTAssertEqual(summary.total, 4)
        XCTAssertEqual(summary.left, 1)
        XCTAssertEqual(summary.right, 2)
        XCTAssertEqual(summary.unknown, 1)
        XCTAssertEqual(summary.invalid, 1)
        XCTAssertEqual(summary.unreviewed, 1)
        XCTAssertEqual(summary.usableForTraining, 1)
    }
}

final class SensorDataReaderTests: XCTestCase {
    func testParsesAndCountsMalformedRows() throws {
        let text = """
        timestamp_seconds,elapsed_seconds,accel_x,accel_y,accel_z
        0.000000,0.000000,0.1,0.2,-0.9
        0.010000,0.010000,oops,0.2,-0.9
        0.020000,0.020000,0.1,0.2,-0.9

        """
        let result = try SensorDataReader.parse(text)
        XCTAssertEqual(result.samples.count, 2)
        XCTAssertEqual(result.malformedRowCount, 1)
        XCTAssertEqual(result.samples[1].timestamp, 0.02)
    }

    func testRejectsMissingColumns() {
        XCTAssertThrowsError(try SensorDataReader.parse("timestamp_seconds,accel_x\n0,1\n"))
    }
}

final class SessionExporterTests: XCTestCase {
    func testZipContainsTheThreeSessionFiles() throws {
        let fm = FileManager.default
        let directory = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: directory) }
        try Data(SensorCSVWriter.header.utf8).write(to: directory.appendingPathComponent(SessionFiles.sensorData))
        try Data(StepLabelStore.header.utf8).write(to: directory.appendingPathComponent(SessionFiles.stepLabels))
        try Data("{}".utf8).write(to: directory.appendingPathComponent(SessionFiles.metadata))
        try Data("private".utf8).write(to: directory.appendingPathComponent("unrelated.txt"))

        let zip = try SessionExporter.makeZip(sessionDirectory: directory, sessionId: "S-TEST")
        defer { try? fm.removeItem(at: zip.deletingLastPathComponent()) }
        XCTAssertEqual(zip.lastPathComponent, "S-TEST.zip")

        let data = try Data(contentsOf: zip)
        XCTAssertEqual(Array(data.prefix(4)), [0x50, 0x4B, 0x03, 0x04], "ZIP local file header signature")
        let names = String(decoding: data, as: UTF8.self)
        for name in SessionExporter.exportedFiles {
            XCTAssertTrue(names.contains("S-TEST/\(name)"), "missing \(name)")
        }
        XCTAssertFalse(names.contains("unrelated.txt"))
    }

    func testMissingFileIsReported() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertThrowsError(try SessionExporter.makeZip(sessionDirectory: directory, sessionId: "S"))
    }
}
