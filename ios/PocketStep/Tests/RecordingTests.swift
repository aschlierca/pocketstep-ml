import XCTest
@testable import PocketStep

final class SensorCSVWriterTests: XCTestCase {
    private var url: URL!

    override func setUpWithError() throws {
        url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).csv")
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: url)
    }

    func testWritesHeaderAndRowsWithIdenticalTimeColumns() throws {
        let writer = try SensorCSVWriter(url: url, flushThreshold: 2)
        try writer.append(SensorSample(timestamp: 0.01, x: 0.1, y: -0.2, z: -0.98))
        try writer.append(SensorSample(timestamp: 0.02, x: 0.11, y: -0.21, z: -0.97))
        try writer.close()

        let lines = try String(contentsOf: url, encoding: .utf8).split(separator: "\n")
        XCTAssertEqual(lines.first.map(String.init), "timestamp_seconds,elapsed_seconds,accel_x,accel_y,accel_z")
        XCTAssertEqual(lines.count, 3)
        XCTAssertEqual(String(lines[1]), "0.010000,0.010000,0.100000,-0.200000,-0.980000")
    }

    func testFlushesIncrementallyBeforeClose() throws {
        let writer = try SensorCSVWriter(url: url, flushThreshold: 3)
        for i in 0..<4 {
            try writer.append(SensorSample(timestamp: Double(i) * 0.01, x: 0, y: 0, z: -1))
        }
        // Three rows reached the threshold and must already be on disk.
        let onDisk = try String(contentsOf: url, encoding: .utf8).split(separator: "\n").count - 1
        XCTAssertEqual(onDisk, 3)
        XCTAssertEqual(writer.writtenCount, 3)
        try writer.close()
        XCTAssertEqual(writer.writtenCount, 4)
    }

    func testReopeningAppendsWithoutSecondHeader() throws {
        var writer = try SensorCSVWriter(url: url, flushThreshold: 1)
        try writer.append(SensorSample(timestamp: 0, x: 0, y: 0, z: -1))
        try writer.close()
        writer = try SensorCSVWriter(url: url, flushThreshold: 1)
        try writer.append(SensorSample(timestamp: 0.01, x: 0, y: 0, z: -1))
        try writer.close()
        let text = try String(contentsOf: url, encoding: .utf8)
        XCTAssertEqual(text.components(separatedBy: "timestamp_seconds").count - 1, 1)
        XCTAssertEqual(text.split(separator: "\n").count, 3)
    }
}

final class SamplingStatisticsTests: XCTestCase {
    func testMedianIntervalAndGapDetection() {
        var stats = SamplingStatistics(requestedInterval: 0.01)
        for t in [0.0, 0.01, 0.02, 0.03, 0.10, 0.11] {
            stats.add(timestamp: t)
        }
        XCTAssertEqual(stats.sampleCount, 6)
        XCTAssertEqual(stats.gapCount, 1)
        XCTAssertEqual(stats.largestGapSeconds, 0.07, accuracy: 1e-9)
        XCTAssertEqual(stats.medianInterval!, 0.01, accuracy: 1e-9)
        XCTAssertEqual(stats.estimatedRateHz!, 100, accuracy: 1e-6)
    }

    func testPauseIsNotCountedAsGap() {
        var stats = SamplingStatistics(requestedInterval: 0.01)
        stats.add(timestamp: 0)
        stats.add(timestamp: 0.01)
        stats.breakContinuity()
        stats.add(timestamp: 5)
        XCTAssertEqual(stats.gapCount, 0)
        XCTAssertEqual(stats.sampleCount, 3)
    }
}

final class SessionMetadataTests: XCTestCase {
    func testEncodesDocumentedKeys() throws {
        var metadata = SessionMetadata(
            participantId: "P-ABC123", sessionId: "S-1", phoneModel: "iPhone15,2", iosVersion: "17.5",
            pocketSide: .right, startingFoot: .left, targetSteps: 40, walkingCondition: .comfortable,
            samplingIntervalSeconds: 0.0101, requestedSamplingIntervalSeconds: 0.01, notes: "")
        metadata.sessionStart = Date(timeIntervalSince1970: 0)
        let json = try JSONSerialization.jsonObject(with: SessionMetadata.makeEncoder().encode(metadata)) as! [String: Any]

        for key in ["schema_version", "participant_id", "session_id", "phone_model", "ios_version", "pocket_side",
                    "starting_foot", "target_steps", "walking_condition", "sampling_interval_seconds",
                    "session_start", "notes"] {
            XCTAssertNotNil(json[key], "missing \(key)")
        }
        XCTAssertEqual(json["schema_version"] as? String, SessionMetadata.currentSchemaVersion)
        XCTAssertEqual(json["pocket_side"] as? String, "right")
        XCTAssertEqual(json["session_start"] as? String, "1970-01-01T00:00:00Z")
    }

    func testRoundTrip() throws {
        var metadata = SessionMetadata(
            participantId: "P1", sessionId: "S-2", phoneModel: "iPhone", iosVersion: "18.0",
            pocketSide: .left, startingFoot: .right, targetSteps: 20, walkingCondition: .brisk,
            samplingIntervalSeconds: nil, requestedSamplingIntervalSeconds: 0.01, notes: "a, \"b\"")
        metadata.pauses = [PauseInterval(startSeconds: 1, endSeconds: 2, reason: "test")]
        let data = try SessionMetadata.makeEncoder().encode(metadata)
        XCTAssertEqual(try SessionMetadata.makeDecoder().decode(SessionMetadata.self, from: data), metadata)
    }

    func testParticipantIdValidation() {
        XCTAssertTrue(Identifiers.isValidParticipantId("P-7Q2K"))
        XCTAssertTrue(Identifiers.isValidParticipantId(Identifiers.randomParticipantId()))
        XCTAssertFalse(Identifiers.isValidParticipantId(""))
        XCTAssertFalse(Identifiers.isValidParticipantId("Jane Doe"))
        XCTAssertFalse(Identifiers.isValidParticipantId(String(repeating: "A", count: 33)))
    }
}

final class StepLabelStoreTests: XCTestCase {
    func testEncodeSortsRenumbersAndEscapes() throws {
        let labels = [
            StepLabel(timestamp: 2.5, footLabel: .right, labelSource: .assisted, reviewed: false),
            StepLabel(timestamp: 1.25, footLabel: .left, notes: "heel, \"soft\""),
        ]
        let text = StepLabelStore.encode(labels)
        let lines = text.split(separator: "\n").map(String.init)
        XCTAssertEqual(lines[0], "step_id,timestamp_seconds,foot_label,label_source,is_valid,reviewed,notes")
        XCTAssertEqual(lines[1], "1,1.250000,left,manual,true,true,\"heel, \"\"soft\"\"\"")
        XCTAssertEqual(lines[2], "2,2.500000,right,assisted,true,false,")
    }

    func testDecodeRoundTrip() throws {
        let labels = StepLabelStore.normalized([
            StepLabel(timestamp: 0.5, footLabel: .left),
            StepLabel(timestamp: 1.0, footLabel: .unknown, isValid: false, notes: "line\nbreak"),
        ])
        let decoded = try StepLabelStore.decode(StepLabelStore.encode(labels))
        XCTAssertEqual(decoded.map(\.timestamp), [0.5, 1.0])
        XCTAssertEqual(decoded.map(\.footLabel), [.left, .unknown])
        XCTAssertEqual(decoded.map(\.isValid), [true, false])
        XCTAssertEqual(decoded[1].notes, "line\nbreak")
    }

    func testDecodeRejectsMissingColumns() {
        XCTAssertThrowsError(try StepLabelStore.decode("step_id,timestamp_seconds\n1,0.5\n"))
    }
}
