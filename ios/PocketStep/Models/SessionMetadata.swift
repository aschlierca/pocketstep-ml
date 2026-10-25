import Foundation

/// Left or right. Used for both the pocket side and the foot of a step.
enum Side: String, Codable, CaseIterable, Identifiable {
    case left
    case right

    var id: String { rawValue }
    var displayName: String { self == .left ? "Left" : "Right" }
    var opposite: Side { self == .left ? .right : .left }
}

enum WalkingCondition: String, Codable, CaseIterable, Identifiable {
    case slow
    case comfortable
    case brisk

    var id: String { rawValue }
    var displayName: String { rawValue.capitalized }
}

enum RecordingStatus: String, Codable {
    case inProgress = "in_progress"
    case completed
    /// The app was terminated or the recording stopped without the researcher pressing Stop.
    case interrupted
    /// A sensor or file error ended the recording. Data written before the error is kept.
    case failed

    var displayName: String {
        switch self {
        case .inProgress: return "In progress"
        case .completed: return "Completed"
        case .interrupted: return "Interrupted"
        case .failed: return "Failed"
        }
    }
}

/// A span with no sensor samples because recording was paused (by the researcher or automatically).
/// Times are seconds since session start on the monotonic clock, the same basis as `timestamp_seconds`.
struct PauseInterval: Codable, Hashable {
    var startSeconds: Double
    var endSeconds: Double?
    var reason: String

    enum CodingKeys: String, CodingKey {
        case startSeconds = "start_seconds"
        case endSeconds = "end_seconds"
        case reason
    }
}

/// Something that disrupted recording, such as the app moving to the background or a sensor error.
struct InterruptionEvent: Codable, Hashable {
    var atSeconds: Double
    var reason: String

    enum CodingKeys: String, CodingKey {
        case atSeconds = "at_seconds"
        case reason
    }
}

/// Contents of `session_metadata.json`. Key names are part of the documented export schema.
struct SessionMetadata: Codable, Hashable, Identifiable {
    static let currentSchemaVersion = "1.0"
    static let timeBasisDescription =
        "timestamp_seconds and elapsed_seconds are both seconds since session_start on the device's monotonic clock (CMAccelerometerData.timestamp minus the start uptime)."

    var schemaVersion: String = SessionMetadata.currentSchemaVersion
    var participantId: String
    var sessionId: String
    var phoneModel: String
    var iosVersion: String
    var pocketSide: Side
    var startingFoot: Side
    var targetSteps: Int
    var walkingCondition: WalkingCondition
    /// Median measured interval between consecutive samples. Nil until a recording has finished.
    var samplingIntervalSeconds: Double?
    var requestedSamplingIntervalSeconds: Double
    var samplingRateHzEstimate: Double?
    var sessionStart: Date?
    var sessionEnd: Date?
    var notes: String
    var recordingStatus: RecordingStatus = .inProgress
    var sampleCount: Int = 0
    /// Gaps between samples longer than three requested intervals, excluding pauses.
    var unexpectedGapCount: Int = 0
    var pauses: [PauseInterval] = []
    var interruptions: [InterruptionEvent] = []
    /// Cumulative CMPedometer count. A reference only; never a ground-truth label.
    var pedometerSteps: Int?
    var timeBasis: String = SessionMetadata.timeBasisDescription
    var appVersion: String = DeviceInfo.appVersion
    var synthetic: Bool = false

    var id: String { sessionId }

    var durationSeconds: Double? {
        guard let start = sessionStart, let end = sessionEnd else { return nil }
        return end.timeIntervalSince(start)
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case participantId = "participant_id"
        case sessionId = "session_id"
        case phoneModel = "phone_model"
        case iosVersion = "ios_version"
        case pocketSide = "pocket_side"
        case startingFoot = "starting_foot"
        case targetSteps = "target_steps"
        case walkingCondition = "walking_condition"
        case samplingIntervalSeconds = "sampling_interval_seconds"
        case requestedSamplingIntervalSeconds = "requested_sampling_interval_seconds"
        case samplingRateHzEstimate = "sampling_rate_hz_estimate"
        case sessionStart = "session_start"
        case sessionEnd = "session_end"
        case notes
        case recordingStatus = "recording_status"
        case sampleCount = "sample_count"
        case unexpectedGapCount = "unexpected_gap_count"
        case pauses
        case interruptions
        case pedometerSteps = "pedometer_steps"
        case timeBasis = "time_basis"
        case appVersion = "app_version"
        case synthetic
    }

    static func makeEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}

enum Identifiers {
    private static let participantPattern = /^[A-Za-z0-9_-]{1,32}$/

    /// Anonymous participant IDs: 1-32 letters, digits, dashes or underscores.
    static func isValidParticipantId(_ id: String) -> Bool {
        id.wholeMatch(of: participantPattern) != nil
    }

    static func randomParticipantId() -> String {
        let alphabet = Array("ABCDEFGHJKLMNPQRSTUVWXYZ23456789")
        return "P-" + String((0..<6).map { _ in alphabet.randomElement()! })
    }

    static func newSessionId(date: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "UTC")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let suffix = String(format: "%04X", Int.random(in: 0...0xFFFF))
        return "S-\(formatter.string(from: date))-\(suffix)"
    }
}
