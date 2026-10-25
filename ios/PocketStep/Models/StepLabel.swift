import Foundation

enum FootLabel: String, Codable, CaseIterable, Identifiable {
    case left
    case right
    case unknown

    var id: String { rawValue }
    var displayName: String { rawValue.capitalized }

    init(_ side: Side) {
        self = side == .left ? .left : .right
    }
}

enum LabelSource: String, Codable {
    /// The researcher placed the step and chose its foot.
    case manual
    /// The foot was proposed by the alternation pattern. Only trusted once `reviewed` is true.
    case assisted
}

/// One row of `step_labels.csv`.
struct StepLabel: Identifiable, Equatable {
    /// In-memory identity for SwiftUI; `stepId` is renumbered in time order on save.
    let id: UUID
    var stepId: Int
    /// Footfall time in seconds since session start (same basis as sensor_data.csv).
    var timestamp: Double
    var footLabel: FootLabel
    var labelSource: LabelSource
    var isValid: Bool
    var reviewed: Bool
    var notes: String

    init(id: UUID = UUID(), stepId: Int = 0, timestamp: Double, footLabel: FootLabel = .unknown,
         labelSource: LabelSource = .manual, isValid: Bool = true, reviewed: Bool = true, notes: String = "") {
        self.id = id
        self.stepId = stepId
        self.timestamp = timestamp
        self.footLabel = footLabel
        self.labelSource = labelSource
        self.isValid = isValid
        self.reviewed = reviewed
        self.notes = notes
    }
}

enum CSV {
    static func escape(_ field: String) -> String {
        guard field.contains(where: { $0 == "," || $0 == "\"" || $0 == "\n" || $0 == "\r" }) else {
            return field
        }
        return "\"" + field.replacingOccurrences(of: "\"", with: "\"\"") + "\""
    }

    /// Minimal RFC 4180 parser: quoted fields, doubled quotes, and newlines inside quotes.
    static func parse(_ text: String) -> [[String]] {
        var rows: [[String]] = []
        var row: [String] = []
        var field = ""
        var inQuotes = false
        var iterator = text.makeIterator()
        var pending: Character? = nil

        func nextChar() -> Character? {
            if let p = pending { pending = nil; return p }
            return iterator.next()
        }

        while let c = nextChar() {
            if inQuotes {
                if c == "\"" {
                    if let n = nextChar() {
                        if n == "\"" { field.append("\"") } else { inQuotes = false; pending = n }
                    } else {
                        inQuotes = false
                    }
                } else {
                    field.append(c)
                }
            } else if c == "\"" {
                inQuotes = true
            } else if c == "," {
                row.append(field)
                field = ""
            } else if c == "\n" || c == "\r\n" || c == "\r" {
                row.append(field)
                rows.append(row)
                row = []
                field = ""
            } else {
                field.append(c)
            }
        }
        if !field.isEmpty || !row.isEmpty {
            row.append(field)
            rows.append(row)
        }
        return rows
    }
}

enum StepLabelStore {
    static let columns = ["step_id", "timestamp_seconds", "foot_label", "label_source", "is_valid", "reviewed", "notes"]
    static var header: String { columns.joined(separator: ",") + "\n" }

    enum ReadError: LocalizedError {
        case missingColumns([String])
        case badRow(line: Int, reason: String)

        var errorDescription: String? {
            switch self {
            case .missingColumns(let names):
                return "step_labels.csv is missing columns: \(names.joined(separator: ", "))"
            case .badRow(let line, let reason):
                return "step_labels.csv line \(line): \(reason)"
            }
        }
    }

    static func createEmptyIfMissing(at url: URL) throws {
        guard !FileManager.default.fileExists(atPath: url.path) else { return }
        try Data(header.utf8).write(to: url, options: .atomic)
    }

    /// Sorts by time and renumbers `step_id` from 1 so exported IDs follow the walking order.
    static func normalized(_ labels: [StepLabel]) -> [StepLabel] {
        labels.sorted { $0.timestamp < $1.timestamp }
            .enumerated()
            .map { index, label in
                var copy = label
                copy.stepId = index + 1
                return copy
            }
    }

    static func encode(_ labels: [StepLabel]) -> String {
        var text = header
        for label in normalized(labels) {
            let fields = [
                String(label.stepId),
                String(format: "%.6f", label.timestamp),
                label.footLabel.rawValue,
                label.labelSource.rawValue,
                label.isValid ? "true" : "false",
                label.reviewed ? "true" : "false",
                CSV.escape(label.notes),
            ]
            text += fields.joined(separator: ",") + "\n"
        }
        return text
    }

    static func decode(_ text: String) throws -> [StepLabel] {
        let rows = CSV.parse(text)
        guard let header = rows.first else { return [] }
        let missing = columns.filter { !header.contains($0) }
        guard missing.isEmpty else { throw ReadError.missingColumns(missing) }
        let index = Dictionary(uniqueKeysWithValues: header.enumerated().map { ($1, $0) })

        return try rows.dropFirst().enumerated().compactMap { offset, row in
            let line = offset + 2
            if row.count == 1 && row[0].isEmpty { return nil }
            guard row.count == header.count else {
                throw ReadError.badRow(line: line, reason: "expected \(header.count) fields, found \(row.count)")
            }
            func value(_ name: String) -> String { row[index[name]!] }
            guard let stepId = Int(value("step_id")),
                  let timestamp = Double(value("timestamp_seconds")),
                  let foot = FootLabel(rawValue: value("foot_label")),
                  let source = LabelSource(rawValue: value("label_source")) else {
                throw ReadError.badRow(line: line, reason: "unreadable step_id, timestamp, foot_label or label_source")
            }
            return StepLabel(stepId: stepId, timestamp: timestamp, footLabel: foot, labelSource: source,
                             isValid: value("is_valid") == "true", reviewed: value("reviewed") == "true",
                             notes: value("notes"))
        }
    }

    static func read(from url: URL) throws -> [StepLabel] {
        guard FileManager.default.fileExists(atPath: url.path) else { return [] }
        return try decode(String(contentsOf: url, encoding: .utf8))
    }

    static func write(_ labels: [StepLabel], to url: URL) throws {
        try Data(encode(labels).utf8).write(to: url, options: .atomic)
    }
}
