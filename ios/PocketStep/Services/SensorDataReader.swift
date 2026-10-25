import Foundation

enum SensorDataReader {
    enum ReadError: LocalizedError {
        case missingHeader
        case missingColumns([String])

        var errorDescription: String? {
            switch self {
            case .missingHeader:
                return "sensor_data.csv is empty or has no header row."
            case .missingColumns(let names):
                return "sensor_data.csv is missing columns: \(names.joined(separator: ", "))."
            }
        }
    }

    struct Result {
        var samples: [SensorSample]
        /// Rows that could not be parsed. They are skipped and reported, never repaired.
        var malformedRowCount: Int
    }

    static func read(from url: URL) throws -> Result {
        try parse(String(contentsOf: url, encoding: .utf8))
    }

    static func parse(_ text: String) throws -> Result {
        var lines = text.split(whereSeparator: \.isNewline).makeIterator()
        guard let headerLine = lines.next() else { throw ReadError.missingHeader }
        let header = headerLine.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        let required = ["timestamp_seconds", "accel_x", "accel_y", "accel_z"]
        let missing = required.filter { !header.contains($0) }
        guard missing.isEmpty else { throw ReadError.missingColumns(missing) }
        let ti = header.firstIndex(of: "timestamp_seconds")!
        let xi = header.firstIndex(of: "accel_x")!
        let yi = header.firstIndex(of: "accel_y")!
        let zi = header.firstIndex(of: "accel_z")!

        var samples: [SensorSample] = []
        var malformed = 0
        while let line = lines.next() {
            let fields = line.split(separator: ",", omittingEmptySubsequences: false)
            guard fields.count == header.count,
                  let t = Double(fields[ti]), let x = Double(fields[xi]),
                  let y = Double(fields[yi]), let z = Double(fields[zi]) else {
                malformed += 1
                continue
            }
            samples.append(SensorSample(timestamp: t, x: x, y: y, z: z))
        }
        return Result(samples: samples, malformedRowCount: malformed)
    }
}
