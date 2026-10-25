import Foundation

/// Appends accelerometer samples to `sensor_data.csv`, flushing to disk in small batches so an
/// interrupted session keeps everything up to the last flush.
///
/// Not thread-safe: use one instance from a single serial queue.
final class SensorCSVWriter {
    static let header = "timestamp_seconds,elapsed_seconds,accel_x,accel_y,accel_z\n"

    private let handle: FileHandle
    private let flushThreshold: Int
    private var buffer = ""
    private var bufferedCount = 0
    private(set) var writtenCount = 0

    /// Opens `url` for appending, creating it with a header row if it does not exist yet.
    init(url: URL, flushThreshold: Int = 100) throws {
        if !FileManager.default.fileExists(atPath: url.path) {
            try Data(Self.header.utf8).write(to: url)
        }
        handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
        self.flushThreshold = max(1, flushThreshold)
    }

    func append(_ sample: SensorSample) throws {
        buffer += Self.row(for: sample)
        bufferedCount += 1
        if bufferedCount >= flushThreshold {
            try flush()
        }
    }

    func flush() throws {
        guard bufferedCount > 0 else { return }
        try handle.write(contentsOf: Data(buffer.utf8))
        try handle.synchronize()
        writtenCount += bufferedCount
        buffer = ""
        bufferedCount = 0
    }

    func close() throws {
        try flush()
        try handle.close()
    }

    /// Both time columns carry the same session-relative monotonic time (see `SessionMetadata.timeBasis`).
    static func row(for sample: SensorSample) -> String {
        String(format: "%.6f,%.6f,%.6f,%.6f,%.6f\n",
               sample.timestamp, sample.timestamp, sample.x, sample.y, sample.z)
    }
}
