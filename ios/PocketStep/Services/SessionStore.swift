import Foundation

/// Owns the on-device session folders: Documents/Sessions/<session_id>/.
/// Data stays on the device unless the researcher exports it.
@MainActor
final class SessionStore: ObservableObject {
    @Published private(set) var sessions: [SessionMetadata] = []
    @Published var lastError: String?

    let rootDirectory: URL

    init(rootDirectory: URL? = nil) {
        if let rootDirectory {
            self.rootDirectory = rootDirectory
        } else {
            let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            self.rootDirectory = documents.appendingPathComponent("Sessions", isDirectory: true)
        }
        try? FileManager.default.createDirectory(at: self.rootDirectory, withIntermediateDirectories: true)
        markAbandonedRecordingsInterrupted()
        reload()
    }

    func directory(for sessionId: String) -> URL {
        rootDirectory.appendingPathComponent(sessionId, isDirectory: true)
    }

    func fileURL(_ name: String, for sessionId: String) -> URL {
        directory(for: sessionId).appendingPathComponent(name)
    }

    func createDirectory(for metadata: SessionMetadata) throws -> URL {
        let url = directory(for: metadata.sessionId)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    func save(_ metadata: SessionMetadata) throws {
        let data = try SessionMetadata.makeEncoder().encode(metadata)
        try data.write(to: fileURL(SessionFiles.metadata, for: metadata.sessionId), options: .atomic)
        if let index = sessions.firstIndex(where: { $0.sessionId == metadata.sessionId }) {
            sessions[index] = metadata
        } else {
            reload()
        }
    }

    func loadMetadata(sessionId: String) throws -> SessionMetadata {
        try Self.readMetadata(at: fileURL(SessionFiles.metadata, for: sessionId))
    }

    func delete(sessionId: String) {
        do {
            try FileManager.default.removeItem(at: directory(for: sessionId))
        } catch {
            lastError = "Could not delete session \(sessionId): \(error.localizedDescription)"
        }
        reload()
    }

    func reload() {
        let fm = FileManager.default
        let folders = (try? fm.contentsOfDirectory(at: rootDirectory, includingPropertiesForKeys: nil)) ?? []
        sessions = folders
            .compactMap { try? Self.readMetadata(at: $0.appendingPathComponent(SessionFiles.metadata)) }
            .sorted { ($0.sessionStart ?? .distantPast) > ($1.sessionStart ?? .distantPast) }
    }

    /// A session still marked in progress at launch was cut off when the app was terminated.
    private func markAbandonedRecordingsInterrupted() {
        let fm = FileManager.default
        let folders = (try? fm.contentsOfDirectory(at: rootDirectory, includingPropertiesForKeys: nil)) ?? []
        for folder in folders {
            let url = folder.appendingPathComponent(SessionFiles.metadata)
            guard var metadata = try? Self.readMetadata(at: url), metadata.recordingStatus == .inProgress else {
                continue
            }
            metadata.recordingStatus = .interrupted
            metadata.interruptions.append(
                InterruptionEvent(atSeconds: -1, reason: "App was terminated during recording; time of termination unknown"))
            if let count = try? Self.countDataRows(in: folder.appendingPathComponent(SessionFiles.sensorData)) {
                metadata.sampleCount = count
            }
            if let data = try? SessionMetadata.makeEncoder().encode(metadata) {
                try? data.write(to: url, options: .atomic)
            }
        }
    }

    private static func readMetadata(at url: URL) throws -> SessionMetadata {
        try SessionMetadata.makeDecoder().decode(SessionMetadata.self, from: Data(contentsOf: url))
    }

    private static func countDataRows(in url: URL) throws -> Int {
        let data = try Data(contentsOf: url)
        let newlines = data.reduce(0) { $1 == UInt8(ascii: "\n") ? $0 + 1 : $0 }
        return max(0, newlines - 1)
    }
}
