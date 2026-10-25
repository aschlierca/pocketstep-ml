import Foundation

/// Packages a session's three files into `<session_id>.zip` containing a `<session_id>/` folder.
/// Uses the system's NSFileCoordinator zipping, so no third-party archive library is needed.
enum SessionExporter {
    enum ExportError: LocalizedError {
        case missingFile(String)
        case zipFailed(String)

        var errorDescription: String? {
            switch self {
            case .missingFile(let name): return "The session is missing \(name), so it cannot be exported."
            case .zipFailed(let reason): return "Creating the ZIP failed: \(reason)"
            }
        }
    }

    static let exportedFiles = [SessionFiles.sensorData, SessionFiles.stepLabels, SessionFiles.metadata]

    static func makeZip(sessionDirectory: URL, sessionId: String) throws -> URL {
        let fm = FileManager.default
        let workDirectory = fm.temporaryDirectory.appendingPathComponent("export-\(UUID().uuidString)", isDirectory: true)
        let staging = workDirectory.appendingPathComponent(sessionId, isDirectory: true)
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)

        // Copy only the documented files so nothing else on the device is swept into the export.
        for name in exportedFiles {
            let source = sessionDirectory.appendingPathComponent(name)
            guard fm.fileExists(atPath: source.path) else { throw ExportError.missingFile(name) }
            try fm.copyItem(at: source, to: staging.appendingPathComponent(name))
        }

        let destination = workDirectory.appendingPathComponent("\(sessionId).zip")
        var coordinationError: NSError?
        var copyError: Error?
        NSFileCoordinator().coordinate(readingItemAt: staging, options: .forUploading, error: &coordinationError) { zipURL in
            do {
                try fm.copyItem(at: zipURL, to: destination)
            } catch {
                copyError = error
            }
        }
        if let error = coordinationError ?? copyError {
            throw ExportError.zipFailed(error.localizedDescription)
        }
        try? fm.removeItem(at: staging)
        return destination
    }
}
