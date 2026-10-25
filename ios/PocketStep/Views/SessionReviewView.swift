import SwiftUI
import UIKit

struct SessionReviewView: View {
    let sessionId: String
    @EnvironmentObject private var store: SessionStore
    @EnvironmentObject private var router: Router

    @State private var metadata: SessionMetadata?
    @State private var labels: [StepLabel] = []
    @State private var sampleCount = 0
    @State private var sampleRange: ClosedRange<Double>?
    @State private var malformedRows = 0
    @State private var loadError: String?
    @State private var exportFile: ExportFile?
    @State private var exportError: String?
    @State private var isExporting = false

    var body: some View {
        List {
            if let loadError {
                WarningBanner(text: loadError)
            }
            if let metadata {
                annotationSection
                warningSection(metadata)
                actionSection
                sessionSection(metadata)
                recordingSection(metadata)
            }
        }
        .navigationTitle("Session Review")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear { Task { await load() } }
        .sheet(item: $exportFile) { file in
            ActivityView(items: [file.url])
        }
    }

    // MARK: Sections

    private func sessionSection(_ m: SessionMetadata) -> some View {
        Section("Session") {
            LabeledContent("Participant", value: m.participantId)
            LabeledContent("Session ID", value: m.sessionId)
            LabeledContent("Phone model", value: m.phoneModel)
            LabeledContent("iOS", value: m.iosVersion)
            LabeledContent("Pocket side", value: m.pocketSide.displayName)
            LabeledContent("Starting foot", value: m.startingFoot.displayName)
            LabeledContent("Target steps", value: "\(m.targetSteps)")
            LabeledContent("Walking condition", value: m.walkingCondition.displayName)
            if !m.notes.isEmpty {
                LabeledContent("Notes", value: m.notes)
            }
            LabeledContent("Schema version", value: m.schemaVersion)
        }
    }

    private func recordingSection(_ m: SessionMetadata) -> some View {
        Section("Recording") {
            LabeledContent("Status") {
                Label(m.recordingStatus.displayName, systemImage: m.recordingStatus.systemImage)
            }
            if let start = m.sessionStart {
                LabeledContent("Started") { Text(start, format: .dateTime) }
            }
            LabeledContent("Duration", value: m.durationSeconds.map(Format.duration) ?? "–")
            LabeledContent("Raw samples", value: "\(sampleCount)")
            LabeledContent("Measured rate", value: m.samplingRateHzEstimate.map { String(format: "%.1f Hz", $0) } ?? "–")
            LabeledContent("Requested rate", value: String(format: "%.0f Hz", 1 / m.requestedSamplingIntervalSeconds))
            LabeledContent("Unexpected gaps", value: "\(m.unexpectedGapCount)")
            LabeledContent("Pauses", value: "\(m.pauses.count)")
            LabeledContent("Pedometer steps (reference)", value: m.pedometerSteps.map(String.init) ?? "–")
            ForEach(Array(m.interruptions.enumerated()), id: \.offset) { _, event in
                Label(event.reason, systemImage: "exclamationmark.circle")
                    .font(.callout)
            }
        }
        .font(.body.monospacedDigit())
    }

    private var annotationSection: some View {
        let summary = AnnotationSummary(labels: labels)
        return Section {
            LabeledContent("Annotated steps", value: "\(summary.total)")
            LabeledContent("Left", value: "\(summary.left)")
            LabeledContent("Right", value: "\(summary.right)")
            LabeledContent("Unknown (uncertain)", value: "\(summary.unknown)")
            LabeledContent("Invalid", value: "\(summary.invalid)")
            LabeledContent("Not yet reviewed", value: "\(summary.unreviewed)")
            LabeledContent("Usable for training", value: "\(summary.usableForTraining)")
        } header: {
            Text("Annotations")
        } footer: {
            Text("Usable for training counts valid, reviewed left and right labels.")
        }
    }

    private func warningSection(_ m: SessionMetadata) -> some View {
        let warnings = AnnotationValidator.warnings(metadata: m, labels: labels,
                                                    sampleTimeRange: sampleRange, malformedRows: malformedRows)
        return Section("Checks") {
            if warnings.isEmpty {
                Label("No problems found.", systemImage: "checkmark.circle")
            }
            ForEach(warnings, id: \.self) { WarningBanner(text: $0) }
        }
    }

    private var actionSection: some View {
        Section {
            LargeButton(title: "Label Steps", systemImage: "tag") {
                router.path.append(.label(sessionId))
            }
            LargeButton(title: isExporting ? "Preparing ZIP…" : "Export ZIP", systemImage: "square.and.arrow.up", prominent: false) {
                Task { await export() }
            }
            .disabled(isExporting)
            if let exportError {
                WarningBanner(text: exportError)
            }
        } footer: {
            Text("The ZIP contains sensor_data.csv, step_labels.csv and session_metadata.json. Data stays on this phone unless you share it.")
        }
    }

    // MARK: Loading and export

    private func load() async {
        do {
            metadata = try store.loadMetadata(sessionId: sessionId)
            labels = StepLabelStore.normalized(try StepLabelStore.read(from: store.fileURL(SessionFiles.stepLabels, for: sessionId)))
        } catch {
            loadError = "Could not open this session: \(error.localizedDescription)"
            return
        }
        let url = store.fileURL(SessionFiles.sensorData, for: sessionId)
        do {
            let result = try await Task.detached(priority: .userInitiated) { try SensorDataReader.read(from: url) }.value
            sampleCount = result.samples.count
            malformedRows = result.malformedRowCount
            if let first = result.samples.first, let last = result.samples.last {
                sampleRange = first.timestamp...last.timestamp
            }
            loadError = nil
        } catch {
            loadError = "Could not read sensor data: \(error.localizedDescription)"
        }
    }

    private func export() async {
        isExporting = true
        defer { isExporting = false }
        let directory = store.directory(for: sessionId)
        let id = sessionId
        do {
            let url = try await Task.detached { try SessionExporter.makeZip(sessionDirectory: directory, sessionId: id) }.value
            exportError = nil
            exportFile = ExportFile(url: url)
        } catch {
            exportError = error.localizedDescription
        }
    }
}

private struct ExportFile: Identifiable {
    let url: URL
    var id: URL { url }
}

private struct ActivityView: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
