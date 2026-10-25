import SwiftUI

struct RecordingView: View {
    @EnvironmentObject private var router: Router
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var recorder: MotionRecorder
    @State private var confirmingStop = false
    private let countdownSeconds: Int

    init(request: RecordingRequest, store: SessionStore) {
        _recorder = StateObject(wrappedValue: MotionRecorder(metadata: request.metadata, store: store))
        countdownSeconds = request.countdownSeconds
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                statusCard
                measurements
                pedometer
                if let warning = recorder.warning {
                    WarningBanner(text: warning)
                }
                if recorder.gapCount > 0 {
                    WarningBanner(text: "\(recorder.gapCount) unexpected gap(s) in the sensor stream so far. They are kept as gaps, not filled in.")
                }
                controls
            }
            .padding()
        }
        .navigationTitle("Recording")
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(recorder.state != .ready)
        .onAppear { recorder.requestMotionPermissionIfNeeded() }
        .onChange(of: scenePhase) { _, phase in
            if phase == .background { recorder.handleBackgrounding() }
        }
        .confirmationDialog("Stop recording?", isPresented: $confirmingStop, titleVisibility: .visible) {
            Button("Stop and Save", role: .destructive) { recorder.stop() }
        } message: {
            Text("No more samples can be added to this session after it stops.")
        }
    }

    // MARK: Sections

    private var statusCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(statusTitle, systemImage: statusImage)
                .font(.title2.bold())
                .foregroundStyle(statusColor)
            Text(Format.duration(recorder.elapsedSeconds))
                .font(.system(.largeTitle, design: .monospaced))
                .accessibilityLabel("Elapsed time \(Int(recorder.elapsedSeconds)) seconds")
            Text(statusDetail)
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .accessibilityElement(children: .combine)
    }

    private var measurements: some View {
        VStack(alignment: .leading, spacing: 6) {
            LabeledContent("Participant", value: recorder.metadata.participantId)
            LabeledContent("Session", value: recorder.metadata.sessionId)
            LabeledContent("Pocket", value: recorder.metadata.pocketSide.displayName)
            LabeledContent("Samples", value: "\(recorder.sampleCount)")
            LabeledContent("Measured rate", value: recorder.measuredRateHz.map { String(format: "%.1f Hz", $0) } ?? "–")
            LabeledContent("Pauses", value: "\(recorder.metadata.pauses.count)")
        }
        .font(.body.monospacedDigit())
    }

    private var pedometer: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Pedometer (reference only)")
                .font(.headline)
            if let message = recorder.pedometerMessage {
                Text(message).font(.callout).foregroundStyle(.secondary)
            } else {
                LabeledContent("Steps counted", value: recorder.pedometerSteps.map(String.init) ?? "–")
                LabeledContent("Cadence", value: recorder.pedometerCadence.map { String(format: "%.2f steps/s", $0) } ?? "–")
            }
            Text("Pedometer counts are approximate and are not used as step labels.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var controls: some View {
        VStack(spacing: 12) {
            switch recorder.state {
            case .ready:
                LargeButton(title: "Start", systemImage: "record.circle") {
                    recorder.start(countdownSeconds: countdownSeconds)
                }
            case .countingDown:
                LargeButton(title: "Cancel", systemImage: "xmark", prominent: false) {
                    recorder.cancelCountdown()
                }
            case .recording:
                LargeButton(title: "Pause", systemImage: "pause.fill", prominent: false) { recorder.pause() }
                LargeButton(title: "Stop", systemImage: "stop.fill", role: .destructive) { confirmingStop = true }
            case .paused:
                LargeButton(title: "Resume", systemImage: "play.fill") { recorder.resume() }
                LargeButton(title: "Stop", systemImage: "stop.fill", role: .destructive, prominent: false) {
                    confirmingStop = true
                }
            case .finished, .failed:
                LargeButton(title: "Done", systemImage: "checkmark", prominent: !recorder.hasSavedData) {
                    router.popToRoot()
                }
            }
        }
    }

    // MARK: Status text

    private var statusTitle: String {
        switch recorder.state {
        case .ready: return "Ready"
        case .countingDown(let remaining): return "Starting in \(remaining)"
        case .recording: return "Recording"
        case .paused: return "Paused"
        case .finished: return "Saved"
        case .failed: return "Stopped"
        }
    }

    private var statusImage: String {
        switch recorder.state {
        case .ready: return "circle"
        case .countingDown: return "timer"
        case .recording: return "record.circle.fill"
        case .paused: return "pause.circle.fill"
        case .finished: return "checkmark.circle.fill"
        case .failed: return "exclamationmark.triangle.fill"
        }
    }

    private var statusColor: Color {
        switch recorder.state {
        case .recording: return .red
        case .failed: return .orange
        case .finished: return .green
        default: return .primary
        }
    }

    private var statusDetail: String {
        let pocket = recorder.metadata.pocketSide.displayName.lowercased()
        switch recorder.state {
        case .ready:
            return "Press Start, then place the phone in the \(pocket) pocket during the countdown."
        case .countingDown:
            return "Place the phone in the \(pocket) pocket now and stand still."
        case .recording:
            return "Walk in place starting with the \(recorder.metadata.startingFoot.displayName.lowercased()) foot. Data is saved continuously."
        case .paused:
            return "No samples are recorded while paused. The pause is saved in the session metadata."
        case .finished:
            return "The session was saved on this phone. Label its steps from the session list."
        case .failed(let message):
            return message
        }
    }
}
