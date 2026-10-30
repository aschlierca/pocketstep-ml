import SwiftUI

struct SessionSetupView: View {
    @EnvironmentObject private var router: Router

    @State private var participantId = ""
    @State private var sessionId = Identifiers.newSessionId()
    @State private var phoneModel = DeviceInfo.modelIdentifier
    @State private var pocketSide: Side = .right
    @State private var startingFoot: Side = .left
    @State private var targetSteps = 40
    @State private var walkingCondition: WalkingCondition = .comfortable
    @State private var samplingRateHz = 100
    @State private var countdownSeconds = 5
    @State private var notes = ""

    private var participantIdIsValid: Bool { Identifiers.isValidParticipantId(participantId) }
    private var canContinue: Bool {
        participantIdIsValid && !phoneModel.trimmingCharacters(in: .whitespaces).isEmpty
    }

    var body: some View {
        Form {
            Section {
                TextField("Participant ID", text: $participantId)
                    .textInputAutocapitalization(.characters)
                    .autocorrectionDisabled()
                if !participantId.isEmpty && !participantIdIsValid {
                    Label("Use 1–32 letters, digits, dashes or underscores.", systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                }
                Button("Generate Anonymous ID") {
                    participantId = Identifiers.randomParticipantId()
                }
                LabeledContent("Session ID") {
                    Text(sessionId).font(.body.monospaced())
                }
            } header: {
                Text("Identifiers")
            } footer: {
                Text("Use an anonymous code. Never enter a participant's name or other identifying details. Sessions shared to the project are published publicly.")
            }

            Section("Device") {
                TextField("Phone model", text: $phoneModel)
                    .autocorrectionDisabled()
                LabeledContent("iOS version", value: DeviceInfo.iosVersion)
            }

            Section("Protocol") {
                sidePicker("Pocket side", selection: $pocketSide)
                sidePicker("Starting foot", selection: $startingFoot)
                Stepper("Target steps: \(targetSteps)", value: $targetSteps, in: 10...1000, step: 10)
                VStack(alignment: .leading) {
                    Text("Walking condition")
                    Picker("Walking condition", selection: $walkingCondition) {
                        ForEach(WalkingCondition.allCases) { Text($0.displayName).tag($0) }
                    }
                    .pickerStyle(.segmented)
                }
            }

            Section {
                Picker("Sampling rate", selection: $samplingRateHz) {
                    ForEach([50, 100, 200], id: \.self) { Text("\($0) Hz").tag($0) }
                }
                Picker("Countdown before recording", selection: $countdownSeconds) {
                    ForEach([0, 3, 5, 10], id: \.self) { Text($0 == 0 ? "None" : "\($0) s").tag($0) }
                }
            } header: {
                Text("Recording")
            } footer: {
                Text("100 Hz is the default. iPhones may deliver fewer samples than requested; the measured rate is saved with the session.")
            }

            Section {
                TextField("Optional, e.g. trouser type, phone orientation", text: $notes, axis: .vertical)
                    .lineLimit(3...6)
            } header: {
                Text("Notes")
            } footer: {
                Text("Notes are published with the data. Do not include personal information.")
            }

            Section("Before You Record") {
                VStack(alignment: .leading, spacing: 8) {
                    instruction(1, "Confirm the participant has consented under your approved protocol, including public release of their anonymized motion data.")
                    instruction(2, "Put the phone in the \(pocketSide.displayName.lowercased()) front trouser pocket, screen facing the leg, top of the phone pointing down. Note any other orientation in Notes.")
                    instruction(3, "The pocket must hold the phone snugly. Keep the same position for the whole session.")
                    instruction(4, "Press Start, then pocket the phone during the countdown. The screen turns off in the pocket.")
                    instruction(5, "Stand still for about 2 seconds, walk in place starting with the \(startingFoot.displayName.lowercased()) foot for \(targetSteps) steps, then stand still again.")
                    instruction(6, "Take the phone out and press Stop.")
                }
                .font(.callout)
            }

            Section {
                LargeButton(title: "Continue to Recording", systemImage: "arrow.right.circle") {
                    router.path.append(.record(RecordingRequest(metadata: makeMetadata(), countdownSeconds: countdownSeconds)))
                }
                .disabled(!canContinue)
            }
        }
        .navigationTitle("New Session")
    }

    private func sidePicker(_ title: String, selection: Binding<Side>) -> some View {
        VStack(alignment: .leading) {
            Text(title)
            Picker(title, selection: selection) {
                ForEach(Side.allCases) { Text($0.displayName).tag($0) }
            }
            .pickerStyle(.segmented)
        }
    }

    private func instruction(_ number: Int, _ text: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text("\(number).").bold()
            Text(text).fixedSize(horizontal: false, vertical: true)
        }
        .accessibilityElement(children: .combine)
    }

    private func makeMetadata() -> SessionMetadata {
        SessionMetadata(
            participantId: participantId,
            sessionId: sessionId,
            phoneModel: phoneModel.trimmingCharacters(in: .whitespaces),
            iosVersion: DeviceInfo.iosVersion,
            pocketSide: pocketSide,
            startingFoot: startingFoot,
            targetSteps: targetSteps,
            walkingCondition: walkingCondition,
            samplingIntervalSeconds: nil,
            requestedSamplingIntervalSeconds: 1.0 / Double(samplingRateHz),
            notes: notes)
    }
}
