import SwiftUI

struct StepEditSheet: View {
    @Environment(\.dismiss) private var dismiss
    @State private var draft: StepLabel
    private let original: StepLabel
    private let snap: (Double) -> Double
    private let onSave: (StepLabel) -> Void
    private let onDelete: () -> Void

    init(label: StepLabel, snap: @escaping (Double) -> Double,
         onSave: @escaping (StepLabel) -> Void, onDelete: @escaping () -> Void) {
        _draft = State(initialValue: label)
        original = label
        self.snap = snap
        self.onSave = onSave
        self.onDelete = onDelete
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Stepper(value: $draft.timestamp, step: 0.01) {
                        Text(Format.seconds(draft.timestamp)).font(.body.monospacedDigit())
                    }
                    Button("Snap to Nearest Peak") { draft.timestamp = snap(draft.timestamp) }
                } header: {
                    Text("Footfall Time")
                } footer: {
                    Text("Place the step where the foot lands, not elsewhere in the movement.")
                }

                Section {
                    Picker("Foot", selection: footBinding) {
                        ForEach(FootLabel.allCases) { Text($0.displayName).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    if draft.labelSource == .assisted && !draft.reviewed {
                        Label("Proposed by assisted labeling. Confirm it or choose a different foot.", systemImage: "eye.trianglebadge.exclamationmark")
                            .font(.callout)
                    }
                    Toggle("Reviewed by a person", isOn: $draft.reviewed)
                } header: {
                    Text("Foot")
                } footer: {
                    Text("Choose Unknown when you cannot tell which foot it was.")
                }

                Section {
                    Toggle("Valid step", isOn: $draft.isValid)
                } footer: {
                    Text("Mark a real step invalid when its signal is unusable, for example if the phone shifted. It stays in the sequence but is excluded from training.")
                }

                Section("Notes") {
                    TextField("Optional", text: $draft.notes, axis: .vertical)
                }

                Section {
                    Button("Delete Step", role: .destructive) {
                        onDelete()
                        dismiss()
                    }
                }
            }
            .navigationTitle("Step \(original.stepId)")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        onSave(draft)
                        dismiss()
                    }
                }
            }
        }
    }

    /// Changing the foot is a human decision: it becomes a reviewed manual label.
    private var footBinding: Binding<FootLabel> {
        Binding(get: { draft.footLabel }, set: { newValue in
            guard newValue != draft.footLabel else { return }
            draft.footLabel = newValue
            draft.labelSource = .manual
            draft.reviewed = true
        })
    }
}

struct AssistedLabelingSheet: View {
    @Environment(\.dismiss) private var dismiss
    let labels: [StepLabel]
    let onApply: (Side, Int) -> Void
    @State private var startingFoot: Side
    @State private var startStep = 1

    init(labels: [StepLabel], defaultStartingFoot: Side, onApply: @escaping (Side, Int) -> Void) {
        self.labels = labels
        self.onApply = onApply
        _startingFoot = State(initialValue: defaultStartingFoot)
    }

    private var affectedCount: Int {
        labels.dropFirst(startStep - 1).filter { !$0.reviewed }.count
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Text("Proposes alternating feet for the existing steps in time order. Proposed labels are marked Needs review and are not treated as ground truth until a person confirms them.")
                }
                Section {
                    Picker("Foot of first step", selection: $startingFoot) {
                        ForEach(Side.allCases) { Text($0.displayName).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    Stepper("Start at step \(startStep)", value: $startStep, in: 1...max(1, labels.count))
                } header: {
                    Text("Pattern")
                } footer: {
                    Text("Start later than step 1 to restart the alternation after a missed or extra step. Steps before it are unchanged.")
                }
                Section {
                    Text("\(affectedCount) unreviewed step(s) will receive proposals. Reviewed steps are never overwritten.")
                }
            }
            .navigationTitle("Assisted Labeling")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Propose") {
                        onApply(startingFoot, startStep - 1)
                        dismiss()
                    }
                    .disabled(affectedCount == 0)
                }
            }
        }
    }
}
