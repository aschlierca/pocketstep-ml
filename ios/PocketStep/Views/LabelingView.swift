import Charts
import SwiftUI

struct LabelingView: View {
    @StateObject private var model: LabelingModel
    @State private var selectedTime: Double?
    @State private var scrollPosition: Double = 0
    @State private var visibleSeconds: Double = 6
    @State private var snapToPeak = true
    @State private var editingLabel: StepLabel?
    @State private var selection = Set<UUID>()
    @State private var editMode: EditMode = .inactive
    @State private var showingAssisted = false
    @State private var confirmingReviewAll = false
    @State private var infoMessage: String?

    init(sessionId: String, store: SessionStore) {
        _model = StateObject(wrappedValue: LabelingModel(sessionId: sessionId, store: store))
    }

    var body: some View {
        Group {
            if model.isLoaded {
                content
            } else {
                ProgressView("Loading recording…")
            }
        }
        .navigationTitle("Label Steps")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            if !model.isLoaded { await model.load() }
        }
    }

    private var content: some View {
        List(selection: $selection) {
            if let error = model.errorMessage {
                WarningBanner(text: error)
            }
            if let info = infoMessage {
                Label(info, systemImage: "info.circle")
                    .font(.callout)
            }
            Section("Signal") {
                chart
                placementControls
            }
            Section {
                if model.labels.isEmpty {
                    Text("No steps yet. Add steps from the chart or choose Detect Candidate Steps in the Actions menu.")
                        .foregroundStyle(.secondary)
                }
                ForEach(model.labels) { label in
                    StepRow(label: label, isEditing: editMode.isEditing,
                            show: { show(label) },
                            edit: { editingLabel = label })
                        .swipeActions(edge: .leading) {
                            Button("Left") { model.setFoot(.left, for: [label.id]) }.tint(.blue)
                            Button("Right") { model.setFoot(.right, for: [label.id]) }.tint(.purple)
                        }
                        .swipeActions(edge: .trailing) {
                            Button("Delete", role: .destructive) { model.delete([label.id]) }
                            Button("Unknown") { model.setFoot(.unknown, for: [label.id]) }.tint(.gray)
                        }
                }
            } header: {
                Text("Steps (\(model.labels.count))")
            } footer: {
                Text("L, R and ? mark left, right and unknown. A dashed marker or \"Needs review\" means the label has not been checked by a person. Invalid steps stay in the sequence but are excluded from training.")
            }
        }
        .environment(\.editMode, $editMode)
        .toolbar { toolbar }
        .sheet(item: $editingLabel) { label in
            StepEditSheet(label: label, snap: model.snapped, onSave: model.replace, onDelete: { model.delete([label.id]) })
        }
        .sheet(isPresented: $showingAssisted) {
            AssistedLabelingSheet(
                labels: model.labels,
                defaultStartingFoot: model.metadata?.startingFoot ?? .left,
                onApply: { foot, index in
                    model.applyAssisted(startingFoot: foot, fromIndex: index)
                    infoMessage = "Proposed labels are marked Needs review. Check each one against the recording before relying on it."
                })
        }
        .confirmationDialog("Mark all labels as reviewed?", isPresented: $confirmingReviewAll, titleVisibility: .visible) {
            Button("I Have Checked Every Label") {
                model.markReviewed(Set(model.labels.map(\.id)))
            }
        } message: {
            Text("Only do this after checking every step's foot against the recording or video. Reviewed labels are used for training.")
        }
    }

    // MARK: Chart

    private var chart: some View {
        Chart {
            ForEach(model.gaps) { gap in
                RectangleMark(xStart: .value("Gap start", gap.start), xEnd: .value("Gap end", gap.end))
                    .foregroundStyle(.gray.opacity(0.25))
            }
            ForEach(model.chartPoints) { point in
                LineMark(x: .value("Time (s)", point.time), y: .value("Magnitude (g)", point.magnitude))
                    .lineStyle(StrokeStyle(lineWidth: 1))
                    .foregroundStyle(.secondary)
            }
            ForEach(model.labels) { label in
                RuleMark(x: .value("Step", label.timestamp))
                    .foregroundStyle(markerColor(label))
                    .lineStyle(StrokeStyle(lineWidth: 2, dash: label.reviewed ? [] : [4, 3]))
                // Letters as well as colors, so the foot is readable without relying on color.
                PointMark(x: .value("Step", label.timestamp), y: .value("Marker row", model.markerRowY))
                    .symbol {
                        Text(markerText(label))
                            .font(.caption.bold())
                            .foregroundStyle(markerColor(label))
                            .padding(2)
                            .background(.background, in: RoundedRectangle(cornerRadius: 3))
                    }
            }
            if let selectedTime {
                RuleMark(x: .value("Selected", selectedTime))
                    .foregroundStyle(.primary)
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [2, 2]))
            }
        }
        .chartXSelection(value: $selectedTime)
        .chartScrollableAxes(.horizontal)
        .chartXVisibleDomain(length: visibleSeconds)
        .chartScrollPosition(x: $scrollPosition)
        .chartYScale(domain: .automatic(includesZero: false))
        .chartXAxisLabel("Time since start (s)")
        .chartYAxisLabel("g")
        .frame(height: 240)
        .accessibilityLabel("Acceleration magnitude over time with step markers. Use the step list below to review steps.")
    }

    private func markerText(_ label: StepLabel) -> String {
        let letter: String
        switch label.footLabel {
        case .left: letter = "L"
        case .right: letter = "R"
        case .unknown: letter = "?"
        }
        return label.isValid ? letter : letter + "×"
    }

    private func markerColor(_ label: StepLabel) -> Color {
        guard label.isValid else { return .gray }
        switch label.footLabel {
        case .left: return .blue
        case .right: return .purple
        case .unknown: return .orange
        }
    }

    private var placementControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            Picker("Visible window", selection: $visibleSeconds) {
                ForEach([3.0, 6.0, 12.0, 30.0], id: \.self) { Text("\(Int($0)) s").tag($0) }
            }
            .pickerStyle(.segmented)

            if let selectedTime {
                HStack {
                    Text("Selected: \(Format.seconds(selectedTime))")
                        .font(.body.monospacedDigit())
                    Spacer()
                    Button("−10 ms") { self.selectedTime = selectedTime - 0.01 }
                    Button("+10 ms") { self.selectedTime = selectedTime + 0.01 }
                }
                .buttonStyle(.bordered)
            } else {
                Text("Press and drag on the chart to choose the moment a foot lands.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            Toggle("Snap to nearest peak", isOn: $snapToPeak)
            Text("Add a step at the selected time:")
                .font(.callout)
            HStack {
                ForEach([FootLabel.left, .right, .unknown]) { foot in
                    Button {
                        guard let selectedTime else { return }
                        model.addStep(at: selectedTime, foot: foot, snapToPeak: snapToPeak)
                    } label: {
                        Text(foot.displayName)
                            .lineLimit(1)
                            .minimumScaleFactor(0.8)
                            .frame(maxWidth: .infinity, minHeight: 36)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(selectedTime == nil)
                    .accessibilityLabel("Add \(foot.displayName) step at selected time")
                }
            }
        }
        .buttonStyle(.borderless)
    }

    private func show(_ label: StepLabel) {
        selectedTime = label.timestamp
        scrollPosition = max(0, label.timestamp - visibleSeconds / 2)
    }

    // MARK: Toolbar

    @ToolbarContentBuilder
    private var toolbar: some ToolbarContent {
        ToolbarItem(placement: .topBarTrailing) {
            Button(editMode.isEditing ? "Done" : "Select") {
                editMode = editMode.isEditing ? .inactive : .active
                selection.removeAll()
            }
        }
        ToolbarItem(placement: .topBarTrailing) {
            Menu {
                Button {
                    let added = model.detectCandidates()
                    infoMessage = added == 0
                        ? "No new candidate steps were found."
                        : "Added \(added) candidate step(s) with unknown foot. Assign each foot, or use Assisted Labeling and then review."
                } label: {
                    Label("Detect Candidate Steps", systemImage: "waveform.path.ecg")
                }
                Button {
                    showingAssisted = true
                } label: {
                    Label("Assisted Labeling…", systemImage: "arrow.left.arrow.right")
                }
                .disabled(model.labels.isEmpty)
                Button {
                    confirmingReviewAll = true
                } label: {
                    Label("Mark All Reviewed…", systemImage: "checkmark.seal")
                }
                .disabled(model.labels.allSatisfy(\.reviewed))
            } label: {
                Label("Actions", systemImage: "ellipsis.circle")
            }
        }
        if editMode.isEditing {
            ToolbarItemGroup(placement: .bottomBar) {
                Menu("Set Foot") {
                    ForEach(FootLabel.allCases) { foot in
                        Button(foot.displayName) { model.setFoot(foot, for: selection) }
                    }
                }
                Menu("Validity") {
                    Button("Mark Valid") { model.setValid(true, for: selection) }
                    Button("Mark Invalid") { model.setValid(false, for: selection) }
                }
                Button("Reviewed") { model.markReviewed(selection) }
                Button("Delete", role: .destructive) {
                    model.delete(selection)
                    selection.removeAll()
                }
            }
        }
    }
}

private struct StepRow: View {
    let label: StepLabel
    let isEditing: Bool
    let show: () -> Void
    let edit: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Step \(label.stepId)")
                    .font(.headline)
                Text(Format.seconds(label.timestamp))
                    .font(.subheadline.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                Label(label.footLabel.displayName, systemImage: footIcon)
                    .font(.headline)
                if !label.isValid {
                    Label("Invalid", systemImage: "xmark.circle").font(.caption)
                }
                if !label.reviewed {
                    Label("Needs review", systemImage: "eye.trianglebadge.exclamationmark").font(.caption)
                        .foregroundStyle(.orange)
                } else if label.labelSource == .assisted {
                    Text("Assisted, reviewed").font(.caption).foregroundStyle(.secondary)
                }
            }
            if !isEditing {
                Button(action: show) {
                    Image(systemName: "scope")
                }
                .accessibilityLabel("Show step \(label.stepId) in chart")
                Button(action: edit) {
                    Image(systemName: "slider.horizontal.3")
                }
                .accessibilityLabel("Edit step \(label.stepId)")
            }
        }
        .buttonStyle(.borderless)
        .imageScale(.large)
        .accessibilityElement(children: .contain)
    }

    private var footIcon: String {
        switch label.footLabel {
        case .left: return "l.circle"
        case .right: return "r.circle"
        case .unknown: return "questionmark.circle"
        }
    }
}
