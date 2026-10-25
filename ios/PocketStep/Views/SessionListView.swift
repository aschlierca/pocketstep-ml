import SwiftUI

struct SessionListView: View {
    @EnvironmentObject private var store: SessionStore
    @EnvironmentObject private var router: Router
    @State private var pendingDeletion: SessionMetadata?

    var body: some View {
        List {
            if store.sessions.isEmpty {
                ContentUnavailableView(
                    "No Sessions Yet",
                    systemImage: "figure.walk",
                    description: Text("Create a session to record labeled walking-in-place data."))
            }
            ForEach(store.sessions) { session in
                NavigationLink(value: Route.review(session.sessionId)) {
                    SessionRow(session: session)
                }
                .swipeActions {
                    Button("Delete", role: .destructive) { pendingDeletion = session }
                }
            }
            if let error = store.lastError {
                WarningBanner(text: error)
            }
        }
        .navigationTitle("PocketStep")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    router.path.append(.setup)
                } label: {
                    Label("New Session", systemImage: "plus")
                }
            }
        }
        .onAppear { store.reload() }
        .confirmationDialog(
            "Delete this session?",
            isPresented: Binding(get: { pendingDeletion != nil }, set: { if !$0 { pendingDeletion = nil } }),
            titleVisibility: .visible,
            presenting: pendingDeletion
        ) { session in
            Button("Delete \(session.sessionId)", role: .destructive) {
                store.delete(sessionId: session.sessionId)
            }
        } message: { _ in
            Text("The recording and its labels are removed from this phone. Exported copies are not affected.")
        }
    }
}

private struct SessionRow: View {
    let session: SessionMetadata

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(session.participantId)
                .font(.headline)
            Text(session.sessionId)
                .font(.subheadline.monospaced())
                .foregroundStyle(.secondary)
            HStack {
                Label(session.recordingStatus.displayName, systemImage: session.recordingStatus.systemImage)
                Text("·")
                Text("\(session.sampleCount) samples")
            }
            .font(.footnote)
            if let start = session.sessionStart {
                Text(start, format: .dateTime.year().month().day().hour().minute())
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
        .accessibilityElement(children: .combine)
    }
}
