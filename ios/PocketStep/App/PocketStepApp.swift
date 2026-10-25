import SwiftUI

@main
struct PocketStepApp: App {
    @StateObject private var store = SessionStore()
    @StateObject private var router = Router()

    var body: some Scene {
        WindowGroup {
            NavigationStack(path: $router.path) {
                SessionListView()
                    .navigationDestination(for: Route.self) { route in
                        destination(for: route)
                    }
            }
            .environmentObject(store)
            .environmentObject(router)
        }
    }

    @ViewBuilder
    private func destination(for route: Route) -> some View {
        switch route {
        case .setup:
            SessionSetupView()
        case .record(let request):
            RecordingView(request: request, store: store)
        case .review(let sessionId):
            SessionReviewView(sessionId: sessionId)
        case .label(let sessionId):
            LabelingView(sessionId: sessionId, store: store)
        }
    }
}
