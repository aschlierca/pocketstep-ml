import SwiftUI

struct RecordingRequest: Hashable {
    var metadata: SessionMetadata
    var countdownSeconds: Int
}

enum Route: Hashable {
    case setup
    case record(RecordingRequest)
    case review(String)
    case label(String)
}

@MainActor
final class Router: ObservableObject {
    @Published var path: [Route] = []

    func popToRoot() {
        path.removeAll()
    }

    /// Replaces the stack so Back from labeling returns to the session review, not to recording.
    func showLabeling(sessionId: String) {
        path = [.review(sessionId), .label(sessionId)]
    }

    func showReview(sessionId: String) {
        path = [.review(sessionId)]
    }
}
