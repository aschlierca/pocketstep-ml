import SwiftUI

struct RecordingRequest: Hashable {
    var metadata: SessionMetadata
    var countdownSeconds: Int
}

enum Route: Hashable {
    case setup
    case record(RecordingRequest)
}

@MainActor
final class Router: ObservableObject {
    @Published var path: [Route] = []

    func popToRoot() {
        path.removeAll()
    }
}
