import SwiftUI

/// A warning shown with an icon and text so it never depends on color alone.
struct WarningBanner: View {
    let text: String
    var systemImage = "exclamationmark.triangle.fill"

    var body: some View {
        Label {
            Text(text)
                .fixedSize(horizontal: false, vertical: true)
        } icon: {
            Image(systemName: systemImage)
                .foregroundStyle(.orange)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 12))
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Warning: \(text)")
    }
}

/// Full-width button with a comfortable touch target.
struct LargeButton: View {
    let title: String
    let systemImage: String
    var role: ButtonRole? = nil
    var prominent = true
    let action: () -> Void

    var body: some View {
        if prominent {
            button.buttonStyle(.borderedProminent)
        } else {
            button.buttonStyle(.bordered)
        }
    }

    private var button: some View {
        Button(role: role, action: action) {
            Label(title, systemImage: systemImage)
                .font(.headline)
                .frame(maxWidth: .infinity, minHeight: 44)
        }
        .controlSize(.large)
    }
}

extension RecordingStatus {
    var systemImage: String {
        switch self {
        case .inProgress: return "record.circle"
        case .completed: return "checkmark.circle"
        case .interrupted: return "exclamationmark.circle"
        case .failed: return "xmark.octagon"
        }
    }
}

enum Format {
    static func duration(_ seconds: Double) -> String {
        let total = max(0, seconds)
        let minutes = Int(total) / 60
        let rest = total - Double(minutes * 60)
        return String(format: "%02d:%04.1f", minutes, rest)
    }

    static func seconds(_ value: Double) -> String {
        String(format: "%.2f s", value)
    }
}
