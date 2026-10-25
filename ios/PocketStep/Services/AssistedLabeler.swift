import Foundation

/// Proposes alternating foot labels. Proposals are written as `assisted` and unreviewed so they
/// are never mistaken for verified ground truth; reviewed labels are left untouched.
enum AssistedLabeler {
    /// - Parameters:
    ///   - labels: Steps in any order; the result is in time order.
    ///   - startingFoot: Foot of the step at `fromIndex`.
    ///   - fromIndex: First step (in time order) to which the alternation applies. Earlier steps are
    ///     unchanged, which lets the researcher restart the pattern after a missed or doubled step.
    static func propose(for labels: [StepLabel], startingFoot: Side, fromIndex: Int = 0) -> [StepLabel] {
        var sorted = StepLabelStore.normalized(labels)
        guard fromIndex < sorted.count else { return sorted }
        for index in max(0, fromIndex)..<sorted.count where !sorted[index].reviewed {
            let foot = (index - fromIndex).isMultiple(of: 2) ? startingFoot : startingFoot.opposite
            sorted[index].footLabel = FootLabel(foot)
            sorted[index].labelSource = .assisted
            sorted[index].reviewed = false
        }
        return sorted
    }
}
