import Foundation

/// Enforces mutual exclusivity and deterministic pipeline execution order across processors.
public enum ProcessorOrder {
    // ponytail: Static sequence is fixed to the 10 engine processors. Upgrade to a dynamic graph if dynamic multi-pass pipelines are needed.
    public static let executionOrder: [ProcessorKind] = [
        .faceSwapper,       // swap identity
        .deepSwapper,       // deep swap (mutually exclusive with faceSwapper)
        .ageModifier,       // age modification
        .expressionRestorer,// expression restoration
        .faceEditor,        // facial orientation and expression editing
        .faceEnhancer,      // facial detail enhancement
        .frameColorizer,    // colorization
        .frameEnhancer,     // 4x frame super-resolution / upscale
        .backgroundRemover, // background segmentation/matting
        .faceDebugger       // visual landmark/mask overlay debugging
    ]

    /// Toggles a processor while preserving the mutual exclusivity between Face Swapper and Deep Swapper.
    public static func toggle(
        _ kind: ProcessorKind,
        in selected: inout Set<ProcessorKind>
    ) {
        if selected.contains(kind) {
            selected.remove(kind)
        } else {
            // Mutual exclusivity constraint: faceSwapper and deepSwapper cannot run together
            if kind == .faceSwapper {
                selected.remove(.deepSwapper)
            } else if kind == .deepSwapper {
                selected.remove(.faceSwapper)
            }
            selected.insert(kind)
        }
    }

    /// Sorts selected processors into the deterministic pipeline sequence.
    public static func sort(_ selected: Set<ProcessorKind>) -> [ProcessorKind] {
        executionOrder.filter { selected.contains($0) }
    }
}
