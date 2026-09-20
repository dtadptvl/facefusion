import Foundation
import UIKit

public enum ProcessingEngineError: LocalizedError {
    case invalidSourceImage(String)
    case invalidTargetImage(String)
    case processingCancelled
    case executionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidSourceImage(let reason):
            return "Invalid source image: \(reason)"
        case .invalidTargetImage(let reason):
            return "Invalid target image: \(reason)"
        case .processingCancelled:
            return "Processing was cancelled"
        case .executionFailed(let reason):
            return "Processing engine execution failed: \(reason)"
        }
    }
}

/// Central high-level execution coordinator managing model lifecycle, Apple Vision face detection,
/// memory-bounded sequential inference, and image rendering.
public actor ProcessingEngine {
    public static let shared = ProcessingEngine()

    public let modelCache: ModelCache
    public let ortBridge: ORTBridge
    public let faceDetector: FaceDetector

    // Individual processors
    private let faceSwapper = FaceSwapperProcessor()
    private let faceEnhancer = FaceEnhancerProcessor()
    private let frameEnhancer = FrameEnhancerProcessor()
    private let frameColorizer = FrameColorizerProcessor()
    private let backgroundRemover = BackgroundRemoverProcessor()
    private let ageModifier = AgeModifierProcessor()
    private let expressionRestorer = ExpressionRestorerProcessor()
    private let faceEditor = FaceEditorProcessor()
    private let deepSwapper = DeepSwapperProcessor()
    private let faceDebugger = FaceDebuggerProcessor()

    public init(customCacheDirectory: URL? = nil) {
        self.modelCache = ModelCache(customCacheDirectory: customCacheDirectory)
        self.ortBridge = ORTBridge.shared
        self.faceDetector = FaceDetector.shared
    }

    /// Primary processing API executing requested processors sequentially on the target image.
    ///
    /// - Parameters:
    ///   - source: Optional source identity image (required when `faceSwapper` is active).
    ///   - target: Target still image to process.
    ///   - settings: Complete configuration for active processors and masking options.
    ///   - progress: Optional progress reporting callback `(fractionCompleted, statusMessage)`.
    /// - Returns: Processed output image preserving full target dimensions (or upscaled by frame_enhancer).
    public func process(
        source: UIImage? = nil,
        target: UIImage,
        settings: ProcessorSettings,
        progress: (@Sendable (Double, String) -> Void)? = nil
    ) async throws -> UIImage {
        if Task.isCancelled { throw ProcessingEngineError.processingCancelled }

        // 1. Ingest and normalize target image
        guard let targetBuffer = ImageBuffer(image: target) else {
            throw ProcessingEngineError.invalidTargetImage("Could not decode target image pixels")
        }

        var currentBuffer = targetBuffer.clone()
        let activeKinds = settings.activeProcessors.isEmpty ? [.faceSwapper] : settings.activeProcessors

        // 2. Determine if any active processor requires face landmarks
        let requiresFace = activeKinds.contains { kind in
            switch kind {
            case .faceSwapper, .faceEnhancer, .ageModifier, .expressionRestorer, .faceEditor, .deepSwapper, .faceDebugger:
                return true
            case .frameEnhancer, .frameColorizer, .backgroundRemover:
                return false
            }
        }

        var targetFace: FaceTarget? = nil
        if requiresFace {
            progress?(0.05, "Detecting target face landmarks...")
            guard let cg = targetBuffer.toCGImage() else {
                throw ProcessingEngineError.invalidTargetImage("Target CGImage conversion failed")
            }
            targetFace = try faceDetector.detectSingleFace(in: cg)
        }

        // 3. Prepare source image and face if faceSwapper is active
        var sourceBuffer: ImageBuffer? = nil
        var sourceFace: FaceTarget? = nil
        if activeKinds.contains(.faceSwapper) {
            guard let srcImg = source else {
                throw ProcessingEngineError.invalidSourceImage("Source image is required for Face Swapper")
            }
            guard let srcBuf = ImageBuffer(image: srcImg) else {
                throw ProcessingEngineError.invalidSourceImage("Could not decode source image pixels")
            }
            guard let srcCG = srcBuf.toCGImage() else {
                throw ProcessingEngineError.invalidSourceImage("Source CGImage conversion failed")
            }
            progress?(0.10, "Extracting source identity face...")
            sourceBuffer = srcBuf
            sourceFace = try faceDetector.detectSingleFace(in: srcCG)
        }

        // 4. Sequential execution of active processors
        let totalSteps = Double(activeKinds.count)
        for (index, kind) in activeKinds.enumerated() {
            if Task.isCancelled { throw ProcessingEngineError.processingCancelled }

            let startFraction = 0.15 + (Double(index) / totalSteps) * 0.80
            progress?(startFraction, "Running \(kind.displayName)...")

            switch kind {
            case .faceSwapper:
                guard let sBuf = sourceBuffer, let sFace = sourceFace, let tFace = targetFace else {
                    throw ProcessingEngineError.executionFailed("Face Swapper missing source or target face")
                }
                currentBuffer = try await faceSwapper.process(
                    sourceImage: sBuf,
                    sourceFace: sFace,
                    targetImage: currentBuffer,
                    targetFace: tFace,
                    settings: settings.faceSwapper,
                    maskSettings: settings.mask,
                    modelCache: modelCache,
                    ortBridge: ortBridge
                )

            case .faceEnhancer:
                guard let tFace = targetFace else {
                    throw ProcessingEngineError.executionFailed("Face Enhancer requires detected target face")
                }
                currentBuffer = try await faceEnhancer.process(
                    targetImage: currentBuffer,
                    targetFace: tFace,
                    settings: settings.faceEnhancer,
                    maskSettings: settings.mask,
                    modelCache: modelCache,
                    ortBridge: ortBridge
                )

            case .frameEnhancer:
                currentBuffer = try await frameEnhancer.process(
                    targetImage: currentBuffer,
                    settings: settings.frameEnhancer,
                    modelCache: modelCache,
                    ortBridge: ortBridge
                )

            case .frameColorizer:
                currentBuffer = try await frameColorizer.process(
                    targetImage: currentBuffer,
                    settings: settings.frameColorizer,
                    modelCache: modelCache,
                    ortBridge: ortBridge
                )

            case .backgroundRemover:
                currentBuffer = try await backgroundRemover.process(
                    targetImage: currentBuffer,
                    settings: settings.backgroundRemover,
                    modelCache: modelCache,
                    ortBridge: ortBridge
                )

            case .ageModifier:
                guard let tFace = targetFace else {
                    throw ProcessingEngineError.executionFailed("Age Modifier requires detected target face")
                }
                currentBuffer = try await ageModifier.process(
                    targetImage: currentBuffer,
                    targetFace: tFace,
                    settings: settings.ageModifier,
                    maskSettings: settings.mask,
                    modelCache: modelCache,
                    ortBridge: ortBridge
                )

            case .expressionRestorer:
                guard let tFace = targetFace else {
                    throw ProcessingEngineError.executionFailed("Expression Restorer requires detected target face")
                }
                currentBuffer = try await expressionRestorer.process(
                    referenceImage: targetBuffer,
                    targetImage: currentBuffer,
                    targetFace: tFace,
                    settings: settings.expressionRestorer,
                    maskSettings: settings.mask,
                    modelCache: modelCache,
                    ortBridge: ortBridge
                )

            case .faceEditor:
                guard let tFace = targetFace else {
                    throw ProcessingEngineError.executionFailed("Face Editor requires detected target face")
                }
                currentBuffer = try await faceEditor.process(
                    targetImage: currentBuffer,
                    targetFace: tFace,
                    settings: settings.faceEditor,
                    maskSettings: settings.mask,
                    modelCache: modelCache,
                    ortBridge: ortBridge
                )

            case .deepSwapper:
                guard let tFace = targetFace else {
                    throw ProcessingEngineError.executionFailed("Deep Swapper requires detected target face")
                }
                let dfmURL = URL(fileURLWithPath: settings.deepSwapper.model)
                currentBuffer = try await deepSwapper.process(
                    targetImage: currentBuffer,
                    targetFace: tFace,
                    modelURL: dfmURL,
                    settings: settings.deepSwapper,
                    maskSettings: settings.mask,
                    ortBridge: ortBridge
                )

            case .faceDebugger:
                guard let tFace = targetFace else {
                    throw ProcessingEngineError.executionFailed("Face Debugger requires detected target face")
                }
                currentBuffer = faceDebugger.process(
                    targetImage: currentBuffer,
                    targetFace: tFace,
                    settings: settings.faceDebugger,
                    maskSettings: settings.mask
                )
            }

            // Memory boundary: release active session to ensure peak memory is bounded
            await ortBridge.releaseSession()
        }

        progress?(1.0, "Processing complete.")

        guard let resultUI = currentBuffer.toUIImage() else {
            throw ProcessingEngineError.executionFailed("Failed to create final UIImage")
        }
        return resultUI
    }
}
