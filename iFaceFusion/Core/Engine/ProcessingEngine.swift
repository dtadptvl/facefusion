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

        let activeKinds = settings.activeProcessors.isEmpty ? [.faceSwapper] : settings.activeProcessors

        // Memory optimization: avoid unnecessary duplicate allocations for high-res inputs (e.g. 48MP).
        // Only clone targetBuffer if expressionRestorer needs a preserved referenceImage.
        let referenceBuffer = activeKinds.contains(.expressionRestorer) ? targetBuffer.clone() : targetBuffer
        var currentBuffer = activeKinds.contains(.expressionRestorer) ? targetBuffer : targetBuffer.clone()

        // 2. Strict face detection: exact one face required on target image even for non-face processors
        progress?(0.05, "Detecting target face landmarks...")
        guard let cg = targetBuffer.toCGImage() else {
            throw ProcessingEngineError.invalidTargetImage("Target CGImage conversion failed")
        }
        var targetFace = try faceDetector.detectSingleFace(in: cg)
        let originalReferenceFace = targetFace // Preserved for original-expression reference geometry

        // 3. Prepare source image and face if provided or if faceSwapper is active
        var sourceBuffer: ImageBuffer? = nil
        var sourceFace: FaceTarget? = nil
        if let srcImg = source {
            guard let srcBuf = ImageBuffer(image: srcImg) else {
                throw ProcessingEngineError.invalidSourceImage("Could not decode source image pixels")
            }
            guard let srcCG = srcBuf.toCGImage() else {
                throw ProcessingEngineError.invalidSourceImage("Source CGImage conversion failed")
            }
            progress?(0.10, "Extracting source identity face...")
            sourceBuffer = srcBuf
            sourceFace = try faceDetector.detectSingleFace(in: srcCG)
        } else if activeKinds.contains(.faceSwapper) {
            throw ProcessingEngineError.invalidSourceImage("Source image is required for Face Swapper")
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
                currentBuffer = try await faceEnhancer.process(
                    targetImage: currentBuffer,
                    targetFace: targetFace,
                    settings: settings.faceEnhancer,
                    maskSettings: settings.mask,
                    modelCache: modelCache,
                    ortBridge: ortBridge
                )

            case .frameEnhancer:
                let prevW = currentBuffer.width
                let prevH = currentBuffer.height
                currentBuffer = try await frameEnhancer.process(
                    targetImage: currentBuffer,
                    settings: settings.frameEnhancer,
                    modelCache: modelCache,
                    ortBridge: ortBridge
                )
                // Coordinate scaling: update target face coordinates when upscale changes image dimensions
                if currentBuffer.width != prevW || currentBuffer.height != prevH {
                    let scaleX = Float(currentBuffer.width) / Float(prevW)
                    let scaleY = Float(currentBuffer.height) / Float(prevH)
                    targetFace = targetFace.rescaled(scaleX: scaleX, scaleY: scaleY)
                }

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
                currentBuffer = try await ageModifier.process(
                    targetImage: currentBuffer,
                    targetFace: targetFace,
                    settings: settings.ageModifier,
                    maskSettings: settings.mask,
                    modelCache: modelCache,
                    ortBridge: ortBridge
                )

            case .expressionRestorer:
                currentBuffer = try await expressionRestorer.process(
                    referenceImage: referenceBuffer,
                    referenceFace: originalReferenceFace,
                    targetImage: currentBuffer,
                    targetFace: targetFace,
                    settings: settings.expressionRestorer,
                    maskSettings: settings.mask,
                    modelCache: modelCache,
                    ortBridge: ortBridge
                )

            case .faceEditor:
                currentBuffer = try await faceEditor.process(
                    targetImage: currentBuffer,
                    targetFace: targetFace,
                    settings: settings.faceEditor,
                    maskSettings: settings.mask,
                    modelCache: modelCache,
                    ortBridge: ortBridge
                )

            case .deepSwapper:
                let dfmURL = URL(fileURLWithPath: settings.deepSwapper.model)
                currentBuffer = try await deepSwapper.process(
                    targetImage: currentBuffer,
                    targetFace: targetFace,
                    modelURL: dfmURL,
                    settings: settings.deepSwapper,
                    maskSettings: settings.mask,
                    ortBridge: ortBridge
                )

            case .faceDebugger:
                currentBuffer = faceDebugger.process(
                    targetImage: currentBuffer,
                    targetFace: targetFace,
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
