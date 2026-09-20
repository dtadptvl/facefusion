import Foundation

/// Primary category of processor supported by the engine.
public enum ProcessorKind: String, CaseIterable, Identifiable, Sendable {
    case ageModifier = "age_modifier"
    case backgroundRemover = "background_remover"
    case deepSwapper = "deep_swapper"
    case expressionRestorer = "expression_restorer"
    case faceDebugger = "face_debugger"
    case faceEditor = "face_editor"
    case faceEnhancer = "face_enhancer"
    case faceSwapper = "face_swapper"
    case frameColorizer = "frame_colorizer"
    case frameEnhancer = "frame_enhancer"

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .ageModifier: return "Age Modifier"
        case .backgroundRemover: return "Background Remover"
        case .deepSwapper: return "Deep Swapper"
        case .expressionRestorer: return "Expression Restorer"
        case .faceDebugger: return "Face Debugger"
        case .faceEditor: return "Face Editor"
        case .faceEnhancer: return "Face Enhancer"
        case .faceSwapper: return "Face Swapper"
        case .frameColorizer: return "Frame Colorizer"
        case .frameEnhancer: return "Frame Enhancer"
        }
    }
}

/// Remote download endpoint for model artifacts.
public struct DownloadSource: Equatable, Sendable {
    public let url: URL
    public let provider: String // "github", "huggingface"

    public init(url: URL, provider: String = "github") {
        self.url = url
        self.provider = provider
    }
}

/// Metadata, licensing, architecture, and integrity specification for an on-device model.
public struct ModelMetadata: Identifiable, Sendable {
    public let id: String
    public let name: String
    public let processor: ProcessorKind?
    public let vendor: String
    public let license: String // e.g. OpenRAIL-AS, MIT, Apache-2.0, ResearchRAIL, Non-Commercial
    public let year: Int
    public let template: WarpTemplate?
    public let inputWidth: Int
    public let inputHeight: Int
    public let mean: [Float]
    public let std: [Float]
    public let isBGR: Bool
    public let precision: String // "fp32" or "fp16"
    public let sources: [DownloadSource]
    public let expectedCRC32: String // 8-character hex string matching FaceFusion .hash files
    public let expectedSHA256: String?
    public let outputScale: Int // For frame_enhancer models (e.g. 2, 4, 8)

    public init(
        id: String,
        name: String,
        processor: ProcessorKind?,
        vendor: String,
        license: String,
        year: Int,
        template: WarpTemplate? = nil,
        inputWidth: Int,
        inputHeight: Int,
        mean: [Float] = [0, 0, 0],
        std: [Float] = [1, 1, 1],
        isBGR: Bool = false,
        precision: String = "fp32",
        sources: [DownloadSource],
        expectedCRC32: String,
        expectedSHA256: String? = nil,
        outputScale: Int = 1
    ) {
        self.id = id
        self.name = name
        self.processor = processor
        self.vendor = vendor
        self.license = license
        self.year = year
        self.template = template
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        self.mean = mean
        self.std = std
        self.isBGR = isBGR
        self.precision = precision
        self.sources = sources
        self.expectedCRC32 = expectedCRC32.lowercased()
        self.expectedSHA256 = expectedSHA256?.lowercased()
        self.outputScale = outputScale
    }
}
