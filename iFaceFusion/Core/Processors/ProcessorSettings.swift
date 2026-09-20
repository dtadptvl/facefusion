import Foundation
import UIKit

public enum DebuggerItem: String, CaseIterable, Sendable {
    case boundingBox = "bounding-box"
    case faceMask = "face-mask"
    case landmark5 = "face-landmark-5"
    case landmark68 = "face-landmark-68"
}

public struct FaceMaskSettings: Sendable {
    public var types: Set<FaceMaskType>
    public var blur: Float // 0.0 ... 1.0 (default: 0.3)
    public var padding: FaceMaskPadding
    public var areas: Set<FaceMaskArea>
    public var regions: Set<FaceMaskRegion>

    public init(
        types: Set<FaceMaskType> = [.box, .occlusion],
        blur: Float = 0.3,
        padding: FaceMaskPadding = FaceMaskPadding(),
        areas: Set<FaceMaskArea> = [.upperFace, .lowerFace, .mouth],
        regions: Set<FaceMaskRegion> = [.skin, .mouth, .upperLip, .lowerLip]
    ) {
        self.types = types
        self.blur = blur
        self.padding = padding
        self.areas = areas
        self.regions = regions
    }
}

public struct FaceSwapperSettings: Sendable {
    public var model: String // e.g. "hyperswap_1a_256", "inswapper_128"
    public var pixelBoost: String // "256x256", "512x512", "1024x1024"
    public var weight: Float // 0.0 ... 1.0 (default 0.5)

    public init(model: String = "hyperswap_1a_256", pixelBoost: String = "256x256", weight: Float = 0.5) {
        self.model = model
        self.pixelBoost = pixelBoost
        self.weight = weight
    }
}

public struct FaceEnhancerSettings: Sendable {
    public var model: String // e.g. "gfpgan_1.4", "codeformer", "restoreformer_plus_plus"
    public var blend: Int // 0 ... 100 (default 80)
    public var weight: Float // 0.0 ... 1.0 (default 0.5)

    public init(model: String = "gfpgan_1.4", blend: Int = 80, weight: Float = 0.5) {
        self.model = model
        self.blend = blend
        self.weight = weight
    }
}

public struct FrameEnhancerSettings: Sendable {
    public var model: String // e.g. "span_kendata_x4", "real_esrgan_x4"
    public var blend: Int // 0 ... 100 (default 80)

    public init(model: String = "span_kendata_x4", blend: Int = 80) {
        self.model = model
        self.blend = blend
    }
}

public struct FrameColorizerSettings: Sendable {
    public var model: String // e.g. "ddcolor", "deoldify"
    public var size: String // "256x256", "512x512"
    public var blend: Int // 0 ... 100 (default 100)

    public init(model: String = "ddcolor", size: String = "256x256", blend: Int = 100) {
        self.model = model
        self.size = size
        self.blend = blend
    }
}

public struct BackgroundRemoverSettings: Sendable {
    public var model: String // e.g. "modnet", "ben_2", "birefnet_portrait"
    public var fillColor: [UInt8] // [R, G, B, A]
    public var despillColor: [UInt8] // [R, G, B, A]

    public init(model: String = "modnet", fillColor: [UInt8] = [0, 0, 0, 0], despillColor: [UInt8] = [0, 0, 0, 0]) {
        self.model = model
        self.fillColor = fillColor
        self.despillColor = despillColor
    }
}

public struct AgeModifierSettings: Sendable {
    public var model: String // "fran", "styleganex_age"
    public var direction: Int // -100 ... 100 (default 0)

    public init(model: String = "fran", direction: Int = 0) {
        self.model = model
        self.direction = direction
    }
}

public struct ExpressionRestorerSettings: Sendable {
    public var model: String // "live_portrait"
    public var factor: Int // 0 ... 100 (default 80)
    public var areas: Set<FaceMaskArea> // upper-face, lower-face

    public init(model: String = "live_portrait", factor: Int = 80, areas: Set<FaceMaskArea> = [.upperFace, .lowerFace]) {
        self.model = model
        self.factor = factor
        self.areas = areas
    }
}

public struct FaceEditorSettings: Sendable {
    public var model: String // "live_portrait"
    public var eyebrowDirection: Float // -1.0 ... 1.0
    public var eyeGazeHorizontal: Float // -1.0 ... 1.0
    public var eyeGazeVertical: Float // -1.0 ... 1.0
    public var eyeOpenRatio: Float // -1.0 ... 1.0
    public var lipOpenRatio: Float // -1.0 ... 1.0
    public var mouthGrim: Float // -1.0 ... 1.0
    public var mouthPout: Float // -1.0 ... 1.0
    public var mouthPurse: Float // -1.0 ... 1.0
    public var mouthSmile: Float // -1.0 ... 1.0
    public var mouthPositionHorizontal: Float // -1.0 ... 1.0
    public var mouthPositionVertical: Float // -1.0 ... 1.0
    public var headPitch: Float // -1.0 ... 1.0
    public var headYaw: Float // -1.0 ... 1.0
    public var headRoll: Float // -1.0 ... 1.0

    public init(
        model: String = "live_portrait",
        eyebrowDirection: Float = 0,
        eyeGazeHorizontal: Float = 0,
        eyeGazeVertical: Float = 0,
        eyeOpenRatio: Float = 0,
        lipOpenRatio: Float = 0,
        mouthGrim: Float = 0,
        mouthPout: Float = 0,
        mouthPurse: Float = 0,
        mouthSmile: Float = 0,
        mouthPositionHorizontal: Float = 0,
        mouthPositionVertical: Float = 0,
        headPitch: Float = 0,
        headYaw: Float = 0,
        headRoll: Float = 0
    ) {
        self.model = model
        self.eyebrowDirection = eyebrowDirection
        self.eyeGazeHorizontal = eyeGazeHorizontal
        self.eyeGazeVertical = eyeGazeVertical
        self.eyeOpenRatio = eyeOpenRatio
        self.lipOpenRatio = lipOpenRatio
        self.mouthGrim = mouthGrim
        self.mouthPout = mouthPout
        self.mouthPurse = mouthPurse
        self.mouthSmile = mouthSmile
        self.mouthPositionHorizontal = mouthPositionHorizontal
        self.mouthPositionVertical = mouthPositionVertical
        self.headPitch = headPitch
        self.headYaw = headYaw
        self.headRoll = headRoll
    }
}

public struct DeepSwapperSettings: Sendable {
    public var model: String // e.g. "iperov/elon_musk_224" or custom path
    public var morph: Int // 0 ... 100 (default 100)

    public init(model: String = "iperov/elon_musk_224", morph: Int = 100) {
        self.model = model
        self.morph = morph
    }
}

public struct FaceDebuggerSettings: Sendable {
    public var items: Set<DebuggerItem>

    public init(items: Set<DebuggerItem> = [.boundingBox, .faceMask, .landmark5, .landmark68]) {
        self.items = items
    }
}

/// Comprehensive settings container encompassing all 10 still-image processors and mask controls.
public struct ProcessorSettings: Sendable {
    public var activeProcessors: [ProcessorKind]
    public var faceSwapper: FaceSwapperSettings
    public var faceEnhancer: FaceEnhancerSettings
    public var frameEnhancer: FrameEnhancerSettings
    public var frameColorizer: FrameColorizerSettings
    public var backgroundRemover: BackgroundRemoverSettings
    public var ageModifier: AgeModifierSettings
    public var expressionRestorer: ExpressionRestorerSettings
    public var faceEditor: FaceEditorSettings
    public var deepSwapper: DeepSwapperSettings
    public var faceDebugger: FaceDebuggerSettings
    public var mask: FaceMaskSettings

    public init(
        activeProcessors: [ProcessorKind] = [.faceSwapper],
        faceSwapper: FaceSwapperSettings = FaceSwapperSettings(),
        faceEnhancer: FaceEnhancerSettings = FaceEnhancerSettings(),
        frameEnhancer: FrameEnhancerSettings = FrameEnhancerSettings(),
        frameColorizer: FrameColorizerSettings = FrameColorizerSettings(),
        backgroundRemover: BackgroundRemoverSettings = BackgroundRemoverSettings(),
        ageModifier: AgeModifierSettings = AgeModifierSettings(),
        expressionRestorer: ExpressionRestorerSettings = ExpressionRestorerSettings(),
        faceEditor: FaceEditorSettings = FaceEditorSettings(),
        deepSwapper: DeepSwapperSettings = DeepSwapperSettings(),
        faceDebugger: FaceDebuggerSettings = FaceDebuggerSettings(),
        mask: FaceMaskSettings = FaceMaskSettings()
    ) {
        self.activeProcessors = activeProcessors
        self.faceSwapper = faceSwapper
        self.faceEnhancer = faceEnhancer
        self.frameEnhancer = frameEnhancer
        self.frameColorizer = frameColorizer
        self.backgroundRemover = backgroundRemover
        self.ageModifier = ageModifier
        self.expressionRestorer = expressionRestorer
        self.faceEditor = faceEditor
        self.deepSwapper = deepSwapper
        self.faceDebugger = faceDebugger
        self.mask = mask
    }
}
