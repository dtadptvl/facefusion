import Foundation
import UIKit

/// Implements FaceFusion frame_colorizer processor (DDColor, DeOldify) restoring color to monochrome frames.
public final class FrameColorizerProcessor: Sendable {
    public init() {}

    public func process(
        targetImage: ImageBuffer,
        settings: FrameColorizerSettings,
        modelCache: ModelCache,
        ortBridge: ORTBridge
    ) async throws -> ImageBuffer {
        guard let metadata = ModelCatalog.model(for: settings.model) else {
            throw ORTBridgeError.sessionCreationFailed("Unknown frame colorizer model: \(settings.model)")
        }

        let modelURL = try await modelCache.ensureModelDownloaded(metadata)
        let modelSize = settings.size == "512x512" ? 512 : 256

        let origW = targetImage.width
        let origH = targetImage.height

        // 1. Direct memory-bounded downscale to model resolution without allocating full-resolution intermediate copy
        var scaledGray = ImageBuffer(width: modelSize, height: modelSize)
        for gy in 0..<modelSize {
            let sy = min(max(Int(Float(gy) * Float(origH) / Float(modelSize)), 0), origH - 1)
            let sRow = sy * origW * 4
            let dRow = gy * modelSize * 4
            for gx in 0..<modelSize {
                let sx = min(max(Int(Float(gx) * Float(origW) / Float(modelSize)), 0), origW - 1)
                let sIdx = sRow + sx * 4
                let dIdx = dRow + gx * 4
                let r = Float(targetImage.data[sIdx + 0])
                let g = Float(targetImage.data[sIdx + 1])
                let b = Float(targetImage.data[sIdx + 2])
                let y = UInt8(min(max(0.299 * r + 0.587 * g + 0.114 * b, 0), 255))
                scaledGray.data[dIdx + 0] = y
                scaledGray.data[dIdx + 1] = y
                scaledGray.data[dIdx + 2] = y
                scaledGray.data[dIdx + 3] = 255
            }
        }

        // 2. Preprocess for DDColor: RGB [0, 1] NCHW
        let inputTensor = scaledGray.toFloatTensorNCHW(mean: [0, 0, 0], std: [1, 1, 1], isBGR: false)
        let inputs = ["input": TensorBuffer(floatData: inputTensor, shape: [1, 3, modelSize, modelSize])]

        let outputs = try await ortBridge.run(modelPath: modelURL.path, inputs: inputs)
        guard let colorTensor = (outputs["output"] ?? (outputs.count == 1 ? outputs.values.first : nil))?.floatData else {
            throw ORTBridgeError.inferenceFailed("Colorizer returned empty tensor")
        }

        // 4. Reconstruct color frame at model resolution
        let coloredModelBuffer = ImageBuffer.fromFloatTensorNCHW(
            tensor: colorTensor,
            width: modelSize,
            height: modelSize,
            mean: [0, 0, 0],
            std: [1, 1, 1],
            isBGR: false
        )

        // 5. Upscale colored chroma back to original target dimensions
        let upscaledColor = coloredModelBuffer.warpAffine(
            matrix: AffineMatrix2x3(m00: Float(origW) / Float(modelSize), m01: 0, m02: 0, m10: 0, m11: Float(origH) / Float(modelSize), m12: 0),
            cropWidth: origW,
            cropHeight: origH
        )

        // 5. Merge predicted chroma with original target luminance, preserving target alpha
        var result = ImageBuffer(width: origW, height: origH)
        for i in 0..<(origW * origH) {
            let idx = i * 4
            let origR = Float(targetImage.data[idx + 0])
            let origG = Float(targetImage.data[idx + 1])
            let origB = Float(targetImage.data[idx + 2])
            let origLum = 0.299 * origR + 0.587 * origG + 0.114 * origB

            let cr = Float(upscaledColor.data[idx + 0])
            let cg = Float(upscaledColor.data[idx + 1])
            let cb = Float(upscaledColor.data[idx + 2])
            let cLum = max(0.299 * cr + 0.587 * cg + 0.114 * cb, 1.0)
            let lumRatio = origLum / cLum

            result.data[idx + 0] = UInt8(min(max(cr * lumRatio, 0), 255))
            result.data[idx + 1] = UInt8(min(max(cg * lumRatio, 0), 255))
            result.data[idx + 2] = UInt8(min(max(cb * lumRatio, 0), 255))
            result.data[idx + 3] = targetImage.data[idx + 3]
        }

        // 7. Blend with original image based on blend factor (0...100)
        let blendFactor = Float(settings.blend) / 100.0
        let finalImage = targetImage.clone()
        finalImage.blend(with: result, alpha: blendFactor)

        return finalImage
    }
}
