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
        guard let metadata = ModelCatalog.model(for: settings.model) ?? ModelCatalog.ddcolor as ModelMetadata? else {
            throw ORTBridgeError.sessionCreationFailed("Unknown frame colorizer model: \(settings.model)")
        }

        let modelURL = try await modelCache.ensureModelDownloaded(metadata)
        let modelSize = settings.size == "512x512" ? 512 : 256

        // 1. Create grayscale luminance representation of target
        let origW = targetImage.width
        let origH = targetImage.height
        var grayBuffer = ImageBuffer(width: origW, height: origH)
        var origLuminance = [Float](repeating: 0, count: origW * origH)

        for i in 0..<(origW * origH) {
            let idx = i * 4
            let r = Float(targetImage.data[idx + 0])
            let g = Float(targetImage.data[idx + 1])
            let b = Float(targetImage.data[idx + 2])
            // ITU-R BT.601 standard luminance calculation
            let y = 0.299 * r + 0.587 * g + 0.114 * b
            let uy = UInt8(min(max(y, 0), 255))
            grayBuffer.data[idx + 0] = uy
            grayBuffer.data[idx + 1] = uy
            grayBuffer.data[idx + 2] = uy
            grayBuffer.data[idx + 3] = 255
            origLuminance[i] = y / 255.0
        }

        // 2. Downscale grayscale buffer to model input size
        let scaledGray = grayBuffer.warpAffine(
            matrix: AffineMatrix2x3(m00: Float(modelSize) / Float(origW), m01: 0, m02: 0, m10: 0, m11: Float(modelSize) / Float(origH), m12: 0),
            cropWidth: modelSize,
            cropHeight: modelSize
        )

        // 3. Preprocess for DDColor: RGB [0, 1] NCHW
        let inputTensor = scaledGray.toFloatTensorNCHW(mean: [0, 0, 0], std: [1, 1, 1], isBGR: false)
        let inputs = ["input": TensorBuffer(floatData: inputTensor, shape: [1, 3, modelSize, modelSize])]

        let outputs = try await ortBridge.run(modelPath: modelURL.path, inputs: inputs)
        guard let colorTensor = outputs.values.first?.floatData else {
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

        // 6. Merge predicted chroma with original target luminance
        var result = ImageBuffer(width: origW, height: origH)
        for i in 0..<(origW * origH) {
            let idx = i * 4
            let cr = Float(upscaledColor.data[idx + 0])
            let cg = Float(upscaledColor.data[idx + 1])
            let cb = Float(upscaledColor.data[idx + 2])
            let cLum = max(0.299 * cr + 0.587 * cg + 0.114 * cb, 1.0)
            let lumRatio = (origLuminance[i] * 255.0) / cLum

            result.data[idx + 0] = UInt8(min(max(cr * lumRatio, 0), 255))
            result.data[idx + 1] = UInt8(min(max(cg * lumRatio, 0), 255))
            result.data[idx + 2] = UInt8(min(max(cb * lumRatio, 0), 255))
            result.data[idx + 3] = 255
        }

        // 7. Blend with original image based on blend factor (0...100)
        let blendFactor = Float(settings.blend) / 100.0
        let finalImage = targetImage.clone()
        finalImage.blend(with: result, alpha: blendFactor)

        return finalImage
    }
}
