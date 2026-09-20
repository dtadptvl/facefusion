import Foundation
import UIKit

/// Implements FaceFusion background_remover processor (MODNet, Ben2, BiRefNet, RMBG, U2Net).
public final class BackgroundRemoverProcessor: Sendable {
    public init() {}

    public func process(
        targetImage: ImageBuffer,
        settings: BackgroundRemoverSettings,
        modelCache: ModelCache,
        ortBridge: ORTBridge
    ) async throws -> ImageBuffer {
        guard let metadata = ModelCatalog.model(for: settings.model) else {
            throw ORTBridgeError.sessionCreationFailed("Unknown background remover model: \(settings.model)")
        }

        let modelURL = try await modelCache.ensureModelDownloaded(metadata)
        let origW = targetImage.width
        let origH = targetImage.height
        let modelW = metadata.inputWidth
        let modelH = metadata.inputHeight

        // 1. Resize input to model dimensions
        let scaledInput = targetImage.warpAffine(
            matrix: AffineMatrix2x3(m00: Float(modelW) / Float(origW), m01: 0, m02: 0, m10: 0, m11: Float(modelH) / Float(origH), m12: 0),
            cropWidth: modelW,
            cropHeight: modelH
        )

        // 2. Preprocess: RGB normalized to mean/std
        let tensor = scaledInput.toFloatTensorNCHW(mean: metadata.mean, std: metadata.std, isBGR: false)
        let inputs = ["input": TensorBuffer(floatData: tensor, shape: [1, 3, modelH, modelW])]

        let outputs = try await ortBridge.run(modelPath: modelURL.path, inputs: inputs)
        guard let maskTensor = (outputs["output"] ?? (outputs.count == 1 ? outputs.values.first : nil))?.floatData else {
            throw ORTBridgeError.inferenceFailed("Background remover returned empty output tensor")
        }

        // 3. Reconstruct alpha mask and resize to original dimensions
        let rawMask = FaceMask(width: modelW, height: modelH, values: maskTensor.map { min(max($0, 0), 1) })
        let fullMask = rawMask.resized(toWidth: origW, toHeight: origH)

        // 4. Apply despill color if configured
        var result = targetImage.clone()
        if settings.despillColor[3] > 0 {
            applyDespillColor(to: &result, despillRGBA: settings.despillColor)
        }

        // 5. Composite foreground with fill color using alpha matte
        applyFillColor(to: &result, mask: fullMask, fillRGBA: settings.fillColor)

        return result
    }

    private func applyFillColor(to image: inout ImageBuffer, mask: FaceMask, fillRGBA: [UInt8]) {
        let fillAlpha = Float(fillRGBA[3]) / 255.0
        let fillR = Float(fillRGBA[0])
        let fillG = Float(fillRGBA[1])
        let fillB = Float(fillRGBA[2])

        for y in 0..<image.height {
            let rowOffset = y * image.width * 4
            for x in 0..<image.width {
                let idx = rowOffset + x * 4
                let fgAlpha = mask[x, y] // 1.0 = foreground, 0.0 = background
                let bgWeight = (1.0 - fgAlpha) * fillAlpha

                let origR = Float(image.data[idx + 0])
                let origG = Float(image.data[idx + 1])
                let origB = Float(image.data[idx + 2])

                let blendedR = origR * (1.0 - bgWeight) + fillR * bgWeight
                let blendedG = origG * (1.0 - bgWeight) + fillG * bgWeight
                let blendedB = origB * (1.0 - bgWeight) + fillB * bgWeight
                let alphaOut = fillRGBA[3] == 0 ? UInt8(min(max(fgAlpha * 255.0, 0), 255)) : 255

                image.data[idx + 0] = UInt8(min(max(blendedR, 0), 255))
                image.data[idx + 1] = UInt8(min(max(blendedG, 0), 255))
                image.data[idx + 2] = UInt8(min(max(blendedB, 0), 255))
                image.data[idx + 3] = alphaOut
            }
        }
    }

    private func applyDespillColor(to image: inout ImageBuffer, despillRGBA: [UInt8]) {
        let alpha = Float(despillRGBA[3]) / 255.0
        guard alpha > 0 else { return }

        let targetG = Float(despillRGBA[1])
        for i in 0..<(image.width * image.height) {
            let idx = i * 4
            let r = Float(image.data[idx + 0])
            let g = Float(image.data[idx + 1])
            let b = Float(image.data[idx + 2])

            // If green spill is detected (g > max(r, b))
            let maxRB = max(r, b)
            if g > maxRB && targetG > 128 {
                let despilledG = g + (maxRB - g) * alpha
                image.data[idx + 1] = UInt8(min(max(despilledG, 0), 255))
            }
        }
    }
}
