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
            Self.applyDespillColor(to: &result, despillRGBA: settings.despillColor)
        }

        // 5. Composite foreground with fill color using alpha matte
        Self.applyFillColor(to: &result, mask: fullMask, fillRGBA: settings.fillColor)

        return result
    }

    // ponytail: Porter-Duff Over composite uses software scalar loops. Upgrade to vImage AlphaBlend / Accelerate if 4K background replacement needs 60fps throughput.
    /// Composites target image over fill color using straight-alpha "over" blending, preserving original alpha.
    public static func applyFillColor(to image: inout ImageBuffer, mask: FaceMask, fillRGBA: [UInt8]) {
        let fillAlpha = Float(fillRGBA[3]) / 255.0
        let fillR = Float(fillRGBA[0])
        let fillG = Float(fillRGBA[1])
        let fillB = Float(fillRGBA[2])

        for y in 0..<image.height {
            let rowOffset = y * image.width * 4
            for x in 0..<image.width {
                let idx = rowOffset + x * 4
                let origAlpha = Float(image.data[idx + 3]) / 255.0
                let matte = mask[x, y] // 1.0 = foreground, 0.0 = background

                // Effective foreground alpha: original alpha multiplied by matte
                let fgAlpha = origAlpha * matte
                let bgAlpha = fillAlpha

                // Porter-Duff Over: foreground over background
                let bgWeight = bgAlpha * (1.0 - fgAlpha)
                let alphaOut = fgAlpha + bgWeight

                let origR = Float(image.data[idx + 0])
                let origG = Float(image.data[idx + 1])
                let origB = Float(image.data[idx + 2])

                if alphaOut > 1e-6 {
                    let blendedR = (origR * fgAlpha + fillR * bgWeight) / alphaOut
                    let blendedG = (origG * fgAlpha + fillG * bgWeight) / alphaOut
                    let blendedB = (origB * fgAlpha + fillB * bgWeight) / alphaOut

                    image.data[idx + 0] = UInt8(min(max(blendedR.rounded(), 0), 255))
                    image.data[idx + 1] = UInt8(min(max(blendedG.rounded(), 0), 255))
                    image.data[idx + 2] = UInt8(min(max(blendedB.rounded(), 0), 255))
                    image.data[idx + 3] = UInt8(min(max((alphaOut * 255.0).rounded(), 0), 255))
                } else {
                    image.data[idx + 0] = 0
                    image.data[idx + 1] = 0
                    image.data[idx + 2] = 0
                    image.data[idx + 3] = 0
                }
            }
        }
    }

    /// Suppresses color spill for arbitrary key channels (green, blue, red) using dominant-channel removal.
    public static func applyDespillColor(to image: inout ImageBuffer, despillRGBA: [UInt8]) {
        let strength = Float(despillRGBA[3]) / 255.0
        guard strength > 0 else { return }

        let dR = Float(despillRGBA[0])
        let dG = Float(despillRGBA[1])
        let dB = Float(despillRGBA[2])

        enum DominantKey { case red, green, blue }
        let key: DominantKey
        if dG >= dR && dG >= dB {
            key = .green
        } else if dB >= dR && dB >= dG {
            key = .blue
        } else {
            key = .red
        }

        for i in 0..<(image.width * image.height) {
            let idx = i * 4
            let r = Float(image.data[idx + 0])
            let g = Float(image.data[idx + 1])
            let b = Float(image.data[idx + 2])

            switch key {
            case .green:
                let limit = max(r, b)
                if g > limit {
                    let despilled = g * (1.0 - strength) + limit * strength
                    image.data[idx + 1] = UInt8(min(max(despilled.rounded(), 0), 255))
                }
            case .blue:
                let limit = max(r, g)
                if b > limit {
                    let despilled = b * (1.0 - strength) + limit * strength
                    image.data[idx + 2] = UInt8(min(max(despilled.rounded(), 0), 255))
                }
            case .red:
                let limit = max(g, b)
                if r > limit {
                    let despilled = r * (1.0 - strength) + limit * strength
                    image.data[idx + 0] = UInt8(min(max(despilled.rounded(), 0), 255))
                }
            }
        }
    }
}
