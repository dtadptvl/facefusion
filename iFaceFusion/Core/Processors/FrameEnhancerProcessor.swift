import Foundation
import UIKit

/// Implements FaceFusion frame_enhancer processor (SPAN, Real-ESRGAN) with memory-bounded tile inference.
public final class FrameEnhancerProcessor: Sendable {
    public init() {}

    public func process(
        targetImage: ImageBuffer,
        settings: FrameEnhancerSettings,
        modelCache: ModelCache,
        ortBridge: ORTBridge
    ) async throws -> ImageBuffer {
        guard let metadata = ModelCatalog.model(for: settings.model) else {
            throw ORTBridgeError.sessionCreationFailed("Unknown frame enhancer model: \(settings.model)")
        }

        let modelURL = try await modelCache.ensureModelDownloaded(metadata)
        let scale = max(1, metadata.outputScale)
        let tileSize = metadata.inputWidth > 0 ? metadata.inputWidth : 128
        let pad = 16 // Tile overlap padding to prevent seam artifacts

        let srcW = targetImage.width
        let srcH = targetImage.height
        let outW = srcW * scale
        let outH = srcH * scale

        // Memory budget check: prohibit allocating beyond 64 megapixels (~256MB) to guard against iOS OOM crashes
        let maxPixelBudget = 64_000_000
        if outW * outH > maxPixelBudget {
            throw ORTBridgeError.inferenceFailed("Frame enhancer output \(outW)x\(outH) (\(scale)x upscale of \(srcW)x\(srcH)) exceeds bounded memory budget of \(maxPixelBudget) pixels")
        }

        var outputBuffer = ImageBuffer(width: outW, height: outH)

        // Process grid tiles sequentially
        let step = tileSize - 2 * pad
        var y = 0
        while y < srcH {
            let tileH = min(tileSize, srcH - y)
            var x = 0
            while x < srcW {
                let tileW = min(tileSize, srcW - x)

                // Extract tile and pad to tileSize x tileSize if at edge
                var tile = ImageBuffer(width: tileSize, height: tileSize)
                for ty in 0..<tileH {
                    let srcRow = (y + ty) * srcW * 4
                    let dstRow = ty * tileSize * 4
                    for tx in 0..<tileW {
                        let sIdx = srcRow + (x + tx) * 4
                        let dIdx = dstRow + tx * 4
                        tile.data[dIdx + 0] = targetImage.data[sIdx + 0]
                        tile.data[dIdx + 1] = targetImage.data[sIdx + 1]
                        tile.data[dIdx + 2] = targetImage.data[sIdx + 2]
                        tile.data[dIdx + 3] = targetImage.data[sIdx + 3]
                    }
                }

                // Forward pass on tile: RGB [0, 1] NCHW
                let tileTensor = tile.toFloatTensorNCHW(mean: [0, 0, 0], std: [1, 1, 1], isBGR: false)
                let inputs = ["input": TensorBuffer(floatData: tileTensor, shape: [1, 3, tileSize, tileSize])]
                let outputs = try await ortBridge.run(modelPath: modelURL.path, inputs: inputs)

                guard let outTileTensor = (outputs["output"] ?? (outputs.count == 1 ? outputs.values.first : nil))?.floatData else {
                    throw ORTBridgeError.inferenceFailed("Tile enhancement returned empty tensor")
                }

                let upTileW = tileSize * scale
                let upTileH = tileSize * scale
                let enhancedTile = ImageBuffer.fromFloatTensorNCHW(
                    tensor: outTileTensor,
                    width: upTileW,
                    height: upTileH,
                    mean: [0, 0, 0],
                    std: [1, 1, 1],
                    isBGR: false
                )

                // Copy non-padding area to output buffer while strictly preserving original alpha
                let copyW = min(tileW * scale, outW - x * scale)
                let copyH = min(tileH * scale, outH - y * scale)

                for cy in 0..<copyH {
                    let outRow = (y * scale + cy) * outW * 4
                    let tileRow = cy * upTileW * 4
                    let srcY = min(max(y + cy / scale, 0), srcH - 1)
                    let srcRow = srcY * srcW * 4
                    for cx in 0..<copyW {
                        let outIdx = outRow + (x * scale + cx) * 4
                        let inIdx = tileRow + cx * 4
                        let srcX = min(max(x + cx / scale, 0), srcW - 1)
                        let srcAlpha = targetImage.data[srcRow + srcX * 4 + 3]

                        outputBuffer.data[outIdx + 0] = enhancedTile.data[inIdx + 0]
                        outputBuffer.data[outIdx + 1] = enhancedTile.data[inIdx + 1]
                        outputBuffer.data[outIdx + 2] = enhancedTile.data[inIdx + 2]
                        outputBuffer.data[outIdx + 3] = srcAlpha
                    }
                }

                x += step
            }
            y += step
        }

        // Apply blend control if blend < 100
        if settings.blend < 100 {
            let blendFactor = Float(settings.blend) / 100.0
            let invBlend = 1.0 - blendFactor
            for py in 0..<outH {
                let outRow = py * outW * 4
                let srcY = min(max(py / scale, 0), srcH - 1)
                let srcRow = srcY * srcW * 4
                for px in 0..<outW {
                    let outIdx = outRow + px * 4
                    let srcX = min(max(px / scale, 0), srcW - 1)
                    let srcIdx = srcRow + srcX * 4
                    for c in 0..<3 {
                        let enhancedVal = Float(outputBuffer.data[outIdx + c])
                        let origVal = Float(targetImage.data[srcIdx + c])
                        let blended = enhancedVal * blendFactor + origVal * invBlend
                        outputBuffer.data[outIdx + c] = UInt8(min(max(blended, 0), 255))
                    }
                }
            }
        }

        return outputBuffer
    }
}
