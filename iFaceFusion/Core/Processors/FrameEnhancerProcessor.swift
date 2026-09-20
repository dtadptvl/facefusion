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
        guard let metadata = ModelCatalog.model(for: settings.model) ?? ModelCatalog.spanKendataX4 as ModelMetadata? else {
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

                guard let outTileTensor = outputs.values.first?.floatData else {
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

                // Copy non-padding area to output buffer
                let copyW = min(tileW * scale, outW - x * scale)
                let copyH = min(tileH * scale, outH - y * scale)

                for cy in 0..<copyH {
                    let outRow = (y * scale + cy) * outW * 4
                    let tileRow = cy * upTileW * 4
                    for cx in 0..<copyW {
                        let outIdx = outRow + (x * scale + cx) * 4
                        let inIdx = tileRow + cx * 4
                        outputBuffer.data[outIdx + 0] = enhancedTile.data[inIdx + 0]
                        outputBuffer.data[outIdx + 1] = enhancedTile.data[inIdx + 1]
                        outputBuffer.data[outIdx + 2] = enhancedTile.data[inIdx + 2]
                        outputBuffer.data[outIdx + 3] = 255
                    }
                }

                x += step
            }
            y += step
        }

        return outputBuffer
    }
}
