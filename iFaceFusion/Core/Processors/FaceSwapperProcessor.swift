import Foundation
import UIKit
import simd

/// Implements FaceFusion face_swapper processor with real ONNX inference and ArcFace embedding integration.
public final class FaceSwapperProcessor: Sendable {
    public init() {}

    public func process(
        sourceImage: ImageBuffer,
        sourceFace: FaceTarget,
        targetImage: ImageBuffer,
        targetFace: FaceTarget,
        settings: FaceSwapperSettings,
        maskSettings: FaceMaskSettings,
        modelCache: ModelCache,
        ortBridge: ORTBridge
    ) async throws -> ImageBuffer {
        // 1. Ensure ArcFace recognizer model is downloaded
        let arcfaceModel = ModelCatalog.arcfaceW600kR50
        let arcfaceURL = try await modelCache.ensureModelDownloaded(arcfaceModel)

        // 2. Extract source face 512D normalized embedding via ArcFace
        let sourceEmbedding = try await extractEmbedding(
            from: sourceImage,
            landmark5: sourceFace.landmark5,
            modelURL: arcfaceURL,
            metadata: arcfaceModel,
            ortBridge: ortBridge
        )

        // 3. Extract target face embedding for balance interpolation
        let targetEmbedding = try await extractEmbedding(
            from: targetImage,
            landmark5: targetFace.landmark5,
            modelURL: arcfaceURL,
            metadata: arcfaceModel,
            ortBridge: ortBridge
        )

        // 4. Ensure swapper model is downloaded (strictly reject unknown model without false fallback)
        guard let swapperMetadata = ModelCatalog.model(for: settings.model) else {
            throw ORTBridgeError.sessionCreationFailed("Unknown swapper model: \(settings.model)")
        }
        let swapperURL = try await modelCache.ensureModelDownloaded(swapperMetadata)

        // 5. Warp target face into swapper template coordinate space with pixel boost support
        let template = swapperMetadata.template ?? .arcface128
        let cropW = swapperMetadata.inputWidth
        let cropH = swapperMetadata.inputHeight

        let boostDim = PixelBoost.parseBoostDimension(settings.pixelBoost, fallback: cropW)
        let total = max(1, boostDim / cropW)
        let boostW = cropW * total
        let boostH = cropH * total

        let dstTemplatePoints = template.targetPoints(width: Float(boostW), height: Float(boostH))
        let affineMatrix = ImageGeometry.estimateSimilarityMatrix(src: targetFace.landmark5.points, dst: dstTemplatePoints)

        let cropBuffer = targetImage.warpAffine(matrix: affineMatrix, cropWidth: boostW, cropHeight: boostH)

        // 6. Blend embedding using weight matching upstream: interp(weight, [0, 1], [0.35, -0.35])
        let w = LivePortraitGeometry.interp(x: settings.weight, xp0: 0, xp1: 1, fp0: 0.35, fp1: -0.35)
        var blendedEmbedding = [Float](repeating: 0, count: 512)
        for i in 0..<512 {
            blendedEmbedding[i] = sourceEmbedding[i] * (1.0 - w) + targetEmbedding[i] * w
        }

        // 7. Implode high-resolution crop into interleaved sub-frames of model size
        let subFrames = PixelBoost.implode(crop: cropBuffer, total: total, modelWidth: cropW, modelHeight: cropH)
        var swappedSubFrames = [ImageBuffer]()
        swappedSubFrames.reserveCapacity(subFrames.count)

        // 8. Execute ONNX inference per sub-frame
        for subFrame in subFrames {
            let subTensor = subFrame.toFloatTensorNCHW(mean: swapperMetadata.mean, std: swapperMetadata.std, isBGR: swapperMetadata.isBGR)
            let inputs: [String: TensorBuffer] = [
                "source": TensorBuffer(floatData: blendedEmbedding, shape: [1, 512]),
                "target": TensorBuffer(floatData: subTensor, shape: [1, 3, cropH, cropW])
            ]

            let outputs = try await ortBridge.run(modelPath: swapperURL.path, inputs: inputs)
            // HyperSwap produces "output" AND "mask"; strictly use "output" name to avoid picking mask
            guard let swappedTensor = (outputs["output"] ?? (outputs.count == 1 ? outputs.values.first : nil))?.floatData else {
                throw ORTBridgeError.inferenceFailed("Swapper returned empty output tensor")
            }

            let swappedSub = ImageBuffer.fromFloatTensorNCHW(
                tensor: swappedTensor,
                width: cropW,
                height: cropH,
                mean: swapperMetadata.mean,
                std: swapperMetadata.std,
                isBGR: swapperMetadata.isBGR
            )
            swappedSubFrames.append(swappedSub)
        }

        // 9. Explode interleaved sub-frames back into full boost resolution
        let swappedCrop = PixelBoost.explode(
            subFrames: swappedSubFrames,
            total: total,
            boostWidth: boostW,
            boostHeight: boostH,
            modelWidth: cropW,
            modelHeight: cropH
        )

        // 10. Generate combined mask matching active types (box, occlusion, region, area)
        let finalMask = try await ProcessorMasks.createCombinedMask(
            cropBuffer: cropBuffer,
            targetFace: targetFace,
            affineMatrix: affineMatrix,
            maskSettings: maskSettings,
            modelCache: modelCache,
            ortBridge: ortBridge
        )

        // 11. Paste back into target buffer
        let resultImage = targetImage.clone()
        resultImage.pasteBack(crop: swappedCrop, mask: finalMask, matrix: affineMatrix)

        return resultImage
    }

    private func extractEmbedding(
        from image: ImageBuffer,
        landmark5: FaceLandmark5,
        modelURL: URL,
        metadata: ModelMetadata,
        ortBridge: ORTBridge
    ) async throws -> [Float] {
        let template = metadata.template ?? .arcface112v2
        let cropSize = 112
        let targetPoints = template.targetPoints(width: Float(cropSize), height: Float(cropSize))
        let matrix = ImageGeometry.estimateSimilarityMatrix(src: landmark5.points, dst: targetPoints)
        let crop = image.warpAffine(matrix: matrix, cropWidth: cropSize, cropHeight: cropSize)

        // ArcFace input: RGB [-1, 1]
        let tensor = crop.toFloatTensorNCHW(mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5], isBGR: false)
        let inputs = ["input": TensorBuffer(floatData: tensor, shape: [1, 3, cropSize, cropSize])]
        let outputs = try await ortBridge.run(modelPath: modelURL.path, inputs: inputs)

        guard let rawEmbedding = outputs.values.first?.floatData, rawEmbedding.count >= 512 else {
            throw ORTBridgeError.inferenceFailed("ArcFace embedding output invalid")
        }

        // L2-normalize embedding vector
        var normSq: Float = 0
        for i in 0..<512 { normSq += rawEmbedding[i] * rawEmbedding[i] }
        let norm = sqrt(max(normSq, 1e-12))
        return (0..<512).map { rawEmbedding[$0] / norm }
    }
}
