import Foundation
import UIKit

/// Implements FaceFusion face_enhancer processor (GFPGAN, CodeFormer, RestoreFormer++, GPEN).
public final class FaceEnhancerProcessor: Sendable {
    public init() {}

    public func process(
        targetImage: ImageBuffer,
        targetFace: FaceTarget,
        settings: FaceEnhancerSettings,
        maskSettings: FaceMaskSettings,
        modelCache: ModelCache,
        ortBridge: ORTBridge
    ) async throws -> ImageBuffer {
        guard let metadata = ModelCatalog.model(for: settings.model) else {
            throw ORTBridgeError.sessionCreationFailed("Unknown face enhancer model: \(settings.model)")
        }

        let modelURL = try await modelCache.ensureModelDownloaded(metadata)
        let template = metadata.template ?? .ffhq512
        let cropW = metadata.inputWidth
        let cropH = metadata.inputHeight
        let dstTemplatePoints = template.targetPoints(width: Float(cropW), height: Float(cropH))
        let affineMatrix = ImageGeometry.estimateSimilarityMatrix(src: targetFace.landmark5.points, dst: dstTemplatePoints)

        let cropBuffer = targetImage.warpAffine(matrix: affineMatrix, cropWidth: cropW, cropHeight: cropH)

        // Preprocess: RGB normalized to [-1, 1]
        let inputTensor = cropBuffer.toFloatTensorNCHW(mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5], isBGR: false)
        var inputs: [String: TensorBuffer] = [
            "input": TensorBuffer(floatData: inputTensor, shape: [1, 3, cropH, cropW])
        ]

        if metadata.id.contains("codeformer") {
            inputs["weight"] = TensorBuffer(doubleData: [Double(settings.weight)], shape: [1])
        }

        let outputs = try await ortBridge.run(modelPath: modelURL.path, inputs: inputs)
        guard let outputTensor = (outputs["output"] ?? (outputs.count == 1 ? outputs.values.first : nil))?.floatData else {
            throw ORTBridgeError.inferenceFailed("Face enhancer returned empty output")
        }

        // Postprocess: [-1, 1] -> [0, 255]
        let enhancedCrop = ImageBuffer.fromFloatTensorNCHW(
            tensor: outputTensor,
            width: cropW,
            height: cropH,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            isBGR: false
        )

        // Blend with original crop based on blend slider (0...100)
        let blendFactor = Float(settings.blend) / 100.0
        let blendedCrop = cropBuffer.clone()
        blendedCrop.blend(with: enhancedCrop, alpha: blendFactor)

        let finalMask = try await ProcessorMasks.createCombinedMask(
            cropBuffer: cropBuffer,
            targetFace: targetFace,
            affineMatrix: affineMatrix,
            maskSettings: maskSettings,
            modelCache: modelCache,
            ortBridge: ortBridge
        )
        let resultImage = targetImage.clone()
        resultImage.pasteBack(crop: blendedCrop, mask: finalMask, matrix: affineMatrix)

        return resultImage
    }
}
