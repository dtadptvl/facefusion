import Foundation
import UIKit

/// Implements FaceFusion age_modifier processor (FRAN, StyleGANEX) altering apparent age.
public final class AgeModifierProcessor: Sendable {
    public init() {}

    public func process(
        targetImage: ImageBuffer,
        targetFace: FaceTarget,
        settings: AgeModifierSettings,
        maskSettings: FaceMaskSettings,
        modelCache: ModelCache,
        ortBridge: ORTBridge
    ) async throws -> ImageBuffer {
        guard let metadata = ModelCatalog.model(for: settings.model) ?? ModelCatalog.fran as ModelMetadata? else {
            throw ORTBridgeError.sessionCreationFailed("Unknown age modifier model: \(settings.model)")
        }

        let modelURL = try await modelCache.ensureModelDownloaded(metadata)
        let template = metadata.template ?? .ffhq512
        let cropW = metadata.inputWidth
        let cropH = metadata.inputHeight
        let dstTemplatePoints = template.targetPoints(width: Float(cropW), height: Float(cropH))
        let affineMatrix = ImageGeometry.estimateSimilarityMatrix(src: targetFace.landmark5.points, dst: dstTemplatePoints)

        let cropBuffer = targetImage.warpAffine(matrix: affineMatrix, cropWidth: cropW, cropHeight: cropH)

        // Calculate age direction vector matching upstream FRAN contract:
        // [ base_age / 100, (base_age + direction) / 100 ] clipped to [0, 1]
        let baseAge: Float = 25.0
        let targetAge = min(max(baseAge + Float(settings.direction), 0.0), 100.0)
        let directionVector: [Float] = [baseAge / 100.0, targetAge / 100.0]

        let inputTensor = cropBuffer.toFloatTensorNCHW(mean: metadata.mean, std: metadata.std, isBGR: metadata.isBGR)
        let inputs: [String: TensorBuffer] = [
            "target": TensorBuffer(floatData: inputTensor, shape: [1, 3, cropH, cropW]),
            "direction": TensorBuffer(floatData: directionVector, shape: [1, 2])
        ]

        let outputs = try await ortBridge.run(modelPath: modelURL.path, inputs: inputs)
        guard let outputTensor = outputs.values.first?.floatData else {
            throw ORTBridgeError.inferenceFailed("Age modifier returned empty output tensor")
        }

        let modifiedCrop = ImageBuffer.fromFloatTensorNCHW(
            tensor: outputTensor,
            width: cropW,
            height: cropH,
            mean: metadata.mean,
            std: metadata.std,
            isBGR: metadata.isBGR
        )

        let boxMask = FaceMask.createBoxMask(width: cropW, height: cropH, blur: maskSettings.blur, padding: maskSettings.padding)
        let resultImage = targetImage.clone()
        resultImage.pasteBack(crop: modifiedCrop, mask: boxMask, matrix: affineMatrix)

        return resultImage
    }
}
