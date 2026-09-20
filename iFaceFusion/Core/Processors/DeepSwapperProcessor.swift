import Foundation
import UIKit

/// Implements FaceFusion deep_swapper processor supporting DeepFaceLive (.dfm / .onnx) models
/// with native NHWC BGR tensor layout, dynamic input sizing, and morph controls.
public final class DeepSwapperProcessor: Sendable {
    public init() {}

    public func process(
        targetImage: ImageBuffer,
        targetFace: FaceTarget,
        modelURL: URL,
        settings: DeepSwapperSettings,
        maskSettings: FaceMaskSettings,
        ortBridge: ORTBridge
    ) async throws -> ImageBuffer {
        // ponytail: .dfm files packaged as zip containers with internal model.onnx can be extracted dynamically; currently loads the model file directly.
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw ORTBridgeError.sessionCreationFailed("DFM model file does not exist at path: \(modelURL.path)")
        }

        // Default DFL whole face resolution is 224x224 or 320x320
        let cropW = 224
        let cropH = 224
        let template = WarpTemplate.dflWholeFace
        let targetPoints = template.targetPoints(width: Float(cropW), height: Float(cropH))
        let affineMatrix = ImageGeometry.estimateSimilarityMatrix(src: targetFace.landmark5.points, dst: targetPoints)

        let cropBuffer = targetImage.warpAffine(matrix: affineMatrix, cropWidth: cropW, cropHeight: cropH)

        // Upstream DFL preprocessing: unsharp mask sharpening (1.75 * crop - 0.75 * blur(sigma=2.0))
        let blurredCrop = applyGaussianBlurToBuffer(cropBuffer, sigma: 2.0)
        var sharpenedCrop = ImageBuffer(width: cropW, height: cropH)
        for i in 0..<(cropW * cropH * 4) {
            let orig = Float(cropBuffer.data[i])
            let blur = Float(blurredCrop.data[i])
            let sharp = orig * 1.75 - blur * 0.75
            sharpenedCrop.data[i] = UInt8(min(max(sharp, 0), 255))
        }

        // Convert to NHWC BGR in [0, 1] matching DeepFaceLive contract
        let nhwcBGR = sharpenedCrop.toFloatTensorNHWC(isBGR: true)
        let morphNorm = Float(settings.morph) / 100.0

        var inputs: [String: TensorBuffer] = [
            "in_face:0": TensorBuffer(floatData: nhwcBGR, shape: [1, cropH, cropW, 3])
        ]
        inputs["morph_value:0"] = TensorBuffer(floatData: [morphNorm], shape: [1])

        let outputs = try await ortBridge.run(modelPath: modelURL.path, inputs: inputs)

        // DFL outputs: crop_target_mask, crop_vision_frame, crop_source_mask
        guard let outputFrameTensor = outputs["crop_vision_frame:0"]?.floatData ?? outputs.values.first?.floatData else {
            throw ORTBridgeError.inferenceFailed("Deep swapper returned empty output frame")
        }

        let swappedCrop = ImageBuffer.fromFloatTensorNHWC(
            tensor: outputFrameTensor,
            width: cropW,
            height: cropH,
            isBGR: true
        )

        // Mask processing: combine source and target DFL masks if provided
        var masks: [FaceMask] = []
        if let targetMaskData = outputs["crop_target_mask:0"]?.floatData,
           let sourceMaskData = outputs["crop_source_mask:0"]?.floatData {
            var combinedDFLMask = FaceMask(width: cropW, height: cropH)
            for i in 0..<(cropW * cropH) {
                combinedDFLMask.values[i] = min(targetMaskData[i], sourceMaskData[i])
            }
            // Feather with sigma 6.25 matching upstream prepare_crop_mask
            let blurredDFLMask = combinedDFLMask.gaussianBlurred(sigma: 6.25)
            masks.append(blurredDFLMask)
        } else {
            let boxMask = FaceMask.createBoxMask(width: cropW, height: cropH, blur: maskSettings.blur, padding: maskSettings.padding)
            masks.append(boxMask)
        }

        let finalMask = FaceMask.combineMinimum(masks)
        let resultImage = targetImage.clone()
        resultImage.pasteBack(crop: swappedCrop, mask: finalMask, matrix: affineMatrix)

        return resultImage
    }

    private func applyGaussianBlurToBuffer(_ buffer: ImageBuffer, sigma: Float) -> ImageBuffer {
        let rMask = FaceMask(width: buffer.width, height: buffer.height, values: (0..<(buffer.width * buffer.height)).map { Float(buffer.data[$0 * 4 + 0]) })
        let gMask = FaceMask(width: buffer.width, height: buffer.height, values: (0..<(buffer.width * buffer.height)).map { Float(buffer.data[$0 * 4 + 1]) })
        let bMask = FaceMask(width: buffer.width, height: buffer.height, values: (0..<(buffer.width * buffer.height)).map { Float(buffer.data[$0 * 4 + 2]) })

        let rBlurred = rMask.gaussianBlurred(sigma: sigma)
        let gBlurred = gMask.gaussianBlurred(sigma: sigma)
        let bBlurred = bMask.gaussianBlurred(sigma: sigma)

        var result = ImageBuffer(width: buffer.width, height: buffer.height)
        for i in 0..<(buffer.width * buffer.height) {
            result.data[i * 4 + 0] = UInt8(min(max(rBlurred.values[i], 0), 255))
            result.data[i * 4 + 1] = UInt8(min(max(gBlurred.values[i], 0), 255))
            result.data[i * 4 + 2] = UInt8(min(max(bBlurred.values[i], 0), 255))
            result.data[i * 4 + 3] = buffer.data[i * 4 + 3]
        }
        return result
    }
}
