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

        // Invariant: settings.inputSize explicitly overrides filename dimension; validate bounded range 128...1024
        var resolvedDim: Int
        if let explicitSize = settings.inputSize {
            resolvedDim = explicitSize
        } else {
            let filename = modelURL.lastPathComponent
            if let regex = try? NSRegularExpression(pattern: "_(\\d+)(?:\\.[a-zA-Z0-9]+)?$", options: .caseInsensitive),
               let match = regex.firstMatch(in: filename, range: NSRange(location: 0, length: filename.utf16.count)),
               let range = Range(match.range(at: 1), in: filename),
               let parsedDim = Int(filename[range]) {
                resolvedDim = parsedDim
            } else {
                resolvedDim = 224
            }
        }

        guard (128...1024).contains(resolvedDim) else {
            throw ORTBridgeError.invalidInput("Deep swapper input dimension \(resolvedDim) is outside supported range 128...1024")
        }

        let cropW = resolvedDim
        let cropH = resolvedDim
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

        // Invariant: Native ORT API session.inputNames/outputNames exposes order; describeNames avoids guessed output names
        let (inputNames, outputNames) = try await ortBridge.describeNames(modelPath: modelURL.path)
        guard !inputNames.isEmpty, !outputNames.isEmpty else {
            throw ORTBridgeError.sessionCreationFailed("DFM model has empty input or output tensor names at path: \(modelURL.path)")
        }

        let faceInputName = inputNames.first(where: {
            let lower = $0.lowercased()
            return lower.contains("face") || lower.contains("frame") || lower.contains("input")
        }) ?? inputNames[0]

        var inputs: [String: TensorBuffer] = [
            faceInputName: TensorBuffer(floatData: nhwcBGR, shape: [1, cropH, cropW, 3])
        ]

        // Invariant: Morph input is optional; omit if absent from model input metadata
        if let morphInputName = inputNames.first(where: { $0.lowercased().contains("morph") }) {
            inputs[morphInputName] = TensorBuffer(floatData: [morphNorm], shape: [1])
        }

        let outputs = try await ortBridge.run(modelPath: modelURL.path, inputs: inputs)

        // Invariant: Output order resolved from actual model metadata, not invented names.
        // DFL output order: [0: target_mask, 1: crop_vision_frame, 2: source_mask] or [0: crop_vision_frame].
        let frameOutputName: String
        if let named = outputNames.first(where: {
            let lower = $0.lowercased()
            return lower.contains("crop_vision_frame") || lower.contains("vision_frame") || lower.contains("frame")
        }) {
            frameOutputName = named
        } else if outputNames.count >= 3 {
            frameOutputName = outputNames[1]
        } else {
            frameOutputName = outputNames[0]
        }

        guard let outputFrameTensor = outputs[frameOutputName]?.floatData,
              outputFrameTensor.count >= cropW * cropH * 3 else {
            throw ORTBridgeError.inferenceFailed("Deep swapper missing expected frame output '\(frameOutputName)' (crop_vision_frame) or invalid length (got \(outputs[frameOutputName]?.floatData?.count ?? 0), expected >= \(cropW * cropH * 3))")
        }

        let swappedCrop = ImageBuffer.fromFloatTensorNHWC(
            tensor: outputFrameTensor,
            width: cropW,
            height: cropH,
            isBGR: true
        )

        // Invariant: Mask processing with strict length validation for all expected masks
        var masks: [FaceMask] = []
        let targetMaskName: String?
        let sourceMaskName: String?
        if outputNames.count >= 3 {
            targetMaskName = outputNames.first(where: { $0.lowercased().contains("target") }) ?? outputNames[0]
            sourceMaskName = outputNames.first(where: { $0.lowercased().contains("source") }) ?? outputNames[2]
        } else {
            targetMaskName = outputNames.first(where: { $0.lowercased().contains("target") })
            sourceMaskName = outputNames.first(where: { $0.lowercased().contains("source") })
        }

        if let tName = targetMaskName,
           let sName = sourceMaskName,
           tName != frameOutputName,
           sName != frameOutputName,
           let targetMaskData = outputs[tName]?.floatData,
           let sourceMaskData = outputs[sName]?.floatData {
            let expectedMaskLength = cropW * cropH
            guard targetMaskData.count >= expectedMaskLength && sourceMaskData.count >= expectedMaskLength else {
                throw ORTBridgeError.invalidOutput("Deep swapper mask tensors length mismatch: target (\(targetMaskData.count)), source (\(sourceMaskData.count)), expected >= \(expectedMaskLength)")
            }
            var combinedDFLMask = FaceMask(width: cropW, height: cropH)
            for i in 0..<expectedMaskLength {
                combinedDFLMask.values[i] = min(targetMaskData[i], sourceMaskData[i])
            }
            // Feather with sigma 6.25 matching upstream prepare_crop_mask
            let blurredDFLMask = combinedDFLMask.gaussianBlurred(sigma: 6.25)
            masks.append(blurredDFLMask)
        }

        let userMask = try await ProcessorMasks.createCombinedMask(
            cropBuffer: cropBuffer,
            targetFace: targetFace,
            affineMatrix: affineMatrix,
            maskSettings: maskSettings,
            modelCache: ModelCache.shared,
            ortBridge: ortBridge
        )
        masks.append(userMask)

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
