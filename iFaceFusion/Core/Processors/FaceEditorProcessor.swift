import Foundation
import UIKit
import simd

/// Implements FaceFusion face_editor processor with complete LivePortrait geometric retargeting,
/// head pose rotation, expression controls, and landmark-driven stitching.
public final class FaceEditorProcessor: Sendable {
    public init() {}

    public func process(
        targetImage: ImageBuffer,
        targetFace: FaceTarget,
        settings: FaceEditorSettings,
        maskSettings: FaceMaskSettings,
        modelCache: ModelCache,
        ortBridge: ORTBridge
    ) async throws -> ImageBuffer {
        // Ensure all 6 LivePortrait sub-models are downloaded
        let featModel = ModelCatalog.livePortraitFeatureExtractor
        let motionModel = ModelCatalog.livePortraitMotionExtractor
        let eyeModel = ModelCatalog.livePortraitEyeRetargeter
        let lipModel = ModelCatalog.livePortraitLipRetargeter
        let stitchModel = ModelCatalog.livePortraitStitcher
        let genModel = ModelCatalog.livePortraitGenerator

        let featURL = try await modelCache.ensureModelDownloaded(featModel)
        let motionURL = try await modelCache.ensureModelDownloaded(motionModel)
        let eyeURL = try await modelCache.ensureModelDownloaded(eyeModel)
        let lipURL = try await modelCache.ensureModelDownloaded(lipModel)
        let stitchURL = try await modelCache.ensureModelDownloaded(stitchModel)
        let genURL = try await modelCache.ensureModelDownloaded(genModel)

        let cropSize = 512
        let prepareSize = 256
        let template = WarpTemplate.ffhq512

        // FaceFusion face_editor scales 5-landmarks by 1.5 relative to nose
        let scaled5 = targetFace.landmark5.scaled(by: 1.5)
        let targetPoints = template.targetPoints(width: Float(cropSize), height: Float(cropSize))
        let affineMatrix = ImageGeometry.estimateSimilarityMatrix(src: scaled5.points, dst: targetPoints)

        let cropBuffer = targetImage.warpAffine(matrix: affineMatrix, cropWidth: cropSize, cropHeight: cropSize)

        // Downscale to 256x256 for feature and motion extractors
        let downscaleMatrix = AffineMatrix2x3(m00: 0.5, m01: 0, m02: 0, m10: 0, m11: 0.5, m12: 0)
        let prepCrop = cropBuffer.warpAffine(matrix: downscaleMatrix, cropWidth: prepareSize, cropHeight: prepareSize)

        // 1. Feature volume extraction: RGB [0, 1] NCHW
        let prepTensor = prepCrop.toFloatTensorNCHW(mean: [0, 0, 0], std: [1, 1, 1], isBGR: false)
        let featOutputs = try await ortBridge.run(modelPath: featURL.path, inputs: [
            "input": TensorBuffer(floatData: prepTensor, shape: [1, 3, prepareSize, prepareSize])
        ])
        guard let featVolume = featOutputs.values.first else {
            throw ORTBridgeError.inferenceFailed("LivePortrait feature volume extraction failed")
        }

        // 2. Motion parameter extraction
        let motionOutputs = try await ortBridge.run(modelPath: motionURL.path, inputs: [
            "input": TensorBuffer(floatData: prepTensor, shape: [1, 3, prepareSize, prepareSize])
        ])

        let pitch = motionOutputs["pitch"]?.floatData?.first ?? 0.0
        let yaw = motionOutputs["yaw"]?.floatData?.first ?? 0.0
        let roll = motionOutputs["roll"]?.floatData?.first ?? 0.0
        let scale = motionOutputs["scale"]?.floatData?.first ?? 1.0
        let transData = motionOutputs["translation"]?.floatData ?? [0, 0, 0]
        let translation = SIMD3<Float>(transData[0], transData[1], transData[2])

        var baseExpression = [[Float]](repeating: [Float](repeating: 0, count: 3), count: 21)
        var canonicalPoints = [[Float]](repeating: [Float](repeating: 0, count: 3), count: 21)

        if let rawExpr = motionOutputs["expression"]?.floatData {
            for i in 0..<21 { for j in 0..<3 { baseExpression[i][j] = rawExpr[i * 3 + j] } }
        }
        if let rawPts = motionOutputs["motion_points"]?.floatData {
            for i in 0..<21 { for j in 0..<3 { canonicalPoints[i][j] = rawPts[i * 3 + j] } }
        }

        // Base target motion points without editing
        let baseRotation = LivePortraitGeometry.createRotation(pitch: pitch, yaw: yaw, roll: roll)
        let targetMotionPts = LivePortraitGeometry.transformMotionPoints(
            points: canonicalPoints,
            rotation: baseRotation,
            expression: baseExpression,
            scale: scale,
            translation: translation
        )

        // 3. Apply slider expressions
        var editedExpression = baseExpression
        LivePortraitGeometry.applyExpressionSliders(
            expression: &editedExpression,
            eyebrowDirection: settings.eyebrowDirection,
            eyeGazeHorizontal: settings.eyeGazeHorizontal,
            eyeGazeVertical: settings.eyeGazeVertical,
            mouthGrim: settings.mouthGrim,
            mouthPout: settings.mouthPout,
            mouthPurse: settings.mouthPurse,
            mouthSmile: settings.mouthSmile,
            mouthPosHorizontal: settings.mouthPositionHorizontal,
            mouthPosVertical: settings.mouthPositionVertical
        )

        // 4. Edited head pose rotation
        let editedRotation = LivePortraitGeometry.editHeadRotation(
            pitch: pitch,
            yaw: yaw,
            roll: roll,
            editPitchSlider: settings.headPitch,
            editYawSlider: settings.headYaw,
            editRollSlider: settings.headRoll
        )

        // Source motion points with rotation and edited expression
        var sourceMotionPts = LivePortraitGeometry.transformMotionPoints(
            points: canonicalPoints,
            rotation: editedRotation,
            expression: editedExpression,
            scale: scale,
            translation: translation
        )

        // 5. Eye retargeting using distance ratios
        if abs(settings.eyeOpenRatio) > 0.01 {
            let leftEyeRatio = targetFace.landmark68.distanceRatio(top: 37, bottom: 40, left: 39, right: 36)
            let rightEyeRatio = targetFace.landmark68.distanceRatio(top: 43, bottom: 46, left: 45, right: 42)
            let targetRatioValue: Float = settings.eyeOpenRatio < 0 ? 0.0 : 0.6

            var eyeInput = targetMotionPts.flatMap { $0 }
            eyeInput.append(contentsOf: [leftEyeRatio, rightEyeRatio, targetRatioValue])

            let eyeOutputs = try await ortBridge.run(modelPath: eyeURL.path, inputs: [
                "input": TensorBuffer(floatData: eyeInput, shape: [1, 66])
            ])

            if let eyeDelta = eyeOutputs.values.first?.floatData {
                let absRatio = abs(settings.eyeOpenRatio)
                for i in 0..<21 {
                    for j in 0..<3 {
                        sourceMotionPts[i][j] += eyeDelta[i * 3 + j] * absRatio
                    }
                }
            }
        }

        // 6. Lip retargeting using distance ratio
        if abs(settings.lipOpenRatio) > 0.01 {
            let lipRatio = targetFace.landmark68.distanceRatio(top: 62, bottom: 66, left: 54, right: 48)
            let targetLipValue: Float = settings.lipOpenRatio < 0 ? 0.0 : 1.0

            var lipInput = targetMotionPts.flatMap { $0 }
            lipInput.append(contentsOf: [lipRatio, targetLipValue])

            let lipOutputs = try await ortBridge.run(modelPath: lipURL.path, inputs: [
                "input": TensorBuffer(floatData: lipInput, shape: [1, 65])
            ])

            if let lipDelta = lipOutputs.values.first?.floatData {
                let absRatio = abs(settings.lipOpenRatio)
                for i in 0..<21 {
                    for j in 0..<3 {
                        sourceMotionPts[i][j] += lipDelta[i * 3 + j] * absRatio
                    }
                }
            }
        }

        // 7. Stitch motion points
        let flatSource = sourceMotionPts.flatMap { $0 }
        let flatTarget = targetMotionPts.flatMap { $0 }

        let stitchOutputs = try await ortBridge.run(modelPath: stitchURL.path, inputs: [
            "source": TensorBuffer(floatData: flatSource, shape: [1, 21, 3]),
            "target": TensorBuffer(floatData: flatTarget, shape: [1, 21, 3])
        ])

        let stitchedPoints = stitchOutputs.values.first?.floatData ?? flatSource

        // 8. Generator forward pass
        let genOutputs = try await ortBridge.run(modelPath: genURL.path, inputs: [
            "feature_volume": featVolume,
            "source": TensorBuffer(floatData: stitchedPoints, shape: [1, 21, 3]),
            "target": TensorBuffer(floatData: flatTarget, shape: [1, 21, 3])
        ])

        guard let genTensor = genOutputs.values.first?.floatData else {
            throw ORTBridgeError.inferenceFailed("LivePortrait generator produced empty tensor")
        }

        let editedCrop = ImageBuffer.fromFloatTensorNCHW(
            tensor: genTensor,
            width: cropSize,
            height: cropSize,
            mean: [0, 0, 0],
            std: [1, 1, 1],
            isBGR: false
        )

        let boxMask = FaceMask.createBoxMask(width: cropSize, height: cropSize, blur: maskSettings.blur, padding: maskSettings.padding)
        let resultImage = targetImage.clone()
        resultImage.pasteBack(crop: editedCrop, mask: boxMask, matrix: affineMatrix)

        return resultImage
    }
}
