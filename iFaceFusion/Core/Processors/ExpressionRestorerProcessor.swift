import Foundation
import UIKit
import simd

/// Implements FaceFusion expression_restorer processor restoring natural expressions using LivePortrait models.
public final class ExpressionRestorerProcessor: Sendable {
    public init() {}

    public func process(
        referenceImage: ImageBuffer,
        targetImage: ImageBuffer,
        targetFace: FaceTarget,
        settings: ExpressionRestorerSettings,
        maskSettings: FaceMaskSettings,
        modelCache: ModelCache,
        ortBridge: ORTBridge
    ) async throws -> ImageBuffer {
        let featModel = ModelCatalog.livePortraitFeatureExtractor
        let motionModel = ModelCatalog.livePortraitMotionExtractor
        let genModel = ModelCatalog.livePortraitGenerator

        let featURL = try await modelCache.ensureModelDownloaded(featModel)
        let motionURL = try await modelCache.ensureModelDownloaded(motionModel)
        let genURL = try await modelCache.ensureModelDownloaded(genModel)

        let cropSize = 512
        let prepareSize = 256
        let template = WarpTemplate.arcface128
        let targetPoints = template.targetPoints(width: Float(cropSize), height: Float(cropSize))
        let affineMatrix = ImageGeometry.estimateSimilarityMatrix(src: targetFace.landmark5.points, dst: targetPoints)

        let targetCrop = referenceImage.warpAffine(matrix: affineMatrix, cropWidth: cropSize, cropHeight: cropSize)
        let tempCrop = targetImage.warpAffine(matrix: affineMatrix, cropWidth: cropSize, cropHeight: cropSize)

        // Downscale to 256x256 for feature and motion extraction
        let downscaleMatrix = AffineMatrix2x3(m00: 0.5, m01: 0, m02: 0, m10: 0, m11: 0.5, m12: 0)
        let prepTarget = targetCrop.warpAffine(matrix: downscaleMatrix, cropWidth: prepareSize, cropHeight: prepareSize)
        let prepTemp = tempCrop.warpAffine(matrix: downscaleMatrix, cropWidth: prepareSize, cropHeight: prepareSize)

        // 1. Feature extraction on current working temp crop
        let tempFeatTensor = prepTemp.toFloatTensorNCHW(mean: [0, 0, 0], std: [1, 1, 1], isBGR: false)
        let featOutputs = try await ortBridge.run(modelPath: featURL.path, inputs: [
            "input": TensorBuffer(floatData: tempFeatTensor, shape: [1, 3, prepareSize, prepareSize])
        ])
        guard let featVolume = featOutputs.values.first else {
            throw ORTBridgeError.inferenceFailed("LivePortrait feature extraction failed")
        }

        // 2. Motion extraction on target crop
        let targetMotionTensor = prepTarget.toFloatTensorNCHW(mean: [0, 0, 0], std: [1, 1, 1], isBGR: false)
        let targetMotionOut = try await ortBridge.run(modelPath: motionURL.path, inputs: [
            "input": TensorBuffer(floatData: targetMotionTensor, shape: [1, 3, prepareSize, prepareSize])
        ])

        // 3. Motion extraction on temp crop
        let tempMotionOut = try await ortBridge.run(modelPath: motionURL.path, inputs: [
            "input": TensorBuffer(floatData: tempFeatTensor, shape: [1, 3, prepareSize, prepareSize])
        ])

        // Extract pitch, yaw, roll, scale, translation, expression, motion_points
        let pitch = tempMotionOut["pitch"]?.floatData?.first ?? 0.0
        let yaw = tempMotionOut["yaw"]?.floatData?.first ?? 0.0
        let roll = tempMotionOut["roll"]?.floatData?.first ?? 0.0
        let scale = tempMotionOut["scale"]?.floatData?.first ?? 1.0
        let transData = tempMotionOut["translation"]?.floatData ?? [0, 0, 0]
        let translation = SIMD3<Float>(transData[0], transData[1], transData[2])

        let rotation = LivePortraitGeometry.createRotation(pitch: pitch, yaw: yaw, roll: roll)

        // Unpack (21, 3) expression and canonical motion points
        var tempExpr = [[Float]](repeating: [Float](repeating: 0, count: 3), count: 21)
        var targetExpr = [[Float]](repeating: [Float](repeating: 0, count: 3), count: 21)
        var motionPoints = [[Float]](repeating: [Float](repeating: 0, count: 3), count: 21)

        if let rawTemp = tempMotionOut["expression"]?.floatData {
            for i in 0..<21 { for j in 0..<3 { tempExpr[i][j] = rawTemp[i * 3 + j] } }
        }
        if let rawTarget = targetMotionOut["expression"]?.floatData {
            for i in 0..<21 { for j in 0..<3 { targetExpr[i][j] = rawTarget[i * 3 + j] } }
        }
        if let rawPts = tempMotionOut["motion_points"]?.floatData {
            for i in 0..<21 { for j in 0..<3 { motionPoints[i][j] = rawPts[i * 3 + j] } }
        }

        // Restrict expression areas matching upstream logic
        let upperIndices = [1, 2, 6, 10, 11, 12, 13, 15, 16]
        let lowerIndices = [3, 7, 14, 17, 18, 19, 20]
        if !settings.areas.contains(.upperFace) {
            for idx in upperIndices { targetExpr[idx] = tempExpr[idx] }
        }
        if !settings.areas.contains(.lowerFace) {
            for idx in lowerIndices { targetExpr[idx] = tempExpr[idx] }
        }
        for idx in [0, 4, 5, 8, 9] { targetExpr[idx] = tempExpr[idx] }

        // Blend expressions with factor
        let factor = LivePortraitGeometry.interp(x: Float(settings.factor), xp0: 0, xp1: 100, fp0: 0.0, fp1: 1.2)
        var blendedExpr = [[Float]](repeating: [Float](repeating: 0, count: 3), count: 21)
        for i in 0..<21 {
            for j in 0..<3 {
                blendedExpr[i][j] = targetExpr[i][j] * factor + tempExpr[i][j] * (1.0 - factor)
            }
        }
        LivePortraitGeometry.limitExpression(&blendedExpr)

        // Transform motion points
        let targetMotionPts = LivePortraitGeometry.transformMotionPoints(
            points: motionPoints,
            rotation: rotation,
            expression: blendedExpr,
            scale: scale,
            translation: translation
        )
        let tempMotionPts = LivePortraitGeometry.transformMotionPoints(
            points: motionPoints,
            rotation: rotation,
            expression: tempExpr,
            scale: scale,
            translation: translation
        )

        let flatTargetPts = targetMotionPts.flatMap { $0 }
        let flatTempPts = tempMotionPts.flatMap { $0 }

        // 4. Generator forward pass
        let genInputs: [String: TensorBuffer] = [
            "feature_volume": featVolume,
            "source": TensorBuffer(floatData: flatTargetPts, shape: [1, 21, 3]),
            "target": TensorBuffer(floatData: flatTempPts, shape: [1, 21, 3])
        ]
        let genOutputs = try await ortBridge.run(modelPath: genURL.path, inputs: genInputs)
        guard let genTensor = genOutputs.values.first?.floatData else {
            throw ORTBridgeError.inferenceFailed("LivePortrait generator returned empty tensor")
        }

        let restoredCrop = ImageBuffer.fromFloatTensorNCHW(
            tensor: genTensor,
            width: cropSize,
            height: cropSize,
            mean: [0, 0, 0],
            std: [1, 1, 1],
            isBGR: false
        )

        let boxMask = FaceMask.createBoxMask(width: cropSize, height: cropSize, blur: maskSettings.blur, padding: maskSettings.padding)
        let resultImage = targetImage.clone()
        resultImage.pasteBack(crop: restoredCrop, mask: boxMask, matrix: affineMatrix)

        return resultImage
    }
}
