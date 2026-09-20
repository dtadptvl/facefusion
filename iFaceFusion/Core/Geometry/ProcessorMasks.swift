import Foundation
import UIKit
import simd

// MARK: - Pixel Boost Interleaving

/// Implements exact sub-pixel interleaving and decimation matching upstream FaceFusion `pixel_boost.py`.
/// - `implode`: Decimates a high-resolution crop (e.g. 512x512 or 1024x1024) into `total^2` interleaved sub-frames of model size (e.g. 256x256).
/// - `explode`: Re-interleaves swapped sub-frames back into the full-resolution crop.
public enum PixelBoost {

    /// Parses the boost pixel dimension from setting string (e.g. "512x512" -> 512, "1024x1024" -> 1024).
    public static func parseBoostDimension(_ pixelBoost: String, fallback: Int) -> Int {
        let parts = pixelBoost.lowercased().split(separator: "x")
        if parts.count == 2, let dim = Int(parts[0]), dim > 0 {
            return dim
        }
        return fallback
    }

    /// Decimates `crop` of size (boostWidth, boostHeight) into `total * total` interleaved sub-frames of size (modelWidth, modelHeight).
    /// Matches upstream:
    /// `crop_vision_frame.reshape(model_size[0], total, model_size[1], total, 3).transpose(1, 3, 0, 2, 4).reshape(total**2, model_size[0], model_size[1], 3)`
    public static func implode(
        crop: ImageBuffer,
        total: Int,
        modelWidth: Int,
        modelHeight: Int
    ) -> [ImageBuffer] {
        guard total > 1 else { return [crop.clone()] }
        var subFrames = [ImageBuffer]()
        subFrames.reserveCapacity(total * total)

        for ySub in 0..<total {
            for xSub in 0..<total {
                var sub = ImageBuffer(width: modelWidth, height: modelHeight)
                for ym in 0..<modelHeight {
                    let origY = ym * total + ySub
                    let origRowOffset = origY * crop.width
                    let subRowOffset = ym * modelWidth
                    for xm in 0..<modelWidth {
                        let origX = xm * total + xSub
                        let origIdx = (origRowOffset + origX) * 4
                        let subIdx = (subRowOffset + xm) * 4

                        sub.data[subIdx + 0] = crop.data[origIdx + 0]
                        sub.data[subIdx + 1] = crop.data[origIdx + 1]
                        sub.data[subIdx + 2] = crop.data[origIdx + 2]
                        sub.data[subIdx + 3] = crop.data[origIdx + 3]
                    }
                }
                subFrames.append(sub)
            }
        }
        return subFrames
    }

    /// Reconstructs the full-resolution crop of size (boostWidth, boostHeight) from `total * total` sub-frames of size (modelWidth, modelHeight).
    /// Matches upstream:
    /// `stack(temp_vision_frames).reshape(total, total, model_size[0], model_size[1], 3).transpose(2, 0, 3, 1, 4).reshape(boost_size[0], boost_size[1], 3)`
    public static func explode(
        subFrames: [ImageBuffer],
        total: Int,
        boostWidth: Int,
        boostHeight: Int,
        modelWidth: Int,
        modelHeight: Int
    ) -> ImageBuffer {
        guard total > 1, !subFrames.isEmpty else {
            return subFrames.first?.clone() ?? ImageBuffer(width: boostWidth, height: boostHeight)
        }
        precondition(subFrames.count == total * total, "Expected \(total * total) sub-frames for boost explode, got \(subFrames.count)")

        var result = ImageBuffer(width: boostWidth, height: boostHeight)
        var subIndex = 0

        for ySub in 0..<total {
            for xSub in 0..<total {
                let sub = subFrames[subIndex]
                subIndex += 1

                for ym in 0..<modelHeight {
                    let origY = ym * total + ySub
                    let origRowOffset = origY * boostWidth
                    let subRowOffset = ym * modelWidth
                    for xm in 0..<modelWidth {
                        let origX = xm * total + xSub
                        let origIdx = (origRowOffset + origX) * 4
                        let subIdx = (subRowOffset + xm) * 4

                        result.data[origIdx + 0] = sub.data[subIdx + 0]
                        result.data[origIdx + 1] = sub.data[subIdx + 1]
                        result.data[origIdx + 2] = sub.data[subIdx + 2]
                        result.data[origIdx + 3] = sub.data[subIdx + 3]
                    }
                }
            }
        }
        return result
    }
}

// MARK: - Reusable Mask Execution

/// Reusable neural, landmark-driven area, and box mask execution pipeline matching FaceFusion `face_masker.py`.
public enum ProcessorMasks {

    /// Standard FaceFusion mask feathering:
    /// `(gaussian_blur(mask.clip(0, 1), sigma=5).clip(0.5, 1) - 0.5) * 2`
    public static func featherMask(_ mask: FaceMask, sigma: Float = 5.0) -> FaceMask {
        let blurred = mask.gaussianBlurred(sigma: sigma)
        var feathered = FaceMask(width: mask.width, height: mask.height, initialValue: 0.0)
        let count = mask.width * mask.height
        for i in 0..<count {
            let v = min(max(blurred.values[i], 0.5), 1.0)
            feathered.values[i] = (v - 0.5) * 2.0
        }
        return feathered
    }

    /// Creates a box mask matching upstream padding and blur specification.
    public static func createBoxMask(
        width: Int,
        height: Int,
        blur: Float,
        padding: FaceMaskPadding
    ) -> FaceMask {
        FaceMask.createBoxMask(width: width, height: height, blur: blur, padding: padding)
    }

    /// Creates an area mask from 68 landmarks transformed into crop space.
    public static func createAreaMask(
        width: Int,
        height: Int,
        landmarks68InCrop: [SIMD2<Float>],
        areas: Set<FaceMaskArea>
    ) -> FaceMask {
        FaceMask.createAreaMask(width: width, height: height, landmarks68InCrop: landmarks68InCrop, areas: areas)
    }

    /// Creates an occlusion mask via neural inference using `ModelCatalog.xseg1` (256x256, NHWC RGB [0, 1]).
    public static func createOcclusionMask(
        cropBuffer: ImageBuffer,
        modelCache: ModelCache,
        ortBridge: ORTBridge
    ) async throws -> FaceMask {
        let metadata = ModelCatalog.xseg1
        let modelURL = try await modelCache.ensureModelDownloaded(metadata)

        let modelW = metadata.inputWidth
        let modelH = metadata.inputHeight

        // Prepare frame: resize to 256x256
        let scaleX = Float(modelW) / Float(cropBuffer.width)
        let scaleY = Float(modelH) / Float(cropBuffer.height)
        let prepMatrix = AffineMatrix2x3(m00: scaleX, m01: 0, m02: 0, m10: 0, m11: scaleY, m12: 0)
        let prepCrop = cropBuffer.warpAffine(matrix: prepMatrix, cropWidth: modelW, cropHeight: modelH)

        // XSeg expects NHWC layout: [1, 256, 256, 3] RGB in [0, 1]
        let nhwcRGB = prepCrop.toFloatTensorNHWC(isBGR: false)
        let inputs: [String: TensorBuffer] = [
            "input": TensorBuffer(floatData: nhwcRGB, shape: [1, modelH, modelW, 3])
        ]

        let outputs = try await ortBridge.run(modelPath: modelURL.path, inputs: inputs)
        guard let rawFloats = (outputs["output"] ?? outputs.values.first)?.floatData,
              rawFloats.count >= modelW * modelH else {
            throw ORTBridgeError.inferenceFailed("XSeg face occluder returned invalid output dimensions: expected >= \(modelW * modelH)")
        }

        // Clip raw mask values to [0, 1]
        var mask256 = [Float](repeating: 0, count: modelW * modelH)
        for i in 0..<(modelW * modelH) {
            mask256[i] = min(max(rawFloats[i], 0.0), 1.0)
        }

        let baseMask = FaceMask(width: modelW, height: modelH, values: mask256)
        let resized = baseMask.resized(toWidth: cropBuffer.width, toHeight: cropBuffer.height)
        return featherMask(resized, sigma: 5.0)
    }

    /// Creates a region parsing mask via neural inference using `ModelCatalog.bisenetResnet18` (512x512, NCHW RGB ImageNet normalized).
    public static func createRegionMask(
        cropBuffer: ImageBuffer,
        regions: Set<FaceMaskRegion>,
        modelCache: ModelCache,
        ortBridge: ORTBridge
    ) async throws -> FaceMask {
        guard !regions.isEmpty else {
            return FaceMask(width: cropBuffer.width, height: cropBuffer.height, initialValue: 0.0)
        }

        let metadata = ModelCatalog.bisenetResnet18
        let modelURL = try await modelCache.ensureModelDownloaded(metadata)

        let modelW = metadata.inputWidth
        let modelH = metadata.inputHeight

        // Prepare frame: resize to 512x512
        let scaleX = Float(modelW) / Float(cropBuffer.width)
        let scaleY = Float(modelH) / Float(cropBuffer.height)
        let prepMatrix = AffineMatrix2x3(m00: scaleX, m01: 0, m02: 0, m10: 0, m11: scaleY, m12: 0)
        let prepCrop = cropBuffer.warpAffine(matrix: prepMatrix, cropWidth: modelW, cropHeight: modelH)

        // BiSeNet ResNet-18 expects NCHW layout: [1, 3, 512, 512] RGB with ImageNet normalization
        let nchwRGB = prepCrop.toFloatTensorNCHW(mean: metadata.mean, std: metadata.std, isBGR: false)
        let inputs: [String: TensorBuffer] = [
            "input": TensorBuffer(floatData: nchwRGB, shape: [1, 3, modelH, modelW])
        ]

        let outputs = try await ortBridge.run(modelPath: modelURL.path, inputs: inputs)
        let planeSize = modelW * modelH
        guard let rawFloats = (outputs["output"] ?? outputs.values.first)?.floatData,
              rawFloats.count >= 19 * planeSize else {
            throw ORTBridgeError.inferenceFailed("BiSeNet face parser returned invalid output: expected at least \(19 * planeSize) elements")
        }

        let selectedClassIds = Set(regions.map { $0.classId })
        var parsedValues = [Float](repeating: 0, count: planeSize)

        // Argmax across 19 classes for each pixel
        for i in 0..<planeSize {
            var bestClass = 0
            var maxLogit = -Float.greatestFiniteMagnitude
            for c in 0..<19 {
                let logit = rawFloats[c * planeSize + i]
                if logit > maxLogit {
                    maxLogit = logit
                    bestClass = c
                }
            }
            parsedValues[i] = selectedClassIds.contains(bestClass) ? 1.0 : 0.0
        }

        let baseMask = FaceMask(width: modelW, height: modelH, values: parsedValues)
        let resized = baseMask.resized(toWidth: cropBuffer.width, toHeight: cropBuffer.height)
        return featherMask(resized, sigma: 5.0)
    }

    /// Central multi-mask execution: generates all active mask types and combines them via element-wise minimum.
    public static func createCombinedMask(
        cropBuffer: ImageBuffer,
        targetFace: FaceTarget,
        affineMatrix: AffineMatrix2x3,
        maskSettings: FaceMaskSettings,
        modelCache: ModelCache,
        ortBridge: ORTBridge
    ) async throws -> FaceMask {
        let width = cropBuffer.width
        let height = cropBuffer.height
        var masks: [FaceMask] = []

        // 1. Box mask
        if maskSettings.types.contains(.box) {
            let boxMask = createBoxMask(width: width, height: height, blur: maskSettings.blur, padding: maskSettings.padding)
            masks.append(boxMask)
        }

        // 2. Occlusion mask (neural)
        if maskSettings.types.contains(.occlusion) {
            let occlusionMask = try await createOcclusionMask(cropBuffer: cropBuffer, modelCache: modelCache, ortBridge: ortBridge)
            masks.append(occlusionMask)
        }

        // 3. Region mask (neural)
        if maskSettings.types.contains(.region) {
            let regionMask = try await createRegionMask(cropBuffer: cropBuffer, regions: maskSettings.regions, modelCache: modelCache, ortBridge: ortBridge)
            masks.append(regionMask)
        }

        // 4. Area mask (geometry)
        if maskSettings.types.contains(.area) {
            let warped68 = targetFace.landmark68.points.map { affineMatrix.transformPoint($0) }
            let areaMask = createAreaMask(width: width, height: height, landmarks68InCrop: warped68, areas: maskSettings.areas)
            masks.append(areaMask)
        }

        guard !masks.isEmpty else {
            return FaceMask(width: width, height: height, initialValue: 1.0)
        }

        return FaceMask.combineMinimum(masks)
    }
}
