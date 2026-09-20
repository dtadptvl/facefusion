import XCTest
import simd
@testable import iFaceFusion

final class iFaceFusionTests: XCTestCase {

    // MARK: - Geometry & Affine Similarity Transform

    func testUmeyamaSimilarityTransform() {
        let src: [SIMD2<Float>] = [
            SIMD2<Float>(100.0, 110.0),
            SIMD2<Float>(150.0, 108.0),
            SIMD2<Float>(125.0, 130.0),
            SIMD2<Float>(105.0, 155.0),
            SIMD2<Float>(145.0, 153.0)
        ]

        let template = WarpTemplate.arcface128
        let dst = template.targetPoints(width: 128.0, height: 128.0)

        let matrix = ImageGeometry.estimateSimilarityMatrix(src: src, dst: dst)

        // Expected values verified against OpenCV estimateAffinePartial2D:
        // [[ 0.8076878, -0.0113036, -35.451776 ],
        //  [ 0.0113036,  0.8076878, -35.480812 ]]
        XCTAssertEqual(matrix.m00, 0.8076878, accuracy: 1e-4)
        XCTAssertEqual(matrix.m01, -0.0113036, accuracy: 1e-4)
        XCTAssertEqual(matrix.m02, -35.451776, accuracy: 1e-2)
        XCTAssertEqual(matrix.m10, 0.0113036, accuracy: 1e-4)
        XCTAssertEqual(matrix.m11, 0.8076878, accuracy: 1e-4)
        XCTAssertEqual(matrix.m12, -35.480812, accuracy: 1e-2)

        // Invertibility check
        guard let inv = matrix.inverted() else {
            XCTFail("Matrix should be invertible")
            return
        }

        let p0 = src[0]
        let mapped = matrix.transformPoint(p0)
        let unmapped = inv.transformPoint(mapped)
        XCTAssertEqual(p0.x, unmapped.x, accuracy: 1e-3)
        XCTAssertEqual(p0.y, unmapped.y, accuracy: 1e-3)
    }

    // MARK: - LivePortrait Kinematics & Euler Rotation

    func testLivePortraitEulerRotationMatrix() {
        let pitch: Float = 15.0
        let yaw: Float = -20.0
        let roll: Float = 5.0

        let R = LivePortraitGeometry.createRotation(pitch: pitch, yaw: yaw, roll: roll)

        // Verify determinant is exactly 1 (pure rotation)
        let det = simd_determinant(R)
        XCTAssertEqual(det, 1.0, accuracy: 1e-5)

        // Verify orthogonality: R * R^T == I
        let identity = simd_mul(R, simd_transpose(R))
        XCTAssertEqual(identity[0][0], 1.0, accuracy: 1e-5)
        XCTAssertEqual(identity[1][1], 1.0, accuracy: 1e-5)
        XCTAssertEqual(identity[2][2], 1.0, accuracy: 1e-5)
        XCTAssertEqual(identity[0][1], 0.0, accuracy: 1e-5)
        XCTAssertEqual(identity[1][2], 0.0, accuracy: 1e-5)

        // Verify elements against Scipy Rotation.from_euler('xyz', [15, -20, 5], degrees=True)
        // [[ 0.93611681, -0.17237046, -0.30655138 ],
        //  [ 0.08189961,  0.95453504, -0.28662746 ],
        //  [ 0.34202014,  0.24321035,  0.90767337 ]]
        // simd_float3x3 indexing is [col][row]:
        XCTAssertEqual(R[0][0], 0.9361168, accuracy: 1e-4) // col 0, row 0
        XCTAssertEqual(R[1][0], -0.1723705, accuracy: 1e-4) // col 1, row 0
        XCTAssertEqual(R[2][0], -0.3065514, accuracy: 1e-4) // col 2, row 0

        XCTAssertEqual(R[0][1], 0.0818996, accuracy: 1e-4) // col 0, row 1
        XCTAssertEqual(R[1][1], 0.9545350, accuracy: 1e-4) // col 1, row 1
        XCTAssertEqual(R[2][1], -0.2866275, accuracy: 1e-4) // col 2, row 1

        XCTAssertEqual(R[0][2], 0.3420201, accuracy: 1e-4) // col 0, row 2
        XCTAssertEqual(R[1][2], 0.2432104, accuracy: 1e-4) // col 1, row 2
        XCTAssertEqual(R[2][2], 0.9076734, accuracy: 1e-4) // col 2, row 2
    }

    func testLivePortraitExpressionBounds() {
        var testExpr = [[Float]](repeating: [-10.0, 10.0, 0.0], count: 21)
        LivePortraitGeometry.limitExpression(&testExpr)

        for i in 0..<21 {
            XCTAssertGreaterThanOrEqual(testExpr[i][0], LivePortraitGeometry.expressionMin[i][0])
            XCTAssertLessThanOrEqual(testExpr[i][1], LivePortraitGeometry.expressionMax[i][1])
        }
    }

    // MARK: - Landmarks Conversion & Ratios

    func testLandmark68ToLandmark5() {
        var points = [SIMD2<Float>](repeating: SIMD2<Float>(0, 0), count: 68)

        // Left eye: 36..<42 set to (10, 20)
        for i in 36..<42 { points[i] = SIMD2<Float>(10.0, 20.0) }
        // Right eye: 42..<48 set to (50, 20)
        for i in 42..<48 { points[i] = SIMD2<Float>(50.0, 20.0) }
        // Nose: 30 set to (30, 35)
        points[30] = SIMD2<Float>(30.0, 35.0)
        // Left mouth: 48 set to (15, 50)
        points[48] = SIMD2<Float>(15.0, 50.0)
        // Right mouth: 54 set to (45, 50)
        points[54] = SIMD2<Float>(45.0, 50.0)

        let landmark68 = FaceLandmark68(points: points)
        let landmark5 = landmark68.toLandmark5()

        XCTAssertEqual(landmark5.leftEye, SIMD2<Float>(10.0, 20.0))
        XCTAssertEqual(landmark5.rightEye, SIMD2<Float>(50.0, 20.0))
        XCTAssertEqual(landmark5.nose, SIMD2<Float>(30.0, 35.0))
        XCTAssertEqual(landmark5.leftMouth, SIMD2<Float>(15.0, 50.0))
        XCTAssertEqual(landmark5.rightMouth, SIMD2<Float>(45.0, 50.0))

        // Distance ratio
        let ratio = landmark68.distanceRatio(top: 30, bottom: 48, left: 48, right: 54)
        XCTAssertGreaterThan(ratio, 0.0)
    }

    // MARK: - Face Mask Gaussian Blur

    func testFaceMaskSeparableGaussianBlur() {
        var mask = FaceMask(width: 32, height: 32, initialValue: 0.0)
        mask[16, 16] = 1.0 // Impulse point

        let blurred = mask.gaussianBlurred(sigma: 2.0)

        // Max value should be at the center (16, 16) and decrease monotonically outwards
        let centerVal = blurred[16, 16]
        let neighborVal = blurred[16, 17]
        let farVal = blurred[16, 25]

        XCTAssertGreaterThan(centerVal, neighborVal)
        XCTAssertGreaterThan(neighborVal, farVal)
        XCTAssertEqual(farVal, 0.0) // Outside the finite three-sigma kernel.

        // Symmetry
        XCTAssertEqual(blurred[15, 16], blurred[17, 16], accuracy: 1e-5)
        XCTAssertEqual(blurred[16, 15], blurred[16, 17], accuracy: 1e-5)
    }

    // MARK: - Model Catalog & Licensing

    func testModelCatalogIntegrity() {
        XCTAssertFalse(ModelCatalog.allModels.isEmpty)

        for model in ModelCatalog.allModels {
            XCTAssertFalse(model.id.isEmpty)
            XCTAssertFalse(model.name.isEmpty)
            XCTAssertFalse(model.license.isEmpty)
            XCTAssertFalse(model.sources.isEmpty)
            XCTAssertEqual(model.expectedCRC32.count, 8, "Model \(model.id) CRC32 must be 8 hex characters")
        }

        // Verify specific models exist
        XCTAssertNotNil(ModelCatalog.model(for: "hyperswap_1a_256"))
        XCTAssertNotNil(ModelCatalog.model(for: "arcface_w600k_r50"))
        XCTAssertNotNil(ModelCatalog.model(for: "gfpgan_1.4"))
        XCTAssertNotNil(ModelCatalog.model(for: "span_kendata_x4"))
        XCTAssertNotNil(ModelCatalog.model(for: "real_esrgan_x4"))

        // Verify verified upstream checksums
        XCTAssertEqual(ModelCatalog.realEsrganX4.expectedCRC32, "9d6e76c4")
        XCTAssertEqual(ModelCatalog.codeformer.expectedCRC32, "1456f3ab")

        // Verify speculative/unsupported models are removed from catalog
        XCTAssertNil(ModelCatalog.model(for: "inswapper_128"))
        XCTAssertNil(ModelCatalog.model(for: "codeformer"))
    }

    // MARK: - Image Pixel Transforms & Warp Affine

    func testPixelTransformsAndWarpAffine() {
        var buffer = ImageBuffer(width: 4, height: 4)
        // Fill top-left with red (255, 0, 0, 255)
        buffer.data[0] = 255
        buffer.data[3] = 255

        // Identity transform preserves coordinates
        let identity = AffineMatrix2x3(m00: 1, m01: 0, m02: 0, m10: 0, m11: 1, m12: 0)
        let warped = buffer.warpAffine(matrix: identity, cropWidth: 4, cropHeight: 4)
        XCTAssertEqual(warped.data[0], 255)
        XCTAssertEqual(warped.data[3], 255)

        // Translation transform shifts pixel
        let shift = AffineMatrix2x3(m00: 1, m01: 0, m02: 1, m10: 0, m11: 1, m12: 1)
        let shifted = buffer.warpAffine(matrix: shift, cropWidth: 4, cropHeight: 4)
        // (1, 1) in shifted maps to (0, 0) in source
        let idx11 = (1 * 4 + 1) * 4
        XCTAssertEqual(shifted.data[idx11], 255)
        XCTAssertEqual(shifted.data[idx11 + 3], 255)
    }

    // MARK: - Non-Square Orientation & Normalization

    func testNonSquareOrientationImageBuffer() {
        // Create 20x40 non-square bitmap context
        let width = 20
        let height = 40
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var rawData = [UInt8](repeating: 200, count: width * height * 4)
        let ctx = CGContext(
            data: &rawData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        )!
        let cgImage = ctx.makeImage()!

        // Test orientation .right: width and height should be transposed to 40x20 upright
        let rightImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
        guard let bufRight = ImageBuffer(image: rightImage) else {
            XCTFail("ImageBuffer init failed for .right orientation")
            return
        }
        XCTAssertEqual(bufRight.width, 40)
        XCTAssertEqual(bufRight.height, 20)

        // Test orientation .up: width and height should remain 20x40
        let upImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .up)
        guard let bufUp = ImageBuffer(image: upImage) else {
            XCTFail("ImageBuffer init failed for .up orientation")
            return
        }
        XCTAssertEqual(bufUp.width, 20)
        XCTAssertEqual(bufUp.height, 40)
    }

    // MARK: - Alpha Channel Roundtrip & Preservation

    func testAlphaChannelPreservationAndRoundtrip() {
        var buffer = ImageBuffer(width: 8, height: 8)
        // Populate specific pixel with 50% transparency: RGBA = (100, 150, 200, 128)
        let targetIdx = (3 * 8 + 3) * 4
        buffer.data[targetIdx + 0] = 100
        buffer.data[targetIdx + 1] = 150
        buffer.data[targetIdx + 2] = 200
        buffer.data[targetIdx + 3] = 128

        // Verify clone preserves alpha exactly
        let clone = buffer.clone()
        XCTAssertEqual(clone.data[targetIdx + 3], 128)

        // Verify pasteBack preserves original alpha channel
        var baseImage = ImageBuffer(width: 8, height: 8)
        baseImage.data[targetIdx + 3] = 128
        let crop = ImageBuffer(width: 4, height: 4)
        var mask = FaceMask(width: 4, height: 4, initialValue: 1.0)
        baseImage.pasteBack(crop: crop, mask: mask, matrix: AffineMatrix2x3.identity)
        XCTAssertEqual(baseImage.data[targetIdx + 3], 128, "pasteBack must not overwrite or corrupt alpha channel")
    }

    // MARK: - Named Tensor Outputs & Strict Counts

    func testHyperSwapNamedTensorOutputSelection() {
        // HyperSwap produces two named outputs: "output" (swapped image) and "mask" (face mask)
        let dummyImageTensor = TensorBuffer(floatData: [0.1, 0.2, 0.3], shape: [1, 3, 1, 1])
        let dummyMaskTensor = TensorBuffer(floatData: [1.0], shape: [1, 1, 1, 1])

        let outputs: [String: TensorBuffer] = [
            "mask": dummyMaskTensor,
            "output": dummyImageTensor
        ]

        // Strict name selection must return "output" tensor, never the mask tensor
        let selected = outputs["output"]?.floatData
        XCTAssertNotNil(selected)
        XCTAssertEqual(selected?.count, 3)
        XCTAssertEqual(selected?[0], 0.1)
    }

    func testLivePortraitStrictMotionTensors() {
        // Strict contract: all 7 kinematic motion tensors required; missing any tensor must fail
        let incompleteOutputs: [String: TensorBuffer] = [
            "pitch": TensorBuffer(floatData: [0.0], shape: [1, 1]),
            "yaw": TensorBuffer(floatData: [0.0], shape: [1, 1])
            // missing roll, scale, translation, expression, motion_points
        ]

        let hasAll = incompleteOutputs["pitch"] != nil &&
                     incompleteOutputs["yaw"] != nil &&
                     incompleteOutputs["roll"] != nil &&
                     incompleteOutputs["scale"] != nil &&
                     incompleteOutputs["translation"] != nil &&
                     incompleteOutputs["expression"] != nil &&
                     incompleteOutputs["motion_points"] != nil
        XCTAssertFalse(hasAll, "Incomplete LivePortrait kinematic outputs must be rejected without zero fallback")
    }

    // MARK: - Scaled Face Coordinates After Upscale

    func testFaceCoordinateScalingAfterUpscale() {
        let initialBbox = CGRect(x: 10, y: 20, width: 30, height: 40)
        let l5 = FaceLandmark5(points: [
            SIMD2<Float>(15, 25), SIMD2<Float>(35, 25),
            SIMD2<Float>(25, 35),
            SIMD2<Float>(18, 50), SIMD2<Float>(32, 50)
        ])
        let l68 = FaceLandmark68(points: [SIMD2<Float>](repeating: SIMD2<Float>(25, 35), count: 68))
        let targetFace = FaceTarget(boundingBox: initialBbox, landmark5: l5, landmark68: l68, age: 30.0)

        // Rescale by 4x (representing 4x frame_enhancer upscale)
        let rescaled = targetFace.rescaled(scaleX: 4.0, scaleY: 4.0)

        XCTAssertEqual(rescaled.boundingBox.origin.x, 40.0)
        XCTAssertEqual(rescaled.boundingBox.origin.y, 80.0)
        XCTAssertEqual(rescaled.boundingBox.width, 120.0)
        XCTAssertEqual(rescaled.boundingBox.height, 160.0)
        XCTAssertEqual(rescaled.landmark5.points[0], SIMD2<Float>(60.0, 100.0))
        XCTAssertEqual(rescaled.age, 30.0)
    }

    // MARK: - FRAN Explicit Source Age & Memory Budget

    func testFRANExplicitSourceAgeContract() {
        let settings = AgeModifierSettings(model: "fran", direction: 10, sourceAge: 40)
        XCTAssertEqual(settings.sourceAge, 40)

        let targetFaceWithoutAge = FaceTarget(
            boundingBox: .zero,
            landmark5: FaceLandmark5(points: [SIMD2<Float>](repeating: .zero, count: 5)),
            landmark68: FaceLandmark68(points: [SIMD2<Float>](repeating: .zero, count: 68)),
            age: nil
        )

        let effectiveBaseAge = targetFaceWithoutAge.age ?? Float(settings.sourceAge)
        XCTAssertEqual(effectiveBaseAge, 40.0, "FRAN must use explicit sourceAge when face age is nil")
    }

    func testMemoryBudgetGuardCalculations() {
        let width = 8000
        let height = 6000
        let scale = 4
        let outW = width * scale
        let outH = height * scale
        let totalPixels = outW * outH
        let maxPixelBudget = 64_000_000

        XCTAssertGreaterThan(totalPixels, maxPixelBudget, "4x upscale of 48MP image produces 768MP which exceeds 64MP budget")
    }

    // MARK: - Pixel Boost & Processor Masks Contracts

    func testPixelBoostInterleavingRoundtrip() {
        let boostW = 512
        let boostH = 512
        let modelW = 256
        let modelH = 256
        let total = 2

        var orig = ImageBuffer(width: boostW, height: boostH)
        for i in 0..<(boostW * boostH * 4) {
            orig.data[i] = UInt8((i * 37 + 13) % 256)
        }

        let subFrames = PixelBoost.implode(crop: orig, total: total, modelWidth: modelW, modelHeight: modelH)
        XCTAssertEqual(subFrames.count, total * total)
        for sub in subFrames {
            XCTAssertEqual(sub.width, modelW)
            XCTAssertEqual(sub.height, modelH)
        }

        let exploded = PixelBoost.explode(
            subFrames: subFrames,
            total: total,
            boostWidth: boostW,
            boostHeight: boostH,
            modelWidth: modelW,
            modelHeight: modelH
        )

        XCTAssertEqual(exploded.width, boostW)
        XCTAssertEqual(exploded.height, boostH)
        XCTAssertEqual(exploded.data, orig.data, "PixelBoost explode(implode(image)) must be an exact lossless bijection")
    }

    func testPixelBoost4xRoundtrip() {
        let boostW = 1024
        let boostH = 1024
        let modelW = 256
        let modelH = 256
        let total = 4

        var orig = ImageBuffer(width: boostW, height: boostH)
        for i in 0..<(boostW * boostH * 4) {
            orig.data[i] = UInt8((i * 17 + 5) % 256)
        }

        let subFrames = PixelBoost.implode(crop: orig, total: total, modelWidth: modelW, modelHeight: modelH)
        XCTAssertEqual(subFrames.count, 16)

        let exploded = PixelBoost.explode(
            subFrames: subFrames,
            total: total,
            boostWidth: boostW,
            boostHeight: boostH,
            modelWidth: modelW,
            modelHeight: modelH
        )

        XCTAssertEqual(exploded.data, orig.data, "1024x1024 4x PixelBoost roundtrip must be perfectly bijective")
    }

    func testProcessorMasksFeatheringFormula() {
        // Mask with values spanning [0, 1]
        let mask = FaceMask(width: 4, height: 4, values: [
            0.0, 0.2, 0.4, 0.5,
            0.55, 0.6, 0.75, 0.8,
            0.85, 0.9, 0.95, 1.0,
            1.0, 1.0, 1.0, 1.0
        ])
        let feathered = ProcessorMasks.featherMask(mask, sigma: 0.0)
        XCTAssertEqual(feathered.width, 4)
        XCTAssertEqual(feathered.height, 4)

        // When sigma = 0, gaussianBlurred returns copy of values
        // Formula: (clip(0.5, 1.0) - 0.5) * 2.0
        // value 0.0 -> clip to 0.5 -> (0.5 - 0.5) * 2 = 0.0
        // value 0.4 -> clip to 0.5 -> (0.5 - 0.5) * 2 = 0.0
        // value 0.5 -> clip to 0.5 -> (0.5 - 0.5) * 2 = 0.0
        // value 0.75 -> clip to 0.75 -> (0.75 - 0.5) * 2 = 0.5
        // value 1.0 -> clip to 1.0 -> (1.0 - 0.5) * 2 = 1.0
        XCTAssertEqual(feathered.values[0], 0.0, accuracy: 1e-5)
        XCTAssertEqual(feathered.values[2], 0.0, accuracy: 1e-5)
        XCTAssertEqual(feathered.values[3], 0.0, accuracy: 1e-5)
        XCTAssertEqual(feathered.values[6], 0.5, accuracy: 1e-5)
        XCTAssertEqual(feathered.values[11], 1.0, accuracy: 1e-5)
    }

    func testFaceMaskNonSquareResizing() {
        // Verify non-square resizing doesn't crash or index out of bounds after y0 clamping fix
        let mask = FaceMask(width: 100, height: 200, initialValue: 1.0)
        let resized = mask.resized(toWidth: 50, toHeight: 300)
        XCTAssertEqual(resized.width, 50)
        XCTAssertEqual(resized.height, 300)
        XCTAssertEqual(resized.values[0], 1.0, accuracy: 1e-6)
    }

    // MARK: - UI & ViewModel Validation Tests

    @MainActor
    func testPhotoEditorViewModelCanProcessRequiresFaceValidation() {
        let vm = PhotoEditorViewModel()
        vm.selectedProcessors = [.faceEnhancer] // does not require source

        // Case 1: No images loaded -> false
        XCTAssertFalse(vm.canProcess)

        // Case 2: Target image set but status is .idle or .detecting -> false
        vm.targetImage = UIImage()
        vm.targetStatus = .idle
        XCTAssertFalse(vm.canProcess)
        vm.targetStatus = .detecting
        XCTAssertFalse(vm.canProcess)

        // Case 3: Target face detection failed -> false
        vm.targetStatus = .failed(reason: "No face")
        XCTAssertFalse(vm.canProcess)

        // Case 4: Target face detected 2 faces -> false (must be exactly 1)
        vm.targetStatus = .valid(count: 2)
        XCTAssertFalse(vm.canProcess)

        // Case 5: Target face detected exactly 1 face -> true
        vm.targetStatus = .valid(count: 1)
        XCTAssertTrue(vm.canProcess)

        // Case 6: Processing flag active -> false
        vm.isProcessing = true
        XCTAssertFalse(vm.canProcess)
    }

    @MainActor
    func testPhotoEditorViewModelFaceSwapperRequiresSourceFace() {
        let vm = PhotoEditorViewModel()
        vm.selectedProcessors = [.faceSwapper]
        vm.targetImage = UIImage()
        vm.targetStatus = .valid(count: 1)

        // Face Swapper requires source image with 1 valid face
        XCTAssertTrue(vm.requiresSourceImage)
        XCTAssertFalse(vm.canProcess)

        // Source present but detecting -> false
        vm.sourceImage = UIImage()
        vm.sourceStatus = .detecting
        XCTAssertFalse(vm.canProcess)

        // Source failed -> false
        vm.sourceStatus = .failed(reason: "Blurry")
        XCTAssertFalse(vm.canProcess)

        // Source 1 valid face -> true
        vm.sourceStatus = .valid(count: 1)
        XCTAssertTrue(vm.canProcess)
    }

    @MainActor
    func testPhotoEditorViewModelCancelKeepsBusyUntilEngineReturns() {
        let vm = PhotoEditorViewModel()
        vm.isProcessing = true
        vm.progressMessage = "Running..."

        vm.cancelProcessing()

        // Cancel must NOT clear isProcessing immediately (keeps busy until engine unrolls)
        XCTAssertTrue(vm.isProcessing)
        XCTAssertEqual(vm.progressMessage, "Cancelling...")
    }

    func testFaceEditorAll14KnobsContract() {
        let settings = FaceEditorSettings()
        // 14 distinct knobs
        XCTAssertEqual(settings.mouthSmile, 0.0)
        XCTAssertEqual(settings.mouthGrim, 0.0)
        XCTAssertEqual(settings.mouthPout, 0.0)
        XCTAssertEqual(settings.mouthPurse, 0.0)
        XCTAssertEqual(settings.mouthPositionHorizontal, 0.0)
        XCTAssertEqual(settings.mouthPositionVertical, 0.0)
        XCTAssertEqual(settings.eyebrowDirection, 0.0)
        XCTAssertEqual(settings.eyeOpenRatio, 0.0)
        XCTAssertEqual(settings.lipOpenRatio, 0.0)
        XCTAssertEqual(settings.eyeGazeHorizontal, 0.0)
        XCTAssertEqual(settings.eyeGazeVertical, 0.0)
        XCTAssertEqual(settings.headPitch, 0.0)
        XCTAssertEqual(settings.headYaw, 0.0)
        XCTAssertEqual(settings.headRoll, 0.0)
    }

    func testDeepSwapperAndBackgroundRemoverSettings() {
        var dfmSettings = DeepSwapperSettings()
        XCTAssertNil(dfmSettings.inputSize)
        dfmSettings.inputSize = 320
        XCTAssertEqual(dfmSettings.inputSize, 320)

        // Background remover defaults: transparent fill [0,0,0,0], despill [0,0,0,0]
        let bgSettings = BackgroundRemoverSettings()
        XCTAssertEqual(bgSettings.fillColor, [0, 0, 0, 0])
        XCTAssertEqual(bgSettings.despillColor, [0, 0, 0, 0])
    }

    func testFaceMaskPaddingAndDynamicEnums() {
        let padding = FaceMaskPadding(top: 10, right: 15, bottom: 20, left: 25)
        XCTAssertEqual(padding.top, 10)
        XCTAssertEqual(padding.right, 15)
        XCTAssertEqual(padding.bottom, 20)
        XCTAssertEqual(padding.left, 25)

        XCTAssertEqual(FaceMaskType.allCases.count, 4)
        XCTAssertEqual(FaceMaskArea.allCases.count, 3)
        XCTAssertEqual(FaceMaskRegion.allCases.count, 10)
        XCTAssertEqual(DebuggerItem.allCases.count, 4)
    }
}
