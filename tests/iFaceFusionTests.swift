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
        XCTAssertGreaterThan(farVal, 0.0)

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
}
