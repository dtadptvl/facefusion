import XCTest
import simd
import UIKit
@testable import iFaceFusion

final class FaceGeometryTests: XCTestCase {

    // MARK: - RGBA Half-Alpha PNG Roundtrip & Preservation

    func testRGBAHalfAlphaPNGRoundtrip() {
        let width = 8
        let height = 8
        var buffer = ImageBuffer(width: width, height: height)

        // Pixel 0: Half-alpha pixel RGBA = (100, 150, 200, 128)
        let idx0 = (2 * width + 2) * 4
        buffer.data[idx0 + 0] = 100
        buffer.data[idx0 + 1] = 150
        buffer.data[idx0 + 2] = 200
        buffer.data[idx0 + 3] = 128

        // Pixel 1: Quarter-alpha pixel RGBA = (220, 80, 40, 64)
        let idx1 = (4 * width + 4) * 4
        buffer.data[idx1 + 0] = 220
        buffer.data[idx1 + 1] = 80
        buffer.data[idx1 + 2] = 40
        buffer.data[idx1 + 3] = 64

        // Pixel 2: Fully transparent pixel RGBA = (0, 0, 0, 0)
        let idx2 = (6 * width + 6) * 4
        buffer.data[idx2 + 0] = 0
        buffer.data[idx2 + 1] = 0
        buffer.data[idx2 + 2] = 0
        buffer.data[idx2 + 3] = 0

        // Export to UIImage -> PNG Data -> UIImage -> ImageBuffer
        guard let uiImage = buffer.toUIImage(),
              let pngData = uiImage.pngData(),
              let decodedImage = UIImage(data: pngData),
              let roundtripBuffer = ImageBuffer(image: decodedImage) else {
            XCTFail("Failed to encode/decode PNG roundtrip for half-alpha image")
            return
        }

        XCTAssertEqual(roundtripBuffer.width, width)
        XCTAssertEqual(roundtripBuffer.height, height)

        // Pixel 0: Half-alpha should preserve straight color within 8-bit quantization tolerance (<= 2 units)
        XCTAssertEqual(Float(roundtripBuffer.data[idx0 + 0]), 100.0, accuracy: 2.0)
        XCTAssertEqual(Float(roundtripBuffer.data[idx0 + 1]), 150.0, accuracy: 2.0)
        XCTAssertEqual(Float(roundtripBuffer.data[idx0 + 2]), 200.0, accuracy: 2.0)
        XCTAssertEqual(roundtripBuffer.data[idx0 + 3], 128)

        // Pixel 1: Quarter-alpha preservation
        XCTAssertEqual(Float(roundtripBuffer.data[idx1 + 0]), 220.0, accuracy: 2.0)
        XCTAssertEqual(Float(roundtripBuffer.data[idx1 + 1]), 80.0, accuracy: 2.0)
        XCTAssertEqual(Float(roundtripBuffer.data[idx1 + 2]), 40.0, accuracy: 2.0)
        XCTAssertEqual(roundtripBuffer.data[idx1 + 3], 64)

        // Pixel 2: Fully transparent pixel maintains zero alpha
        XCTAssertEqual(roundtripBuffer.data[idx2 + 3], 0)
    }

    // MARK: - 0/1/2 Face Count Helper Invariants

    func testFaceCountHelperZeroOneTwo() {
        // Count 0: Must throw .noFaceDetected
        XCTAssertThrowsError(try FaceDetector.validateFaceCount(0)) { error in
            guard let detectorError = error as? FaceDetectorError,
                  detectorError == .noFaceDetected else {
                XCTFail("Expected .noFaceDetected for 0 faces, got \(error)")
                return
            }
        }

        // Count 1: Must succeed and return 1
        XCTAssertNoThrow(try FaceDetector.validateFaceCount(1))
        XCTAssertEqual(try? FaceDetector.validateFaceCount(1), 1)

        // Count 2: Must throw .multipleFacesDetected(2)
        XCTAssertThrowsError(try FaceDetector.validateFaceCount(2)) { error in
            guard let detectorError = error as? FaceDetectorError,
                  detectorError == .multipleFacesDetected(2) else {
                XCTFail("Expected .multipleFacesDetected(2) for 2 faces, got \(error)")
                return
            }
        }

        // selectSingleFace helper tests
        XCTAssertThrowsError(try FaceDetector.selectSingleFace([String]())) { error in
            XCTAssertEqual(error as? FaceDetectorError, .noFaceDetected)
        }

        let single = try? FaceDetector.selectSingleFace(["primary_face"])
        XCTAssertEqual(single, "primary_face")

        XCTAssertThrowsError(try FaceDetector.selectSingleFace(["face_1", "face_2"])) { error in
            XCTAssertEqual(error as? FaceDetectorError, .multipleFacesDetected(2))
        }
    }

    // MARK: - 5 Landmarks Semantic Ordering & Extrema

    func testSemanticVision5OrderingAndExtrema() throws {
        // Construct synthetic regions with deliberately reversed eye order
        // Left eye region (in input) is situated at viewer's right (X = 65)
        // Right eye region (in input) is situated at viewer's left (X = 35)
        let invertedEyeLeft = [
            SIMD2<Float>(60.0, 50.0), SIMD2<Float>(65.0, 45.0),
            SIMD2<Float>(70.0, 50.0), SIMD2<Float>(65.0, 55.0)
        ]
        let invertedEyeRight = [
            SIMD2<Float>(30.0, 50.0), SIMD2<Float>(35.0, 45.0),
            SIMD2<Float>(40.0, 50.0), SIMD2<Float>(35.0, 55.0)
        ]

        let outerLips = [
            SIMD2<Float>(32.0, 90.0), // Leftmost X extrema
            SIMD2<Float>(50.0, 84.0),
            SIMD2<Float>(68.0, 90.0), // Rightmost X extrema
            SIMD2<Float>(50.0, 96.0)
        ]

        let innerLips = [
            SIMD2<Float>(38.0, 90.0),
            SIMD2<Float>(50.0, 88.0),
            SIMD2<Float>(62.0, 90.0),
            SIMD2<Float>(50.0, 92.0)
        ]

        let noseCrest = [
            SIMD2<Float>(50.0, 45.0),
            SIMD2<Float>(50.0, 55.0),
            SIMD2<Float>(50.0, 65.0)
        ]

        let noseBase = [
            SIMD2<Float>(42.0, 74.0),
            SIMD2<Float>(50.0, 72.0), // Actual apex / columella tip
            SIMD2<Float>(58.0, 74.0)
        ]

        let faceContour = [
            SIMD2<Float>(20.0, 60.0),
            SIMD2<Float>(50.0, 110.0),
            SIMD2<Float>(80.0, 60.0)
        ]

        let leftEyebrow = [SIMD2<Float>(30.0, 40.0), SIMD2<Float>(42.0, 38.0)]
        let rightEyebrow = [SIMD2<Float>(58.0, 38.0), SIMD2<Float>(70.0, 40.0)]

        let regions = RawVisionLandmarkRegions(
            faceContour: faceContour,
            leftEyebrow: leftEyebrow,
            rightEyebrow: rightEyebrow,
            noseCrest: noseCrest,
            nose: noseBase,
            leftEye: invertedEyeLeft,
            rightEye: invertedEyeRight,
            outerLips: outerLips,
            innerLips: innerLips
        )

        let (_, landmark5) = try FaceDetector.mapLandmarks(from: regions)

        // 1. Eye centers must be strictly sorted left-to-right (viewer coordinates)
        XCTAssertLessThan(landmark5.leftEye.x, landmark5.rightEye.x)
        XCTAssertEqual(landmark5.leftEye.x, 35.0, accuracy: 1e-3)
        XCTAssertEqual(landmark5.rightEye.x, 65.0, accuracy: 1e-3)

        // 2. Lips extrema: left corner must be min X (32.0), right corner max X (68.0)
        XCTAssertLessThan(landmark5.leftMouth.x, landmark5.rightMouth.x)
        XCTAssertEqual(landmark5.leftMouth.x, 32.0, accuracy: 1e-3)
        XCTAssertEqual(landmark5.rightMouth.x, 68.0, accuracy: 1e-3)

        // 3. Nose tip: Anatomical tip positioned at apex between crest and nose base, not arbitrary truncated crest end
        XCTAssertEqual(landmark5.nose.x, 50.0, accuracy: 1e-3)
        XCTAssertGreaterThan(landmark5.nose.y, noseCrest.first!.y)
        XCTAssertLessThanOrEqual(landmark5.nose.y, noseBase[1].y)
    }

    // MARK: - Region Resampling & Geometric Orientation Normalization

    func testRegionInterpolationAndOrientationNormalization() throws {
        // Sparse synthetic regions (e.g. 3-point jaw, 2-point brows, reversed direction)
        let jawReversed = [
            SIMD2<Float>(80.0, 60.0),
            SIMD2<Float>(50.0, 110.0),
            SIMD2<Float>(20.0, 60.0)
        ]

        let browL = [SIMD2<Float>(42.0, 38.0), SIMD2<Float>(30.0, 40.0)] // reversed
        let browR = [SIMD2<Float>(70.0, 40.0), SIMD2<Float>(58.0, 38.0)] // reversed

        let crest = [SIMD2<Float>(50.0, 65.0), SIMD2<Float>(50.0, 45.0)] // reversed Y
        let base = [SIMD2<Float>(58.0, 74.0), SIMD2<Float>(50.0, 72.0), SIMD2<Float>(42.0, 74.0)] // reversed X

        let eyeL = [
            SIMD2<Float>(30.0, 50.0), SIMD2<Float>(35.0, 45.0),
            SIMD2<Float>(40.0, 50.0), SIMD2<Float>(35.0, 55.0)
        ]
        let eyeR = [
            SIMD2<Float>(60.0, 50.0), SIMD2<Float>(65.0, 45.0),
            SIMD2<Float>(70.0, 50.0), SIMD2<Float>(65.0, 55.0)
        ]

        let outerLips = [
            SIMD2<Float>(32.0, 90.0), SIMD2<Float>(50.0, 84.0),
            SIMD2<Float>(68.0, 90.0), SIMD2<Float>(50.0, 96.0)
        ]
        let innerLips = [
            SIMD2<Float>(38.0, 90.0), SIMD2<Float>(50.0, 88.0),
            SIMD2<Float>(62.0, 90.0), SIMD2<Float>(50.0, 92.0)
        ]

        let regions = RawVisionLandmarkRegions(
            faceContour: jawReversed,
            leftEyebrow: browL,
            rightEyebrow: browR,
            noseCrest: crest,
            nose: base,
            leftEye: eyeL,
            rightEye: eyeR,
            outerLips: outerLips,
            innerLips: innerLips
        )

        let (l68, _) = try FaceDetector.mapLandmarks(from: regions)

        // Must populate all 68 points without leaving any zero points
        XCTAssertEqual(l68.points.count, 68)
        for i in 0..<68 {
            let pt = l68.points[i]
            XCTAssertFalse(pt.x == 0 && pt.y == 0, "Landmark index \(i) was left as zero (0, 0)")
        }

        // Jawline (0..<17) must be normalized left-to-right
        XCTAssertLessThan(l68.points[0].x, l68.points[16].x)

        // Eyebrows (17..<22 and 22..<27) must be normalized left-to-right
        XCTAssertLessThan(l68.points[17].x, l68.points[21].x)
        XCTAssertLessThan(l68.points[22].x, l68.points[26].x)

        // Nose crest (27..<31) must run top-to-bottom
        XCTAssertLessThan(l68.points[27].y, l68.points[30].y)

        // Outer lips: 48 is leftmost corner, 54 is rightmost corner
        XCTAssertEqual(l68.points[48].x, 32.0, accuracy: 1e-3)
        XCTAssertEqual(l68.points[54].x, 68.0, accuracy: 1e-3)

        // Inner lips: 60 is leftmost corner, 64 is rightmost corner
        XCTAssertEqual(l68.points[60].x, 38.0, accuracy: 1e-3)
        XCTAssertEqual(l68.points[64].x, 62.0, accuracy: 1e-3)
    }

    // MARK: - Strict Missing Required Landmarks Error

    func testStrictMissingRequiredLandmarksError() {
        // Omit faceContour entirely
        let regionsMissingJaw = RawVisionLandmarkRegions(
            faceContour: nil,
            leftEyebrow: [SIMD2<Float>(30, 40), SIMD2<Float>(40, 40)],
            rightEyebrow: [SIMD2<Float>(60, 40), SIMD2<Float>(70, 40)],
            noseCrest: [SIMD2<Float>(50, 50), SIMD2<Float>(50, 60)],
            nose: [SIMD2<Float>(45, 70), SIMD2<Float>(55, 70)],
            leftEye: [SIMD2<Float>(30, 50), SIMD2<Float>(40, 50), SIMD2<Float>(35, 55)],
            rightEye: [SIMD2<Float>(60, 50), SIMD2<Float>(70, 50), SIMD2<Float>(65, 55)],
            outerLips: [SIMD2<Float>(35, 90), SIMD2<Float>(65, 90), SIMD2<Float>(50, 95)],
            innerLips: [SIMD2<Float>(40, 90), SIMD2<Float>(60, 90), SIMD2<Float>(50, 92)]
        )

        XCTAssertThrowsError(try FaceDetector.mapLandmarks(from: regionsMissingJaw)) { error in
            guard let detErr = error as? FaceDetectorError,
                  case .detectionFailed(let reason) = detErr else {
                XCTFail("Expected .detectionFailed, got \(error)")
                return
            }
            XCTAssertTrue(reason.contains("faceContour"))
        }

        // Omit outerLips
        var regionsMissingLips = regionsMissingJaw
        regionsMissingLips.faceContour = [SIMD2<Float>(20, 60), SIMD2<Float>(80, 60)]
        regionsMissingLips.outerLips = nil

        XCTAssertThrowsError(try FaceDetector.mapLandmarks(from: regionsMissingLips)) { error in
            guard let detErr = error as? FaceDetectorError,
                  case .detectionFailed(let reason) = detErr else {
                XCTFail("Expected .detectionFailed, got \(error)")
                return
            }
            XCTAssertTrue(reason.contains("outerLips"))
        }
    }

    // MARK: - Background Remover Compositing & Despill

    func testBackgroundRemoverStraightAlphaCompositing() {
        var image = ImageBuffer(width: 2, height: 2)
        // Set all pixels to semi-transparent foreground: RGBA = (200, 100, 50, 128)
        for i in 0..<4 {
            let idx = i * 4
            image.data[idx + 0] = 200
            image.data[idx + 1] = 100
            image.data[idx + 2] = 50
            image.data[idx + 3] = 128
        }

        // Mask: Pixel (0, 0) is foreground (1.0), Pixel (1, 1) is background (0.0)
        let mask = FaceMask(width: 2, height: 2, values: [1.0, 0.5, 0.5, 0.0])

        // 1. Transparent fill [0, 0, 0, 0]: Must preserve original alpha * matte
        var testTransparent = image.clone()
        BackgroundRemoverProcessor.applyFillColor(to: &testTransparent, mask: mask, fillRGBA: [0, 0, 0, 0])

        // (0, 0): matte 1.0 -> alpha remains 128, color remains (200, 100, 50)
        XCTAssertEqual(testTransparent.data[0], 200)
        XCTAssertEqual(testTransparent.data[1], 100)
        XCTAssertEqual(testTransparent.data[2], 50)
        XCTAssertEqual(testTransparent.data[3], 128)

        // (1, 1): matte 0.0 -> alpha becomes 0
        let idx11 = 3 * 4
        XCTAssertEqual(testTransparent.data[idx11 + 3], 0)

        // 2. Partial fill [0, 255, 0, 128] (50% green fill)
        var testPartial = image.clone()
        BackgroundRemoverProcessor.applyFillColor(to: &testPartial, mask: mask, fillRGBA: [0, 255, 0, 128])

        // (1, 1): matte 0.0 -> pure background fill color (0, 255, 0) with alpha 128
        XCTAssertEqual(testPartial.data[idx11 + 0], 0)
        XCTAssertEqual(testPartial.data[idx11 + 1], 255)
        XCTAssertEqual(testPartial.data[idx11 + 2], 0)
        XCTAssertEqual(testPartial.data[idx11 + 3], 128)

        // 3. Dominant-channel despill for Green, Blue, and Red
        var spillImage = ImageBuffer(width: 3, height: 1)
        // Green spill pixel: (50, 220, 60, 255)
        spillImage.data[0] = 50; spillImage.data[1] = 220; spillImage.data[2] = 60; spillImage.data[3] = 255
        // Blue spill pixel: (40, 50, 210, 255)
        spillImage.data[4] = 40; spillImage.data[5] = 50; spillImage.data[6] = 210; spillImage.data[7] = 255
        // Red spill pixel: (230, 40, 50, 255)
        spillImage.data[8] = 230; spillImage.data[9] = 40; spillImage.data[10] = 50; spillImage.data[11] = 255

        // Despill green with strength 255
        BackgroundRemoverProcessor.applyDespillColor(to: &spillImage, despillRGBA: [0, 255, 0, 255])
        // Green should be clamped to max(50, 60) = 60
        XCTAssertEqual(spillImage.data[1], 60)

        // Despill blue with strength 255
        BackgroundRemoverProcessor.applyDespillColor(to: &spillImage, despillRGBA: [0, 0, 255, 255])
        // Blue should be clamped to max(40, 50) = 50
        XCTAssertEqual(spillImage.data[6], 50)

        // Despill red with strength 255
        BackgroundRemoverProcessor.applyDespillColor(to: &spillImage, despillRGBA: [255, 0, 0, 255])
        // Red should be clamped to max(40, 50) = 50
        XCTAssertEqual(spillImage.data[8], 50)
    }
}
