import XCTest
import simd
import UIKit
@testable import iFaceFusion

/// Unit tests for FrameColorizerProcessor: exact CIE L*a*b* D65 conversions, reference roundtrips,
/// neutral chroma invariance, strict shape & finiteness validation, and memory-bounded luminance recombination.
final class ColorizerTests: XCTestCase {

    // MARK: - 1. Exact Reference Roundtrips (Black, White, Red)

    func testExactSRGBToLabAndBackReferenceColors() {
        // Black: RGB (0, 0, 0) -> Lab (0, 0, 0)
        let (lBlack, aBlack, bBlack) = FrameColorizerProcessor.sRGBToLab(r: 0.0, g: 0.0, b: 0.0)
        XCTAssertEqual(lBlack, 0.0, accuracy: 1e-4)
        XCTAssertEqual(aBlack, 0.0, accuracy: 1e-4)
        XCTAssertEqual(bBlack, 0.0, accuracy: 1e-4)

        let (rBlackRec, gBlackRec, bBlackRec) = FrameColorizerProcessor.labToSRGB(L: lBlack, a: aBlack, b: bBlack)
        XCTAssertEqual(rBlackRec, 0.0, accuracy: 1e-4)
        XCTAssertEqual(gBlackRec, 0.0, accuracy: 1e-4)
        XCTAssertEqual(bBlackRec, 0.0, accuracy: 1e-4)

        // White: RGB (1, 1, 1) -> Lab (100, 0, 0)
        let (lWhite, aWhite, bWhite) = FrameColorizerProcessor.sRGBToLab(r: 1.0, g: 1.0, b: 1.0)
        XCTAssertEqual(lWhite, 100.0, accuracy: 1e-4)
        XCTAssertEqual(aWhite, 0.0, accuracy: 1e-4)
        XCTAssertEqual(bWhite, 0.0, accuracy: 1e-4)

        let (rWhiteRec, gWhiteRec, bWhiteRec) = FrameColorizerProcessor.labToSRGB(L: lWhite, a: aWhite, b: bWhite)
        XCTAssertEqual(rWhiteRec, 1.0, accuracy: 1e-4)
        XCTAssertEqual(gWhiteRec, 1.0, accuracy: 1e-4)
        XCTAssertEqual(bWhiteRec, 1.0, accuracy: 1e-4)

        // Red: RGB (1, 0, 0) -> Lab approx (53.24, 80.09, 67.20)
        let (lRed, aRed, bRed) = FrameColorizerProcessor.sRGBToLab(r: 1.0, g: 0.0, b: 0.0)
        XCTAssertEqual(lRed, 53.2406, accuracy: 0.01)
        XCTAssertEqual(aRed, 80.0942, accuracy: 0.01)
        XCTAssertEqual(bRed, 67.2015, accuracy: 0.01)

        let (rRedRec, gRedRec, bRedRec) = FrameColorizerProcessor.labToSRGB(L: lRed, a: aRed, b: bRed)
        XCTAssertEqual(rRedRec, 1.0, accuracy: 1e-4)
        XCTAssertEqual(gRedRec, 0.0, accuracy: 1e-4)
        XCTAssertEqual(bRedRec, 0.0, accuracy: 1e-4)
    }

    // MARK: - 2. Neutral Chroma Invariance (a = 0, b = 0 => R == G == B)

    func testNeutralChromaYieldsEqualChannels() {
        let testLightnessValues: [Float] = [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0]
        for L in testLightnessValues {
            let (r, g, b) = FrameColorizerProcessor.labToSRGB(L: L, a: 0.0, b: 0.0)
            XCTAssertEqual(r, g, accuracy: 1e-5, "R and G mismatch at L=\(L)")
            XCTAssertEqual(g, b, accuracy: 1e-5, "G and B mismatch at L=\(L)")
            XCTAssertGreaterThanOrEqual(r, 0.0)
            XCTAssertLessThanOrEqual(r, 1.0)
        }
    }

    // MARK: - 3. Fast Luminance Equivalence

    func testFastLuminanceMatchesFullLabL() {
        let testColors: [(Float, Float, Float)] = [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.5, 0.5),
            (0.8, 0.2, 0.4)
        ]
        for (r, g, b) in testColors {
            let directL = FrameColorizerProcessor.sRGBToLuminance(r: r, g: g, b: b)
            let fullLab = FrameColorizerProcessor.sRGBToLab(r: r, g: g, b: b)
            XCTAssertEqual(directL, fullLab.L, accuracy: 1e-5, "Luminance mismatch for RGB (\(r), \(g), \(b))")
        }
    }

    // MARK: - 4. Strict Shape & Finiteness Validation

    func testRecombineRejectsShapeMismatch() {
        let target = ImageBuffer(width: 8, height: 8)
        let modelSize = 4
        // Expected: 2 * 4 * 4 = 32 elements.
        // Old buggy behavior provided 3 channels = 48 elements.
        let threeChannelTensor = [Float](repeating: 0.0, count: 3 * modelSize * modelSize)

        XCTAssertThrowsError(
            try FrameColorizerProcessor.recombine(
                targetImage: target,
                colorTensor: threeChannelTensor,
                modelSize: modelSize
            )
        ) { error in
            guard case ORTBridgeError.inferenceFailed(let msg) = error else {
                XCTFail("Expected ORTBridgeError.inferenceFailed, got: \(error)")
                return
            }
            XCTAssertTrue(msg.contains("shape mismatch") || msg.contains("2 channels"), "Message: \(msg)")
        }

        // Also test empty tensor
        XCTAssertThrowsError(
            try FrameColorizerProcessor.recombine(
                targetImage: target,
                colorTensor: [],
                modelSize: modelSize
            )
        )
    }

    func testRecombineRejectsNonFiniteValues() {
        let target = ImageBuffer(width: 8, height: 8)
        let modelSize = 4
        let count = 2 * modelSize * modelSize

        // Test NaN
        var nanTensor = [Float](repeating: 0.0, count: count)
        nanTensor[5] = Float.nan
        XCTAssertThrowsError(
            try FrameColorizerProcessor.recombine(
                targetImage: target,
                colorTensor: nanTensor,
                modelSize: modelSize
            )
        )

        // Test Infinity
        var infTensor = [Float](repeating: 0.0, count: count)
        infTensor[10] = Float.infinity
        XCTAssertThrowsError(
            try FrameColorizerProcessor.recombine(
                targetImage: target,
                colorTensor: infTensor,
                modelSize: modelSize
            )
        )
    }

    // MARK: - 5. Preservation of Dimensions, Alpha, and Luminance

    func testRecombinePreservesOriginalDimensionsAndAlpha() throws {
        let origW = 12
        let origH = 16
        var target = ImageBuffer(width: origW, height: origH)

        // Populate with synthetic varying pixels and distinct alpha values
        for y in 0..<origH {
            for x in 0..<origW {
                let idx = (y * origW + x) * 4
                target.data[idx + 0] = UInt8((x * 20) % 256)
                target.data[idx + 1] = UInt8((y * 15) % 256)
                target.data[idx + 2] = UInt8(((x + y) * 10) % 256)
                // Distinct alpha: 0, 64, 128, 255
                target.data[idx + 3] = UInt8(((x + y) % 4) * 85)
            }
        }

        let modelSize = 8
        // Neutral chroma: a = 0, b = 0 everywhere
        let neutralTensor = [Float](repeating: 0.0, count: 2 * modelSize * modelSize)

        let result = try FrameColorizerProcessor.recombine(
            targetImage: target,
            colorTensor: neutralTensor,
            modelSize: modelSize
        )

        XCTAssertEqual(result.width, origW)
        XCTAssertEqual(result.height, origH)
        XCTAssertEqual(result.data.count, origW * origH * 4)

        // Verify every pixel preserves exact target alpha
        for y in 0..<origH {
            for x in 0..<origW {
                let idx = (y * origW + x) * 4
                let expectedAlpha = target.data[idx + 3]
                let actualAlpha = result.data[idx + 3]
                XCTAssertEqual(actualAlpha, expectedAlpha, "Alpha mismatch at (\(x), \(y))")
            }
        }
    }

    func testRecombineNeutralChromaPreservesMonochromeLuminance() throws {
        let width = 10
        let height = 10
        var target = ImageBuffer(width: width, height: height)

        // Flat neutral gray image (128, 128, 128, 255)
        for i in 0..<(width * height) {
            let idx = i * 4
            target.data[idx + 0] = 128
            target.data[idx + 1] = 128
            target.data[idx + 2] = 128
            target.data[idx + 3] = 255
        }

        let modelSize = 4
        let neutralTensor = [Float](repeating: 0.0, count: 2 * modelSize * modelSize)

        let result = try FrameColorizerProcessor.recombine(
            targetImage: target,
            colorTensor: neutralTensor,
            modelSize: modelSize
        )

        for i in 0..<(width * height) {
            let idx = i * 4
            let r = result.data[idx + 0]
            let g = result.data[idx + 1]
            let b = result.data[idx + 2]
            // Neutral chroma must preserve R == G == B
            XCTAssertEqual(r, g)
            XCTAssertEqual(g, b)
            // Should match original gray (128) within rounding tolerance (<= 1)
            XCTAssertEqual(Int(r), 128, accuracy: 1)
        }
    }
}
