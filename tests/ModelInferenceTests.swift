import XCTest
import simd
import UIKit
@testable import iFaceFusion

/// Real CPU inference smoke tests for default neural processors, auxiliary models, and DFM handling.
/// Uses synthetic portrait-like buffers and manual valid FaceTarget to validate pipeline tensor runtimes.
final class ModelInferenceTests: XCTestCase {

    private let modelCache = ModelCache.shared
    private let ortBridge = ORTBridge.shared

    override func tearDown() async throws {
        await ortBridge.releaseSession()
        try await super.tearDown()
    }

    // MARK: - Test Fixture Builders

    private func createSyntheticPortrait512() -> ImageBuffer {
        var buffer = ImageBuffer(width: 512, height: 512)
        for y in 0..<512 {
            let rowOffset = y * 512 * 4
            let bgGrad = UInt8(min(max(40 + y / 4, 0), 255))
            for x in 0..<512 {
                let idx = rowOffset + x * 4
                // Face oval centered at (256, 260) with rx = 110, ry = 140
                let dx = Float(x - 256) / 110.0
                let dy = Float(y - 260) / 140.0
                let distSq = dx * dx + dy * dy
                if distSq <= 1.0 {
                    // Face skin tone
                    buffer.data[idx + 0] = 230
                    buffer.data[idx + 1] = 185
                    buffer.data[idx + 2] = 150
                    buffer.data[idx + 3] = 255
                } else {
                    buffer.data[idx + 0] = 30
                    buffer.data[idx + 1] = 40
                    buffer.data[idx + 2] = bgGrad
                    buffer.data[idx + 3] = 255
                }
            }
        }
        return buffer
    }

    private func createValidFaceTarget(width: Int = 512, height: Int = 512) -> FaceTarget {
        let w = Float(width)
        let h = Float(height)
        let templatePts = WarpTemplate.ffhq512.targetPoints(width: w, height: h)
        let l5 = FaceLandmark5(points: templatePts)

        // Construct 68 landmarks aligned with template points
        var l68Pts = [SIMD2<Float>](repeating: templatePts[2], count: 68)

        // Jawline: 0..<17
        for i in 0..<17 {
            let angle = Float.pi * Float(i) / 16.0
            let jx = 256.0 - cos(angle) * 110.0
            let jy = 260.0 + sin(angle) * 130.0
            l68Pts[i] = SIMD2<Float>(jx, jy)
        }
        // Left eyebrow: 17..<22
        for i in 17..<22 {
            l68Pts[i] = SIMD2<Float>(170.0 + Float(i - 17) * 12.0, 210.0)
        }
        // Right eyebrow: 22..<27
        for i in 22..<27 {
            l68Pts[i] = SIMD2<Float>(280.0 + Float(i - 22) * 12.0, 210.0)
        }
        // Nose bridge: 27..<31
        for i in 27..<31 {
            l68Pts[i] = SIMD2<Float>(templatePts[2].x, 230.0 + Float(i - 27) * 20.0)
        }
        // Nose nostrils & tip: 31..<36 (30 is tip)
        l68Pts[30] = templatePts[2]
        l68Pts[31] = templatePts[2] + SIMD2<Float>(-15, -5)
        l68Pts[32] = templatePts[2] + SIMD2<Float>(-8, 0)
        l68Pts[33] = templatePts[2] + SIMD2<Float>(0, 2)
        l68Pts[34] = templatePts[2] + SIMD2<Float>(8, 0)
        l68Pts[35] = templatePts[2] + SIMD2<Float>(15, -5)

        // Left eye: 36..<42 (centered at templatePts[0])
        for i in 36..<42 {
            let offsetAngle = Float(i - 36) * Float.pi / 3.0
            l68Pts[i] = templatePts[0] + SIMD2<Float>(cos(offsetAngle) * 8.0, sin(offsetAngle) * 4.0)
        }
        // Right eye: 42..<48 (centered at templatePts[1])
        for i in 42..<48 {
            let offsetAngle = Float(i - 42) * Float.pi / 3.0
            l68Pts[i] = templatePts[1] + SIMD2<Float>(cos(offsetAngle) * 8.0, sin(offsetAngle) * 4.0)
        }
        // Mouth: 48..<68 (48 is left corner, 54 is right corner)
        l68Pts[48] = templatePts[3]
        l68Pts[54] = templatePts[4]
        for i in 49..<54 {
            let t = Float(i - 48) / 6.0
            l68Pts[i] = templatePts[3] * (1.0 - t) + templatePts[4] * t + SIMD2<Float>(0, sin(t * Float.pi) * 8.0)
        }
        for i in 55..<68 {
            let t = Float(i - 54) / 14.0
            l68Pts[i] = templatePts[3] * t + templatePts[4] * (1.0 - t)
        }

        let l68 = FaceLandmark68(points: l68Pts)
        let bbox = CGRect(x: 128, y: 120, width: 256, height: 280)
        return FaceTarget(boundingBox: bbox, landmark5: l5, landmark68: l68, angle: 0, score: 0.99, age: 35.0)
    }

    private func getOrCreateTinyDFMFixtureURL() throws -> URL {
        if let bundleURL = Bundle(for: Self.self).url(forResource: "tiny_dfm_224", withExtension: "onnx") {
            return bundleURL
        }
        let directPath = "tests/fixtures/tiny_dfm_224.onnx"
        if FileManager.default.fileExists(atPath: directPath) {
            return URL(fileURLWithPath: directPath)
        }
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("tiny_dfm_224.onnx")
        if FileManager.default.fileExists(atPath: tempURL.path) {
            return tempURL
        }
        let b64 = "CA4SGGlGYWNlRnVzaW9uVGVzdEdlbmVyYXRvcjqwAwpCCgppbnB1dF9mYWNlCgZzdGFydHMKBGVuZHMKBGF4ZXMKBXN0ZXBzEhJjdXN0b21fdGFyZ2V0X21hc2siBVNsaWNlCisKCmlucHV0X2ZhY2USE2N1c3RvbV92aXNpb25fZnJhbWUiCElkZW50aXR5CjIKEmN1c3RvbV90YXJnZXRfbWFzaxISY3VzdG9tX3NvdXJjZV9tYXNrIghJZGVudGl0eRIRdGlueV9kZm1fbm9ubW9ycGgqDwgBEAc6AQBCBnN0YXJ0cyoNCAEQBzoBAUIEZW5kcyoNCAEQBzoBA0IEYXhlcyoOCAEQBzoBAUIFc3RlcHNaJgoKaW5wdXRfZmFjZRIYChYIARISCgIIAQoDCOABCgMI4AEKAggDYi4KEmN1c3RvbV90YXJnZXRfbWFzaxIYChYIARISCgIIAQoDCOABCgMI4AEKAggBYi8KE2N1c3RvbV92aXNpb25fZnJhbWUSGAoWCAESEgoCCAEKAwjgAQoDCOABCgIIA2IuChJjdXN0b21fc291cmNlX21hc2sSGAoWCAESEgoCCAEKAwjgAQoDCOABCgIIAUIECgAQDQ=="
        guard let data = Data(base64Encoded: b64) else {
            throw ORTBridgeError.sessionCreationFailed("Failed to decode tiny DFM fixture data")
        }
        try data.write(to: tempURL)
        return tempURL
    }

    // MARK: - 1. Face Debugger (Non-Neural)

    func test01_FaceDebuggerProcessor_NonNeural() {
        let processor = FaceDebuggerProcessor()
        let target = createSyntheticPortrait512()
        let face = createValidFaceTarget()
        let settings = FaceDebuggerSettings(items: [.boundingBox, .faceMask, .landmark5, .landmark68])
        let maskSettings = FaceMaskSettings(types: [.box])

        let result = processor.process(targetImage: target, targetFace: face, settings: settings, maskSettings: maskSettings)

        XCTAssertEqual(result.width, 512)
        XCTAssertEqual(result.height, 512)
        XCTAssertEqual(result.data.count, 512 * 512 * 4)
        XCTAssertTrue(result.data.contains(where: { $0 > 0 }))
    }

    // MARK: - 2. Face Swapper (HyperSwap 1a 256 + ArcFace)

    func test02_FaceSwapperProcessor_Inference() async throws {
        let processor = FaceSwapperProcessor()
        let target = createSyntheticPortrait512()
        let face = createValidFaceTarget()
        let settings = FaceSwapperSettings(model: "hyperswap_1a_256", pixelBoost: "256x256", weight: 0.5)
        let maskSettings = FaceMaskSettings(types: [.box])

        let arcfaceMeta = ModelCatalog.arcfaceW600kR50
        let swapperMeta = ModelCatalog.hyperswap1a256
        let arcfaceURL = try await modelCache.ensureModelDownloaded(arcfaceMeta)
        let swapperURL = try await modelCache.ensureModelDownloaded(swapperMeta)
        XCTAssertTrue(FileManager.default.fileExists(atPath: arcfaceURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: swapperURL.path))

        let result = try await processor.process(
            sourceImage: target,
            sourceFace: face,
            targetImage: target,
            targetFace: face,
            settings: settings,
            maskSettings: maskSettings,
            modelCache: modelCache,
            ortBridge: ortBridge
        )

        XCTAssertEqual(result.width, 512)
        XCTAssertEqual(result.height, 512)
        XCTAssertEqual(result.data.count, 512 * 512 * 4)
        XCTAssertTrue(result.data.contains(where: { $0 > 0 }))
    }

    // MARK: - 3. Face Enhancer (GFPGAN 1.4)

    func test03_FaceEnhancerProcessor_Inference() async throws {
        let processor = FaceEnhancerProcessor()
        let target = createSyntheticPortrait512()
        let face = createValidFaceTarget()
        let settings = FaceEnhancerSettings(model: "gfpgan_1.4", blend: 80, weight: 0.5)
        let maskSettings = FaceMaskSettings(types: [.box])

        let gfpganMeta = ModelCatalog.gfpgan14
        let gfpganURL = try await modelCache.ensureModelDownloaded(gfpganMeta)
        XCTAssertTrue(FileManager.default.fileExists(atPath: gfpganURL.path))

        let result = try await processor.process(
            targetImage: target,
            targetFace: face,
            settings: settings,
            maskSettings: maskSettings,
            modelCache: modelCache,
            ortBridge: ortBridge
        )

        XCTAssertEqual(result.width, 512)
        XCTAssertEqual(result.height, 512)
        XCTAssertTrue(result.data.contains(where: { $0 > 0 }))
    }

    // MARK: - 4. Frame Enhancer (SPAN Kendata x4)

    func test04_FrameEnhancerProcessor_Inference() async throws {
        let processor = FrameEnhancerProcessor()
        // Low-cost option: 128x128 single tile input for fast CPU verification
        let input128 = ImageBuffer(width: 128, height: 128)
        let settings = FrameEnhancerSettings(model: "span_kendata_x4", blend: 80)

        let spanMeta = ModelCatalog.spanKendataX4
        let spanURL = try await modelCache.ensureModelDownloaded(spanMeta)
        XCTAssertTrue(FileManager.default.fileExists(atPath: spanURL.path))

        let result = try await processor.process(
            targetImage: input128,
            settings: settings,
            modelCache: modelCache,
            ortBridge: ortBridge
        )

        // 4x upscale: 128x128 -> 512x512
        XCTAssertEqual(result.width, 512)
        XCTAssertEqual(result.height, 512)
        XCTAssertEqual(result.data.count, 512 * 512 * 4)
    }

    // MARK: - 5. Frame Colorizer (DDColor)

    func test05_FrameColorizerProcessor_Inference() async throws {
        let processor = FrameColorizerProcessor()
        let target = createSyntheticPortrait512()
        let settings = FrameColorizerSettings(model: "ddcolor", size: "256x256", blend: 100)

        let ddcolorMeta = ModelCatalog.ddcolor
        let ddcolorURL = try await modelCache.ensureModelDownloaded(ddcolorMeta)
        XCTAssertTrue(FileManager.default.fileExists(atPath: ddcolorURL.path))

        let result = try await processor.process(
            targetImage: target,
            settings: settings,
            modelCache: modelCache,
            ortBridge: ortBridge
        )

        XCTAssertEqual(result.width, 512)
        XCTAssertEqual(result.height, 512)
        XCTAssertTrue(result.data.contains(where: { $0 > 0 }))
    }

    // MARK: - 6. Background Remover (MODNet)

    func test06_BackgroundRemoverProcessor_Inference() async throws {
        let processor = BackgroundRemoverProcessor()
        let target = createSyntheticPortrait512()
        let settings = BackgroundRemoverSettings(model: "modnet")

        let modnetMeta = ModelCatalog.modnet
        let modnetURL = try await modelCache.ensureModelDownloaded(modnetMeta)
        XCTAssertTrue(FileManager.default.fileExists(atPath: modnetURL.path))

        let result = try await processor.process(
            targetImage: target,
            settings: settings,
            modelCache: modelCache,
            ortBridge: ortBridge
        )

        XCTAssertEqual(result.width, 512)
        XCTAssertEqual(result.height, 512)
        XCTAssertTrue(result.data.contains(where: { $0 > 0 }))
    }

    // MARK: - 7. Expression Restorer (LivePortrait)

    func test07_ExpressionRestorerProcessor_Inference() async throws {
        let processor = ExpressionRestorerProcessor()
        let target = createSyntheticPortrait512()
        let face = createValidFaceTarget()
        let settings = ExpressionRestorerSettings(model: "live_portrait", factor: 80)
        let maskSettings = FaceMaskSettings(types: [.box])

        let result = try await processor.process(
            referenceImage: target,
            referenceFace: face,
            targetImage: target,
            targetFace: face,
            settings: settings,
            maskSettings: maskSettings,
            modelCache: modelCache,
            ortBridge: ortBridge
        )

        XCTAssertEqual(result.width, 512)
        XCTAssertEqual(result.height, 512)
        XCTAssertTrue(result.data.contains(where: { $0 > 0 }))
    }

    // MARK: - 8. Face Editor (LivePortrait 6 Sub-models)

    func test08_FaceEditorProcessor_Inference() async throws {
        let processor = FaceEditorProcessor()
        let target = createSyntheticPortrait512()
        let face = createValidFaceTarget()
        let settings = FaceEditorSettings(model: "live_portrait", eyeOpenRatio: 0.2, lipOpenRatio: 0.2, mouthSmile: 0.5)
        let maskSettings = FaceMaskSettings(types: [.box])

        let result = try await processor.process(
            targetImage: target,
            targetFace: face,
            settings: settings,
            maskSettings: maskSettings,
            modelCache: modelCache,
            ortBridge: ortBridge
        )

        XCTAssertEqual(result.width, 512)
        XCTAssertEqual(result.height, 512)
        XCTAssertTrue(result.data.contains(where: { $0 > 0 }))
    }

    // MARK: - 9. Deep Swapper (DFM Tiny Fixture, Ordered Names, Non-Morph)

    func test09_DeepSwapperProcessor_NonMorphFixture() async throws {
        let processor = DeepSwapperProcessor()
        let target = createSyntheticPortrait512()
        let face = createValidFaceTarget()
        let fixtureURL = try getOrCreateTinyDFMFixtureURL()

        // Verify native ORT API describeNames query order
        let (inputs, outputs) = try await ortBridge.describeNames(modelPath: fixtureURL.path)
        XCTAssertTrue(inputs.contains("input_face"), "Must contain actual model input name 'input_face'")
        XCTAssertFalse(inputs.contains("morph_value:0"), "Non-morph fixture must not expose morph input")
        XCTAssertEqual(outputs, ["custom_target_mask", "custom_vision_frame", "custom_source_mask"])

        // Execute inference with explicit inputSize override
        let settings = DeepSwapperSettings(model: fixtureURL.path, morph: 50, inputSize: 224)
        let maskSettings = FaceMaskSettings(types: [.box])

        let result = try await processor.process(
            targetImage: target,
            targetFace: face,
            modelURL: fixtureURL,
            settings: settings,
            maskSettings: maskSettings,
            ortBridge: ortBridge
        )

        XCTAssertEqual(result.width, 512)
        XCTAssertEqual(result.height, 512)
        XCTAssertTrue(result.data.contains(where: { $0 > 0 }))

        // Verify dimension validation: 64 is out of 128...1024
        let invalidLowSettings = DeepSwapperSettings(model: fixtureURL.path, inputSize: 64)
        do {
            _ = try await processor.process(
                targetImage: target,
                targetFace: face,
                modelURL: fixtureURL,
                settings: invalidLowSettings,
                maskSettings: maskSettings,
                ortBridge: ortBridge
            )
            XCTFail("Expected dimension 64 to fail validation (128...1024)")
        } catch {
            // Success: rejected out-of-range dimension
        }
    }

    // MARK: - 10. Age Modifier (FRAN 1024x1024 Heavy Inference)

    func test10_AgeModifierProcessor_FRAN_HeavyInference() async throws {
        let processor = AgeModifierProcessor()
        let target = createSyntheticPortrait512()
        let face = createValidFaceTarget()
        let settings = AgeModifierSettings(model: "fran", direction: 10, sourceAge: 35)
        let maskSettings = FaceMaskSettings(types: [.box])

        let franMeta = ModelCatalog.fran
        let franURL = try await modelCache.ensureModelDownloaded(franMeta)
        XCTAssertTrue(FileManager.default.fileExists(atPath: franURL.path))

        let startTime = CFAbsoluteTimeGetCurrent()
        print("==> Starting FRAN 1024x1024 CPU inference (FRAN is very heavy; measuring duration honestly)...")

        do {
            let result = try await processor.process(
                targetImage: target,
                targetFace: face,
                settings: settings,
                maskSettings: maskSettings,
                modelCache: modelCache,
                ortBridge: ortBridge
            )
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            print("==> FRAN CPU inference completed successfully in \(String(format: "%.2f", elapsed)) seconds.")
            XCTAssertEqual(result.width, 512)
            XCTAssertEqual(result.height, 512)
            XCTAssertTrue(result.data.contains(where: { $0 > 0 }))
        } catch {
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            print("==> FRAN CPU inference stopped after \(String(format: "%.2f", elapsed)) seconds: \(error.localizedDescription)")
            throw error
        }
    }

    // MARK: - Separate Mask Inference Tests (XSeg & BiSeNet)

    func test11_SeparateOcclusionMask_XSeg() async throws {
        let crop = ImageBuffer(width: 256, height: 256)
        let mask = try await ProcessorMasks.createOcclusionMask(
            cropBuffer: crop,
            modelCache: modelCache,
            ortBridge: ortBridge
        )
        XCTAssertEqual(mask.width, 256)
        XCTAssertEqual(mask.height, 256)
        XCTAssertEqual(mask.values.count, 256 * 256)
        for v in mask.values {
            XCTAssertGreaterThanOrEqual(v, 0.0)
            XCTAssertLessThanOrEqual(v, 1.0)
        }
    }

    func test12_SeparateRegionMask_BiSeNet() async throws {
        let crop = ImageBuffer(width: 512, height: 512)
        let mask = try await ProcessorMasks.createRegionMask(
            cropBuffer: crop,
            regions: [.skin, .mouth],
            modelCache: modelCache,
            ortBridge: ortBridge
        )
        XCTAssertEqual(mask.width, 512)
        XCTAssertEqual(mask.height, 512)
        XCTAssertEqual(mask.values.count, 512 * 512)
        for v in mask.values {
            XCTAssertGreaterThanOrEqual(v, 0.0)
            XCTAssertLessThanOrEqual(v, 1.0)
        }
    }
}
