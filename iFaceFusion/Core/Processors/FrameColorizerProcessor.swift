import Foundation
import UIKit

/// Implements FaceFusion frame_colorizer processor (DDColor) restoring color to monochrome frames.
// ponytail: Per-pixel D65 CIE Lab conversion uses inline scalar arithmetic; upgrade to vImage/Accelerate batch vector primitives if high-throughput 60fps 4K video colorization is required.
public final class FrameColorizerProcessor: Sendable {
    public init() {}

    // MARK: - CIE 1931 D65 / CIE L*a*b* Color Conversions

    @inline(__always)
    public static func sRGBToLinear(_ c: Float) -> Float {
        if c <= 0.04045 {
            return c / 12.92
        } else {
            return pow((c + 0.055) / 1.055, 2.4)
        }
    }

    @inline(__always)
    public static func linearToSRGB(_ c: Float) -> Float {
        let clamped = max(0.0, min(1.0, c))
        if clamped <= 0.0031308 {
            return clamped * 12.92
        } else {
            return 1.055 * pow(clamped, 1.0 / 2.4) - 0.055
        }
    }

    /// Converts normalized sRGB [0, 1] to CIE L*a*b* under standard D65 illuminant.
    public static func sRGBToLab(r: Float, g: Float, b: Float) -> (L: Float, a: Float, b: Float) {
        let r_l = sRGBToLinear(r)
        let g_l = sRGBToLinear(g)
        let b_l = sRGBToLinear(b)

        // D65 reference white: Xn = 0.950456, Yn = 1.000000, Zn = 1.088754
        let x = 0.412453 * r_l + 0.357580 * g_l + 0.180423 * b_l
        let y = 0.212671 * r_l + 0.715160 * g_l + 0.072169 * b_l
        let z = 0.019334 * r_l + 0.119193 * g_l + 0.950227 * b_l

        let xr = x / 0.950456
        let yr = y / 1.000000
        let zr = z / 1.088754

        let deltaCube: Float = 216.0 / 24389.0 // (6/29)^3 ≈ 0.00885645
        let deltaSqDiv3: Float = (1.0 / 3.0) * (29.0 / 6.0) * (29.0 / 6.0) // 841 / 108 ≈ 7.787037

        let fx = xr > deltaCube ? cbrt(xr) : (deltaSqDiv3 * xr + 4.0 / 29.0)
        let fy = yr > deltaCube ? cbrt(yr) : (deltaSqDiv3 * yr + 4.0 / 29.0)
        let fz = zr > deltaCube ? cbrt(zr) : (deltaSqDiv3 * zr + 4.0 / 29.0)

        let L = 116.0 * fy - 16.0
        let a = 500.0 * (fx - fy)
        let bVal = 200.0 * (fy - fz)
        return (L, a, bVal)
    }

    /// Computes only CIE L* luminance from normalized sRGB [0, 1] (avoids unused X and Z calculations).
    @inline(__always)
    public static func sRGBToLuminance(r: Float, g: Float, b: Float) -> Float {
        let r_l = sRGBToLinear(r)
        let g_l = sRGBToLinear(g)
        let b_l = sRGBToLinear(b)
        let y = 0.212671 * r_l + 0.715160 * g_l + 0.072169 * b_l
        let deltaCube: Float = 216.0 / 24389.0
        let deltaSqDiv3: Float = (1.0 / 3.0) * (29.0 / 6.0) * (29.0 / 6.0)
        let fy = y > deltaCube ? cbrt(y) : (deltaSqDiv3 * y + 4.0 / 29.0)
        return 116.0 * fy - 16.0
    }

    /// Converts CIE L*a*b* back to normalized sRGB [0, 1] under standard D65 illuminant.
    public static func labToSRGB(L: Float, a: Float, b: Float) -> (r: Float, g: Float, b: Float) {
        let fy = (L + 16.0) / 116.0
        let fx = fy + a / 500.0
        let fz = fy - b / 200.0

        let delta: Float = 6.0 / 29.0
        let threeDeltaSq: Float = 3.0 * (6.0 / 29.0) * (6.0 / 29.0)

        let xr = fx > delta ? (fx * fx * fx) : (threeDeltaSq * (fx - 4.0 / 29.0))
        let yr = fy > delta ? (fy * fy * fy) : (threeDeltaSq * (fy - 4.0 / 29.0))
        let zr = fz > delta ? (fz * fz * fz) : (threeDeltaSq * (fz - 4.0 / 29.0))

        let x = xr * 0.950456
        let y = yr * 1.000000
        let z = zr * 1.088754

        let r_l = 3.24048134 * x - 1.53715152 * y - 0.49853633 * z
        let g_l = -0.96925495 * x + 1.87599000 * y + 0.04155593 * z
        let b_l = 0.05564664 * x - 0.20404134 * y + 1.05731107 * z

        let r = linearToSRGB(r_l)
        let g = linearToSRGB(g_l)
        let bVal = linearToSRGB(b_l)
        return (r, g, bVal)
    }

    // MARK: - Recombination

    private struct BilinearWeight {
        let i0: Int
        let i1: Int
        let w0: Float
        let w1: Float
    }

    /// Recombines target image original per-pixel luminance L with predicted 2-channel (a, b) chroma tensor.
    /// Preserves original image dimensions and alpha channel without allocating full-float intermediate image buffers.
    public static func recombine(
        targetImage: ImageBuffer,
        colorTensor: [Float],
        modelSize: Int
    ) throws -> ImageBuffer {
        let origW = targetImage.width
        let origH = targetImage.height
        guard origW > 0, origH > 0, modelSize > 0 else {
            throw ORTBridgeError.inferenceFailed("Invalid image or model dimensions")
        }

        let planeSize = modelSize * modelSize
        let expectedElements = 2 * planeSize
        guard colorTensor.count == expectedElements else {
            throw ORTBridgeError.inferenceFailed("Colorizer tensor shape mismatch: expected 2 channels (\(expectedElements) elements), got \(colorTensor.count)")
        }

        for val in colorTensor {
            guard val.isFinite else {
                throw ORTBridgeError.inferenceFailed("Colorizer tensor contains non-finite values")
            }
        }

        let scaleX = Float(modelSize) / Float(origW)
        var xWeights = [BilinearWeight]()
        xWeights.reserveCapacity(origW)
        for x in 0..<origW {
            let srcX = (Float(x) + 0.5) * scaleX - 0.5
            let x0 = max(0, min(modelSize - 1, Int(floor(srcX))))
            let x1 = max(0, min(modelSize - 1, x0 + 1))
            let fx = max(0.0, min(1.0, srcX - Float(x0)))
            xWeights.append(BilinearWeight(i0: x0, i1: x1, w0: 1.0 - fx, w1: fx))
        }

        let scaleY = Float(modelSize) / Float(origH)
        var result = ImageBuffer(width: origW, height: origH)

        for y in 0..<origH {
            let srcY = (Float(y) + 0.5) * scaleY - 0.5
            let y0 = max(0, min(modelSize - 1, Int(floor(srcY))))
            let y1 = max(0, min(modelSize - 1, y0 + 1))
            let fy = max(0.0, min(1.0, srcY - Float(y0)))
            let wy0 = 1.0 - fy
            let wy1 = fy

            let row0 = y0 * modelSize
            let row1 = y1 * modelSize
            let yOffset = y * origW

            for x in 0..<origW {
                let xw = xWeights[x]
                let idx00 = row0 + xw.i0
                let idx10 = row0 + xw.i1
                let idx01 = row1 + xw.i0
                let idx11 = row1 + xw.i1

                let a00 = colorTensor[idx00]
                let a10 = colorTensor[idx10]
                let a01 = colorTensor[idx01]
                let a11 = colorTensor[idx11]
                let a = wy0 * (xw.w0 * a00 + xw.w1 * a10) + wy1 * (xw.w0 * a01 + xw.w1 * a11)

                let b00 = colorTensor[planeSize + idx00]
                let b10 = colorTensor[planeSize + idx10]
                let b01 = colorTensor[planeSize + idx01]
                let b11 = colorTensor[planeSize + idx11]
                let b = wy0 * (xw.w0 * b00 + xw.w1 * b10) + wy1 * (xw.w0 * b01 + xw.w1 * b11)

                let idx = (yOffset + x) * 4
                let rU8 = targetImage.data[idx + 0]
                let gU8 = targetImage.data[idx + 1]
                let bU8 = targetImage.data[idx + 2]

                // Compute exact luminance L per-pixel from original sRGB
                let origL = sRGBToLuminance(
                    r: Float(rU8) / 255.0,
                    g: Float(gU8) / 255.0,
                    b: Float(bU8) / 255.0
                )

                // Recombine original luminance L with predicted chroma ab -> sRGB
                let (outR, outG, outB) = labToSRGB(L: origL, a: a, b: b)

                result.data[idx + 0] = UInt8(min(max(outR * 255.0, 0.0), 255.0).rounded())
                result.data[idx + 1] = UInt8(min(max(outG * 255.0, 0.0), 255.0).rounded())
                result.data[idx + 2] = UInt8(min(max(outB * 255.0, 0.0), 255.0).rounded())
                result.data[idx + 3] = targetImage.data[idx + 3]
            }
        }

        return result
    }

    // MARK: - Pipeline Execution

    public func process(
        targetImage: ImageBuffer,
        settings: FrameColorizerSettings,
        modelCache: ModelCache,
        ortBridge: ORTBridge
    ) async throws -> ImageBuffer {
        guard let metadata = ModelCatalog.model(for: settings.model) else {
            throw ORTBridgeError.sessionCreationFailed("Unknown frame colorizer model: \(settings.model)")
        }

        let modelURL = try await modelCache.ensureModelDownloaded(metadata)
        let modelSize = settings.size == "512x512" ? 512 : 256

        let origW = targetImage.width
        let origH = targetImage.height
        guard origW > 0 && origH > 0 else {
            throw ORTBridgeError.inferenceFailed("Invalid target image dimensions: \(origW)x\(origH)")
        }

        // 1. Direct memory-bounded downscale to model resolution and upstream grayscale Lab preprocess
        let planeSize = modelSize * modelSize
        var inputTensor = [Float](repeating: 0, count: 3 * planeSize)

        for gy in 0..<modelSize {
            let sy = min(max(Int(Float(gy) * Float(origH) / Float(modelSize)), 0), origH - 1)
            let sRow = sy * origW * 4
            let gRow = gy * modelSize
            for gx in 0..<modelSize {
                let sx = min(max(Int(Float(gx) * Float(origW) / Float(modelSize)), 0), origW - 1)
                let sIdx = sRow + sx * 4
                let r = Float(targetImage.data[sIdx + 0])
                let g = Float(targetImage.data[sIdx + 1])
                let b = Float(targetImage.data[sIdx + 2])
                let grayU8 = UInt8(min(max(0.299 * r + 0.587 * g + 0.114 * b, 0.0), 255.0))
                let grayNorm = Float(grayU8) / 255.0

                // Upstream DDColor grayscale Lab preprocess: (gray -> Lab L -> zero ab -> sRGB)
                let (lVal, _, _) = FrameColorizerProcessor.sRGBToLab(r: grayNorm, g: grayNorm, b: grayNorm)
                let (recR, recG, recB) = FrameColorizerProcessor.labToSRGB(L: lVal, a: 0.0, b: 0.0)

                let pIdx = gRow + gx
                inputTensor[0 * planeSize + pIdx] = recR
                inputTensor[1 * planeSize + pIdx] = recG
                inputTensor[2 * planeSize + pIdx] = recB
            }
        }

        let inputs = ["input": TensorBuffer(floatData: inputTensor, shape: [1, 3, modelSize, modelSize])]
        let outputs = try await ortBridge.run(modelPath: modelURL.path, inputs: inputs)
        guard let outputTensor = outputs["output"] ?? (outputs.count == 1 ? outputs.values.first : nil),
              let colorTensor = outputTensor.floatData else {
            throw ORTBridgeError.inferenceFailed("Colorizer returned empty tensor")
        }

        // 2. Recombine predicted 2-channel chroma (a, b) with original target luminance
        let result = try Self.recombine(
            targetImage: targetImage,
            colorTensor: colorTensor,
            modelSize: modelSize
        )

        // 3. Blend with original image based on blend factor (0...100)
        if settings.blend < 100 {
            let blendFactor = Float(settings.blend) / 100.0
            let finalImage = targetImage.clone()
            finalImage.blend(with: result, alpha: blendFactor)
            return finalImage
        } else {
            return result
        }
    }
}
