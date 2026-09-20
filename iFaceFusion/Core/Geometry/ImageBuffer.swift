import Foundation
import CoreGraphics
import UIKit
import Accelerate

/// In-memory RGBA byte buffer with image geometry manipulation, color space conversion, and ONNX tensor staging.
// ponytail: ImageBuffer is isolated in sequential engine passes; locks omitted to prevent CPU loop overhead. Upgrade to actor/Sendable value type if concurrent sharing across threads is ever required.
public final class ImageBuffer: @unchecked Sendable {
    public let width: Int
    public let height: Int
    public var data: [UInt8] // Length: width * height * 4, format: RGBA8888

    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
        self.data = [UInt8](repeating: 0, count: width * height * 4)
    }

    public init(width: Int, height: Int, data: [UInt8]) {
        precondition(data.count == width * height * 4, "Data count must match width * height * 4")
        self.width = width
        self.height = height
        self.data = data
    }

    /// Initializes from a UIImage, normalizing orientation so pixel memory is upright.
    public convenience init?(image: UIImage) {
        guard let cgImage = image.cgImage else { return nil }
        // If image has non-up orientation, redraw into upright context
        if image.imageOrientation != .up {
            let width = cgImage.width
            let height = cgImage.height
            var bounds = CGRect(x: 0, y: 0, width: width, height: height)
            switch image.imageOrientation {
            case .left, .right, .leftMirrored, .rightMirrored:
                bounds = CGRect(x: 0, y: 0, width: height, height: width)
            default:
                break
            }
            // Use scale 1.0 with exact pixel bounds to prevent dimension distortion on non-square/scaled images
            UIGraphicsBeginImageContextWithOptions(bounds.size, false, 1.0)
            image.draw(in: bounds)
            let uprightImage = UIGraphicsGetImageFromCurrentImageContext()
            UIGraphicsEndImageContext()
            guard let uprightCg = uprightImage?.cgImage else { return nil }
            self.init(cgImage: uprightCg)
        } else {
            self.init(cgImage: cgImage)
        }
    }

    /// Initializes from a CGImage, safely drawing within a bounded pointer scope and unpremultiplying to straight RGBA.
    public convenience init?(cgImage: CGImage) {
        let width = cgImage.width
        let height = cgImage.height
        var data = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)

        let drawn = data.withUnsafeMutableBytes { ptr -> Bool in
            guard let baseAddress = ptr.baseAddress else { return false }
            guard let context = CGContext(
                data: baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo.rawValue
            ) else { return false }

            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }

        guard drawn else { return nil }

        // Unpremultiply imported pixels to guarantee consistent straight RGBA internally
        for i in 0..<(width * height) {
            let idx = i * 4
            let a = UInt32(data[idx + 3])
            if a == 0 {
                data[idx + 0] = 0
                data[idx + 1] = 0
                data[idx + 2] = 0
            } else if a < 255 {
                let r = (UInt32(data[idx + 0]) * 255 + a / 2) / a
                let g = (UInt32(data[idx + 1]) * 255 + a / 2) / a
                let b = (UInt32(data[idx + 2]) * 255 + a / 2) / a
                data[idx + 0] = UInt8(min(r, 255))
                data[idx + 1] = UInt8(min(g, 255))
                data[idx + 2] = UInt8(min(b, 255))
            }
        }

        self.init(width: width, height: height, data: data)
    }

    /// Renders the straight RGBA pixel buffer to a CGImage, using straight .last when supported or a premultiplied copy.
    public func toCGImage() -> CGImage? {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: Data(data) as CFData) else { return nil }

        // Attempt export as straight RGBA (.last)
        let straightInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
        if let straightCG = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: straightInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        ) {
            return straightCG
        }

        // ponytail: Fallback premultiplied copy used if CGImage provider rejects straight .last on legacy/hardware configurations. Upgrade to direct vImage unpremultiplied surface if pipeline latency demands zero-copy export.
        var premulData = [UInt8](repeating: 0, count: width * height * 4)
        for i in 0..<(width * height) {
            let idx = i * 4
            let a = UInt32(data[idx + 3])
            premulData[idx + 0] = UInt8((UInt32(data[idx + 0]) * a + 127) / 255)
            premulData[idx + 1] = UInt8((UInt32(data[idx + 1]) * a + 127) / 255)
            premulData[idx + 2] = UInt8((UInt32(data[idx + 2]) * a + 127) / 255)
            premulData[idx + 3] = data[idx + 3]
        }

        guard let premulProvider = CGDataProvider(data: Data(premulData) as CFData) else { return nil }
        let premulInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: premulInfo,
            provider: premulProvider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
    }

    /// Renders to UIImage.
    public func toUIImage() -> UIImage? {
        guard let cg = toCGImage() else { return nil }
        return UIImage(cgImage: cg)
    }

    /// Creates a deep copy of this image buffer.
    public func clone() -> ImageBuffer {
        ImageBuffer(width: width, height: height, data: data)
    }

    // MARK: - Warping with replicate border

    /// Warps this image using an affine matrix with bilinear sampling and border replication.
    /// Matches OpenCV `cv2.warpAffine(..., borderMode=cv2.BORDER_REPLICATE)`.
    public func warpAffine(
        matrix: AffineMatrix2x3,
        cropWidth: Int,
        cropHeight: Int
    ) -> ImageBuffer {
        guard let inv = matrix.inverted() else {
            return ImageBuffer(width: cropWidth, height: cropHeight)
        }

        var result = [UInt8](repeating: 0, count: cropWidth * cropHeight * 4)
        let srcW = Float(width)
        let srcH = Float(height)

        for cy in 0..<cropHeight {
            let rowOffset = cy * cropWidth * 4
            for cx in 0..<cropWidth {
                let p = inv.transformPoint(SIMD2<Float>(Float(cx) + 0.5, Float(cy) + 0.5))
                let sx = min(max(p.x - 0.5, 0), Float(width - 1))
                let sy = min(max(p.y - 0.5, 0), Float(height - 1))

                let x0 = max(0, min(width - 1, Int(floor(sx))))
                let y0 = max(0, min(height - 1, Int(floor(sy))))
                let x1 = max(0, min(width - 1, x0 + 1))
                let y1 = max(0, min(height - 1, y0 + 1))

                let fx = sx - Float(x0)
                let fy = sy - Float(y0)

                let w00 = (1.0 - fx) * (1.0 - fy)
                let w10 = fx * (1.0 - fy)
                let w01 = (1.0 - fx) * fy
                let w11 = fx * fy

                let idx00 = (y0 * width + x0) * 4
                let idx10 = (y0 * width + x1) * 4
                let idx01 = (y1 * width + x0) * 4
                let idx11 = (y1 * width + x1) * 4

                let dstIdx = rowOffset + cx * 4
                for c in 0..<4 {
                    let v = w00 * Float(data[idx00 + c]) +
                            w10 * Float(data[idx10 + c]) +
                            w01 * Float(data[idx01 + c]) +
                            w11 * Float(data[idx11 + c])
                    result[dstIdx + c] = UInt8(min(max(v, 0), 255))
                }
            }
        }

        return ImageBuffer(width: cropWidth, height: cropHeight, data: result)
    }

    /// Blends warped crop buffer and feathered mask back into target buffer matching `paste_back`.
    public func pasteBack(crop: ImageBuffer, mask: FaceMask, matrix: AffineMatrix2x3) {
        guard let (box, pasteMatrix) = ImageGeometry.calculatePasteArea(
            targetWidth: width,
            targetHeight: height,
            cropWidth: crop.width,
            cropHeight: crop.height,
            affineMatrix: matrix
        ) else { return }

        let pasteW = box.x2 - box.x1
        let pasteH = box.y2 - box.y1
        guard pasteW > 0 && pasteH > 0 else { return }

        for py in 0..<pasteH {
            let targetY = box.y1 + py
            let targetRowOffset = targetY * width * 4

            for px in 0..<pasteW {
                let targetX = box.x1 + px
                let targetIdx = targetRowOffset + targetX * 4

                // Sample crop coordinate from paste area
                let cropCoord = pasteMatrix.transformPoint(SIMD2<Float>(Float(px) + 0.5, Float(py) + 0.5))
                let cx = cropCoord.x - 0.5
                let cy = cropCoord.y - 0.5

                let cx0 = max(0, min(crop.width - 1, Int(floor(cx))))
                let cy0 = max(0, min(crop.height - 1, Int(floor(cy))))
                let cx1 = max(0, min(crop.width - 1, cx0 + 1))
                let cy1 = max(0, min(crop.height - 1, cy0 + 1))

                let fx = cx - Float(cx0)
                let fy = cy - Float(cy0)

                let w00 = (1.0 - fx) * (1.0 - fy)
                let w10 = fx * (1.0 - fy)
                let w01 = (1.0 - fx) * fy
                let w11 = fx * fy

                let m00 = mask[cx0, cy0]
                let m10 = mask[cx1, cy0]
                let m01 = mask[cx0, cy1]
                let m11 = mask[cx1, cy1]
                let alpha = min(max(w00 * m00 + w10 * m10 + w01 * m01 + w11 * m11, 0.0), 1.0)

                if alpha <= 0.001 { continue }

                let cIdx00 = (cy0 * crop.width + cx0) * 4
                let cIdx10 = (cy0 * crop.width + cx1) * 4
                let cIdx01 = (cy1 * crop.width + cx0) * 4
                let cIdx11 = (cy1 * crop.width + cx1) * 4

                for c in 0..<3 { // RGB
                    let cropColor = w00 * Float(crop.data[cIdx00 + c]) +
                                    w10 * Float(crop.data[cIdx10 + c]) +
                                    w01 * Float(crop.data[cIdx01 + c]) +
                                    w11 * Float(crop.data[cIdx11 + c])
                    let origColor = Float(data[targetIdx + c])
                    let blended = origColor * (1.0 - alpha) + cropColor * alpha
                    data[targetIdx + c] = UInt8(min(max(blended, 0), 255))
                }
            }
        }
    }

    // MARK: - Tensor conversions

    /// Converts RGBA pixels to planar NCHW float tensor with channel order (RGB or BGR) and normalization.
    public func toFloatTensorNCHW(
        mean: [Float] = [0, 0, 0],
        std: [Float] = [1, 1, 1],
        isBGR: Bool = false
    ) -> [Float] {
        let channelCount = 3
        let planeSize = width * height
        var tensor = [Float](repeating: 0, count: channelCount * planeSize)

        let ch0 = isBGR ? 2 : 0 // Red channel in input
        let ch1 = 1             // Green channel in input
        let ch2 = isBGR ? 0 : 2 // Blue channel in input

        for y in 0..<height {
            let rowOffset = y * width
            for x in 0..<width {
                let pixelIdx = (rowOffset + x) * 4
                let planeIdx = rowOffset + x

                let v0 = (Float(data[pixelIdx + ch0]) / 255.0 - mean[0]) / std[0]
                let v1 = (Float(data[pixelIdx + ch1]) / 255.0 - mean[1]) / std[1]
                let v2 = (Float(data[pixelIdx + ch2]) / 255.0 - mean[2]) / std[2]

                tensor[0 * planeSize + planeIdx] = v0
                tensor[1 * planeSize + planeIdx] = v1
                tensor[2 * planeSize + planeIdx] = v2
            }
        }
        return tensor
    }

    /// Converts RGBA pixels to NHWC float tensor (used by DeepSwapper: [1, H, W, 3] BGR in [0, 1]).
    public func toFloatTensorNHWC(isBGR: Bool = true) -> [Float] {
        var tensor = [Float](repeating: 0, count: width * height * 3)
        let ch0 = isBGR ? 2 : 0
        let ch1 = 1
        let ch2 = isBGR ? 0 : 2

        for i in 0..<(width * height) {
            let inIdx = i * 4
            let outIdx = i * 3
            tensor[outIdx + 0] = Float(data[inIdx + ch0]) / 255.0
            tensor[outIdx + 1] = Float(data[inIdx + ch1]) / 255.0
            tensor[outIdx + 2] = Float(data[inIdx + ch2]) / 255.0
        }
        return tensor
    }

    /// Fills buffer from NCHW float tensor output.
    public static func fromFloatTensorNCHW(
        tensor: [Float],
        width: Int,
        height: Int,
        mean: [Float] = [0, 0, 0],
        std: [Float] = [1, 1, 1],
        isBGR: Bool = false
    ) -> ImageBuffer {
        var buffer = ImageBuffer(width: width, height: height)
        let planeSize = width * height
        let ch0 = isBGR ? 2 : 0
        let ch1 = 1
        let ch2 = isBGR ? 0 : 2

        for y in 0..<height {
            let rowOffset = y * width
            for x in 0..<width {
                let planeIdx = rowOffset + x
                let pixelIdx = (rowOffset + x) * 4

                let v0 = (tensor[0 * planeSize + planeIdx] * std[0] + mean[0]) * 255.0
                let v1 = (tensor[1 * planeSize + planeIdx] * std[1] + mean[1]) * 255.0
                let v2 = (tensor[2 * planeSize + planeIdx] * std[2] + mean[2]) * 255.0

                buffer.data[pixelIdx + ch0] = UInt8(min(max(v0, 0), 255))
                buffer.data[pixelIdx + ch1] = UInt8(min(max(v1, 0), 255))
                buffer.data[pixelIdx + ch2] = UInt8(min(max(v2, 0), 255))
                buffer.data[pixelIdx + 3] = 255
            }
        }
        return buffer
    }

    /// Fills buffer from NHWC float tensor output (used by DeepSwapper).
    public static func fromFloatTensorNHWC(
        tensor: [Float],
        width: Int,
        height: Int,
        isBGR: Bool = true
    ) -> ImageBuffer {
        var buffer = ImageBuffer(width: width, height: height)
        let ch0 = isBGR ? 2 : 0
        let ch1 = 1
        let ch2 = isBGR ? 0 : 2

        for i in 0..<(width * height) {
            let inIdx = i * 3
            let outIdx = i * 4
            buffer.data[outIdx + ch0] = UInt8(min(max(tensor[inIdx + 0] * 255.0, 0), 255))
            buffer.data[outIdx + ch1] = UInt8(min(max(tensor[inIdx + 1] * 255.0, 0), 255))
            buffer.data[outIdx + ch2] = UInt8(min(max(tensor[inIdx + 2] * 255.0, 0), 255))
            buffer.data[outIdx + 3] = 255
        }
        return buffer
    }

    /// Blends this image with another using opacity factor in [0, 1].
    public func blend(with other: ImageBuffer, alpha: Float) {
        guard width == other.width && height == other.height else { return }
        let a = min(max(alpha, 0), 1)
        let invA = 1.0 - a
        for i in 0..<(width * height * 4) {
            let v = Float(data[i]) * invA + Float(other.data[i]) * a
            data[i] = UInt8(min(max(v, 0), 255))
        }
    }
}
