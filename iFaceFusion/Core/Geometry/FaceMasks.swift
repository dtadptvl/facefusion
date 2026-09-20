import Foundation
import CoreGraphics
import simd

/// Types of face masks supported for edge feathering, occlusion handling, and facial region filtering.
public enum FaceMaskType: String, CaseIterable, Sendable {
    case box
    case occlusion
    case area
    case region
}

public enum FaceMaskArea: String, CaseIterable, Sendable {
    case upperFace = "upper-face"
    case lowerFace = "lower-face"
    case mouth = "mouth"

    public var landmarkIndices: [Int] {
        switch self {
        case .upperFace:
            return [0, 1, 2, 31, 32, 33, 34, 35, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]
        case .lowerFace:
            return [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 35, 34, 33, 32, 31]
        case .mouth:
            return Array(48...67)
        }
    }
}

public enum FaceMaskRegion: String, CaseIterable, Sendable {
    case skin
    case leftEyebrow = "left-eyebrow"
    case rightEyebrow = "right-eyebrow"
    case leftEye = "left-eye"
    case rightEye = "right-eye"
    case glasses
    case nose
    case mouth
    case upperLip = "upper-lip"
    case lowerLip = "lower-lip"

    public var classId: Int {
        switch self {
        case .skin: return 1
        case .leftEyebrow: return 2
        case .rightEyebrow: return 3
        case .leftEye: return 4
        case .rightEye: return 5
        case .glasses: return 6
        case .nose: return 10
        case .mouth: return 11
        case .upperLip: return 12
        case .lowerLip: return 13
        }
    }
}

/// Padding margins (top, right, bottom, left) in percentage of dimension (0...100).
public struct FaceMaskPadding: Equatable, Sendable {
    public var top: Int
    public var right: Int
    public var bottom: Int
    public var left: Int

    public init(top: Int = 0, right: Int = 0, bottom: Int = 0, left: Int = 0) {
        self.top = top
        self.right = right
        self.bottom = bottom
        self.left = left
    }

    public static let zero = FaceMaskPadding(top: 0, right: 0, bottom: 0, left: 0)
}

/// 2D Float mask representation and operations (blurring, clipping, combining).
public struct FaceMask: Sendable {
    public let width: Int
    public let height: Int
    public var values: [Float] // Count: width * height, row-major in [0, 1]

    public init(width: Int, height: Int, initialValue: Float = 1.0) {
        self.width = width
        self.height = height
        self.values = [Float](repeating: initialValue, count: width * height)
    }

    public init(width: Int, height: Int, values: [Float]) {
        precondition(values.count == width * height, "Mask values count must equal width * height")
        self.width = width
        self.height = height
        self.values = values
    }

    public subscript(x: Int, y: Int) -> Float {
        get { values[y * width + x] }
        set { values[y * width + x] = newValue }
    }

    /// Creates box mask with specified edge padding percentages and blur radius.
    public static func createBoxMask(width: Int, height: Int, blur: Float, padding: FaceMaskPadding) -> FaceMask {
        var mask = FaceMask(width: width, height: height, initialValue: 1.0)
        let blurAmount = Int(Float(width) * 0.5 * blur)
        let blurArea = max(blurAmount / 2, 1)

        let topBorder = max(blurArea, Int(Float(height) * Float(padding.top) / 100.0))
        let bottomBorder = max(blurArea, Int(Float(height) * Float(padding.bottom) / 100.0))
        let leftBorder = max(blurArea, Int(Float(width) * Float(padding.left) / 100.0))
        let rightBorder = max(blurArea, Int(Float(width) * Float(padding.right) / 100.0))

        for y in 0..<height {
            for x in 0..<width {
                if y < topBorder || y >= (height - bottomBorder) || x < leftBorder || x >= (width - rightBorder) {
                    mask[x, y] = 0.0
                }
            }
        }

        if blurAmount > 0 {
            let sigma = Float(blurAmount) * 0.25
            mask = mask.gaussianBlurred(sigma: sigma)
        }

        return mask
    }

    /// Creates area mask from warped 68-point landmarks convex hull polygon with feathering.
    public static func createAreaMask(width: Int, height: Int, landmarks68InCrop: [SIMD2<Float>], areas: Set<FaceMaskArea>) -> FaceMask {
        var mask = FaceMask(width: width, height: height, initialValue: 0.0)
        var selectedPoints: [SIMD2<Float>] = []

        for area in areas {
            for idx in area.landmarkIndices {
                if idx < landmarks68InCrop.count {
                    selectedPoints.append(landmarks68InCrop[idx])
                }
            }
        }

        guard selectedPoints.count >= 3 else {
            return FaceMask(width: width, height: height, initialValue: 1.0)
        }

        let hull = convexHull(points: selectedPoints)
        fillConvexPolygon(mask: &mask, polygon: hull, value: 1.0)

        // Feathering matching upstream: (gaussian_blur(mask, sigma=5).clip(0.5, 1) - 0.5) * 2
        let blurred = mask.gaussianBlurred(sigma: 5.0)
        var feathered = FaceMask(width: width, height: height, initialValue: 0.0)
        for i in 0..<blurred.values.count {
            let v = min(max(blurred.values[i], 0.5), 1.0)
            feathered.values[i] = (v - 0.5) * 2.0
        }
        return feathered
    }

    /// Combines multiple masks using element-wise minimum.
    public static func combineMinimum(_ masks: [FaceMask]) -> FaceMask {
        guard let first = masks.first else {
            return FaceMask(width: 1, height: 1, initialValue: 1.0)
        }
        var combined = first
        for m in masks.dropFirst() {
            guard m.width == combined.width && m.height == combined.height else { continue }
            for i in 0..<combined.values.count {
                combined.values[i] = min(combined.values[i], m.values[i])
            }
        }
        for i in 0..<combined.values.count {
            combined.values[i] = min(max(combined.values[i], 0.0), 1.0)
        }
        return combined
    }

    /// High-performance 2D separable Gaussian blur.
    public func gaussianBlurred(sigma: Float) -> FaceMask {
        guard sigma > 0.1 else { return self }
        let radius = max(1, Int(ceil(sigma * 3.0)))
        var kernel = [Float](repeating: 0, count: 2 * radius + 1)
        var sum: Float = 0
        let twoSigmaSq = 2.0 * sigma * sigma
        for i in -radius...radius {
            let val = exp(-Float(i * i) / twoSigmaSq)
            kernel[i + radius] = val
            sum += val
        }
        for i in 0..<kernel.count {
            kernel[i] /= sum
        }

        // Horizontal pass
        var temp = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            let rowOffset = y * width
            for x in 0..<width {
                var acc: Float = 0
                for k in -radius...radius {
                    let sampleX = min(max(x + k, 0), width - 1)
                    acc += values[rowOffset + sampleX] * kernel[k + radius]
                }
                temp[rowOffset + x] = acc
            }
        }

        // Vertical pass
        var result = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                var acc: Float = 0
                for k in -radius...radius {
                    let sampleY = min(max(y + k, 0), height - 1)
                    acc += temp[sampleY * width + x] * kernel[k + radius]
                }
                result[y * width + x] = acc
            }
        }

        return FaceMask(width: width, height: height, values: result)
    }

    /// Resizes mask to target dimensions using bilinear interpolation.
    public func resized(toWidth targetW: Int, toHeight targetH: Int) -> FaceMask {
        guard targetW != width || targetH != height else { return self }
        var result = FaceMask(width: targetW, height: targetH, initialValue: 0.0)
        let scaleX = Float(width) / Float(targetW)
        let scaleY = Float(height) / Float(targetH)

        for ty in 0..<targetH {
            let srcY = (Float(ty) + 0.5) * scaleY - 0.5
            let y0 = max(0, min(width - 1, Int(floor(srcY))))
            let y1 = max(0, min(height - 1, y0 + 1))
            let fy = srcY - Float(y0)

            for tx in 0..<targetW {
                let srcX = (Float(tx) + 0.5) * scaleX - 0.5
                let x0 = max(0, min(width - 1, Int(floor(srcX))))
                let x1 = max(0, min(width - 1, x0 + 1))
                let fx = srcX - Float(x0)

                let v00 = self[x0, y0]
                let v10 = self[x1, y0]
                let v01 = self[x0, y1]
                let v11 = self[x1, y1]

                let val = (1.0 - fx) * (1.0 - fy) * v00 +
                          fx * (1.0 - fy) * v10 +
                          (1.0 - fx) * fy * v01 +
                          fx * fy * v11
                result[tx, ty] = min(max(val, 0.0), 1.0)
            }
        }
        return result
    }

    // MARK: - Polygon helpers

    private static func convexHull(points: [SIMD2<Float>]) -> [SIMD2<Float>] {
        guard points.count >= 3 else { return points }
        let sorted = points.sorted { a, b in
            if a.x != b.x { return a.x < b.x }
            return a.y < b.y
        }

        func crossProduct(_ o: SIMD2<Float>, _ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
            (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
        }

        var lower: [SIMD2<Float>] = []
        for p in sorted {
            while lower.count >= 2 && crossProduct(lower[lower.count - 2], lower[lower.count - 1], p) <= 0 {
                lower.removeLast()
            }
            lower.append(p)
        }

        var upper: [SIMD2<Float>] = []
        for p in sorted.reversed() {
            while upper.count >= 2 && crossProduct(upper[upper.count - 2], upper[upper.count - 1], p) <= 0 {
                upper.removeLast()
            }
            upper.append(p)
        }

        lower.removeLast()
        upper.removeLast()
        return lower + upper
    }

    private static func fillConvexPolygon(mask: inout FaceMask, polygon: [SIMD2<Float>], value: Float) {
        guard polygon.count >= 3 else { return }
        var minY = Float(mask.height)
        var maxY: Float = 0
        for p in polygon {
            minY = min(minY, p.y)
            maxY = max(maxY, p.y)
        }
        let startY = max(0, Int(floor(minY)))
        let endY = min(mask.height - 1, Int(ceil(maxY)))

        for y in startY...endY {
            let scanY = Float(y) + 0.5
            var nodeX: [Float] = []
            let count = polygon.count
            var j = count - 1
            for i in 0..<count {
                let pi = polygon[i]
                let pj = polygon[j]
                if (pi.y < scanY && pj.y >= scanY) || (pj.y < scanY && pi.y >= scanY) {
                    let x = pi.x + (scanY - pi.y) / (pj.y - pi.y) * (pj.x - pi.x)
                    nodeX.append(x)
                }
                j = i
            }
            nodeX.sort()
            var k = 0
            while k < nodeX.count - 1 {
                let xStart = max(0, Int(ceil(nodeX[k])))
                let xEnd = min(mask.width - 1, Int(floor(nodeX[k + 1])))
                if xStart <= xEnd {
                    for x in xStart...xEnd {
                        mask[x, y] = value
                    }
                }
                k += 2
            }
        }
    }
}
