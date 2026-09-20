import Foundation
import CoreGraphics
import simd

/// 2x3 Affine transformation matrix [ m00 m01 m02 ]
///                                 [ m10 m11 m12 ]
public struct AffineMatrix2x3: Equatable, Sendable {
    public var m00: Float
    public var m01: Float
    public var m02: Float
    public var m10: Float
    public var m11: Float
    public var m12: Float

    public init(m00: Float, m01: Float, m02: Float, m10: Float, m11: Float, m12: Float) {
        self.m00 = m00
        self.m01 = m01
        self.m02 = m02
        self.m10 = m10
        self.m11 = m11
        self.m12 = m12
    }

    public static let identity = AffineMatrix2x3(m00: 1, m01: 0, m02: 0, m10: 0, m11: 1, m12: 0)

    /// Inverts the affine transform matrix using standard 2D affine inversion.
    public func inverted() -> AffineMatrix2x3? {
        let det = m00 * m11 - m01 * m10
        if abs(det) < 1e-10 { return nil }
        let invDet = 1.0 / det
        let i00 = m11 * invDet
        let i01 = -m01 * invDet
        let i10 = -m10 * invDet
        let i11 = m00 * invDet
        let i02 = (m01 * m12 - m11 * m02) * invDet
        let i12 = (m10 * m02 - m00 * m12) * invDet
        return AffineMatrix2x3(m00: i00, m01: i01, m02: i02, m10: i10, m11: i11, m12: i12)
    }

    /// Transforms a single 2D point.
    public func transformPoint(_ pt: SIMD2<Float>) -> SIMD2<Float> {
        let x = m00 * pt.x + m01 * pt.y + m02
        let y = m10 * pt.x + m11 * pt.y + m12
        return SIMD2<Float>(x, y)
    }

    /// Composes this matrix with another (this * next).
    public func concatenated(with next: AffineMatrix2x3) -> AffineMatrix2x3 {
        // [next.m00 next.m01 next.m02] * [m00 m01 m02]
        // [next.m10 next.m11 next.m12]   [m10 m11 m12]
        // [    0        0        1   ]   [ 0   0   1 ]
        let r00 = next.m00 * m00 + next.m01 * m10
        let r01 = next.m00 * m01 + next.m01 * m11
        let r02 = next.m00 * m02 + next.m01 * m12 + next.m02

        let r10 = next.m10 * m00 + next.m11 * m10
        let r11 = next.m10 * m01 + next.m11 * m11
        let r12 = next.m10 * m02 + next.m11 * m12 + next.m12

        return AffineMatrix2x3(m00: r00, m01: r01, m02: r02, m10: r10, m11: r11, m12: r12)
    }

    /// Converts to CoreGraphics CGAffineTransform.
    public var cgAffineTransform: CGAffineTransform {
        CGAffineTransform(a: CGFloat(m00), b: CGFloat(m10), c: CGFloat(m01), d: CGFloat(m11), tx: CGFloat(m02), ty: CGFloat(m12))
    }
}

/// Geometry helper routines for face cropping, similarity transform estimation, and paste-back.
public enum ImageGeometry {

    /// Estimates 2D partial affine similarity transform (scale, rotation, translation) from source to destination points.
    /// Matches OpenCV `cv2.estimateAffinePartial2D` and FaceFusion `estimate_matrix_by_face_landmark_5`.
    public static func estimateSimilarityMatrix(src: [SIMD2<Float>], dst: [SIMD2<Float>]) -> AffineMatrix2x3 {
        precondition(src.count == dst.count && src.count >= 2, "Source and destination point counts must match and be at least 2")
        let n = Float(src.count)
        let mx = src.reduce(0) { $0 + $1.x } / n
        let my = src.reduce(0) { $0 + $1.y } / n
        let mu = dst.reduce(0) { $0 + $1.x } / n
        let mv = dst.reduce(0) { $0 + $1.y } / n

        var denom: Float = 0
        var numA: Float = 0
        var numB: Float = 0

        for i in 0..<src.count {
            let xc = src[i].x - mx
            let yc = src[i].y - my
            let uc = dst[i].x - mu
            let vc = dst[i].y - mv

            denom += xc * xc + yc * yc
            numA += xc * uc + yc * vc
            numB += xc * vc - yc * uc
        }

        let a = denom > 1e-8 ? numA / denom : 1.0
        let b = denom > 1e-8 ? numB / denom : 0.0
        let tx = mu - (a * mx - b * my)
        let ty = mv - (b * mx + a * my)

        return AffineMatrix2x3(m00: a, m01: -b, m02: tx, m10: b, m11: a, m12: ty)
    }

    /// Calculates paste area bounding box and localized inverse transform matrix matching calculate_paste_area.
    public static func calculatePasteArea(
        targetWidth: Int,
        targetHeight: Int,
        cropWidth: Int,
        cropHeight: Int,
        affineMatrix: AffineMatrix2x3
    ) -> (pasteBox: (x1: Int, y1: Int, x2: Int, y2: Int), pasteMatrix: AffineMatrix2x3)? {
        guard let invMatrix = affineMatrix.inverted() else { return nil }

        let corners: [SIMD2<Float>] = [
            SIMD2<Float>(0, 0),
            SIMD2<Float>(Float(cropWidth), 0),
            SIMD2<Float>(Float(cropWidth), Float(cropHeight)),
            SIMD2<Float>(0, Float(cropHeight))
        ]

        var minX = Float.greatestFiniteMagnitude
        var minY = Float.greatestFiniteMagnitude
        var maxX = -Float.greatestFiniteMagnitude
        var maxY = -Float.greatestFiniteMagnitude

        for c in corners {
            let p = invMatrix.transformPoint(c)
            minX = min(minX, p.x)
            minY = min(minY, p.y)
            maxX = max(maxX, p.x)
            maxY = max(maxY, p.y)
        }

        let x1 = max(0, min(targetWidth, Int(floor(minX))))
        let y1 = max(0, min(targetHeight, Int(floor(minY))))
        let x2 = max(0, min(targetWidth, Int(ceil(maxX))))
        let y2 = max(0, min(targetHeight, Int(ceil(maxY))))

        // Sampling maps local destination pixels back into the aligned crop.
        var pasteMatrix = affineMatrix
        pasteMatrix.m02 += affineMatrix.m00 * Float(x1) + affineMatrix.m01 * Float(y1)
        pasteMatrix.m12 += affineMatrix.m10 * Float(x1) + affineMatrix.m11 * Float(y1)

        return ((x1, y1, x2, y2), pasteMatrix)
    }
}
