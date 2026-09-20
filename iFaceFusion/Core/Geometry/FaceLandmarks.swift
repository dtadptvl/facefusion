import Foundation
import CoreGraphics
import simd

/// 5-point landmark set representing eyes, nose tip, and mouth corners.
public struct FaceLandmark5: Equatable, Sendable {
    public var points: [SIMD2<Float>] // Count: 5. Indices: 0: left eye, 1: right eye, 2: nose, 3: left mouth, 4: right mouth

    public init(points: [SIMD2<Float>]) {
        precondition(points.count == 5, "FaceLandmark5 must contain exactly 5 points")
        self.points = points
    }

    public var leftEye: SIMD2<Float> { points[0] }
    public var rightEye: SIMD2<Float> { points[1] }
    public var nose: SIMD2<Float> { points[2] }
    public var leftMouth: SIMD2<Float> { points[3] }
    public var rightMouth: SIMD2<Float> { points[4] }

    public func scaled(by scale: Float) -> FaceLandmark5 {
        // Scaling is performed relative to the nose point (index 2), matching upstream face_helper.scale_face_landmark_5
        let center = points[2]
        let scaledPoints = points.map { pt in
            (pt - center) * scale + center
        }
        return FaceLandmark5(points: scaledPoints)
    }
}

/// 68-point facial landmark set following standard Dlib/Multi-PIE landmark layout.
public struct FaceLandmark68: Equatable, Sendable {
    public var points: [SIMD2<Float>] // Count: 68

    public init(points: [SIMD2<Float>]) {
        precondition(points.count == 68, "FaceLandmark68 must contain exactly 68 points")
        self.points = points
    }

    public subscript(index: Int) -> SIMD2<Float> {
        get { points[index] }
        set { points[index] = newValue }
    }

    /// Converts 68-point landmarks to 5-point landmarks using upstream FaceFusion landmark conversion logic.
    public func toLandmark5() -> FaceLandmark5 {
        // Left eye center: mean of 36..<42
        var leftEyeSum = SIMD2<Float>(0, 0)
        for i in 36..<42 {
            leftEyeSum += points[i]
        }
        let leftEye = leftEyeSum / 6.0

        // Right eye center: mean of 42..<48
        var rightEyeSum = SIMD2<Float>(0, 0)
        for i in 42..<48 {
            rightEyeSum += points[i]
        }
        let rightEye = rightEyeSum / 6.0

        let nose = points[30]
        let leftMouth = points[48]
        let rightMouth = points[54]

        return FaceLandmark5(points: [leftEye, rightEye, nose, leftMouth, rightMouth])
    }

    /// Computes distance ratio between two vertical landmark indices and two horizontal landmark indices.
    /// Used for LivePortrait eye and mouth aspect ratios.
    public func distanceRatio(top: Int, bottom: Int, left: Int, right: Int) -> Float {
        let vDiff = points[top] - points[bottom]
        let hDiff = points[left] - points[right]
        let vDist = simd_length(vDiff)
        let hDist = simd_length(hDiff)
        return vDist / (hDist + 1e-6)
    }

    /// Bounding box derived from minimum and maximum coordinates.
    public var boundingBox: CGRect {
        var minX = Float.greatestFiniteMagnitude
        var minY = Float.greatestFiniteMagnitude
        var maxX = -Float.greatestFiniteMagnitude
        var maxY = -Float.greatestFiniteMagnitude

        for p in points {
            minX = min(minX, p.x)
            minY = min(minY, p.y)
            maxX = max(maxX, p.x)
            maxY = max(maxY, p.y)
        }

        return CGRect(x: CGFloat(minX), y: CGFloat(minY), width: CGFloat(maxX - minX), height: CGFloat(maxY - minY))
    }

    /// Estimates 2D in-plane face angle (0, 90, 180, 270 degrees) from jaw line points 0 and 16.
    public func estimateAngle() -> Int {
        let p0 = points[0]
        let p16 = points[16]
        let dy = p16.y - p0.y
        let dx = p16.x - p0.x
        var theta = atan2(dy, dx) * 180.0 / Float.pi
        if theta < 0 {
            theta += 360.0
        }
        let candidateAngles: [Float] = [0.0, 90.0, 180.0, 270.0, 360.0]
        var minDiff = Float.greatestFiniteMagnitude
        var bestAngle: Int = 0
        for cand in candidateAngles {
            let diff = abs(cand - theta)
            if diff < minDiff {
                minDiff = diff
                bestAngle = Int(cand.truncatingRemainder(dividingBy: 360.0))
            }
        }
        return bestAngle
    }
}

/// Complete face detection record holding landmarks, bounding box, and estimated angle.
public struct FaceTarget: Equatable, Sendable {
    public var boundingBox: CGRect
    public var landmark5: FaceLandmark5
    public var landmark68: FaceLandmark68
    public var angle: Int
    public var score: Float
    public var age: Float?

    public init(boundingBox: CGRect, landmark5: FaceLandmark5, landmark68: FaceLandmark68, angle: Int = 0, score: Float = 1.0, age: Float? = nil) {
        self.boundingBox = boundingBox
        self.landmark5 = landmark5
        self.landmark68 = landmark68
        self.angle = angle
        self.score = score
        self.age = age
    }

    /// Rescales all coordinates when the underlying image has been upscaled/downscaled.
    public func rescaled(scaleX: Float, scaleY: Float) -> FaceTarget {
        let newBbox = CGRect(
            x: boundingBox.origin.x * CGFloat(scaleX),
            y: boundingBox.origin.y * CGFloat(scaleY),
            width: boundingBox.width * CGFloat(scaleX),
            height: boundingBox.height * CGFloat(scaleY)
        )
        let newL5 = FaceLandmark5(points: landmark5.points.map { SIMD2<Float>($0.x * scaleX, $0.y * scaleY) })
        let newL68 = FaceLandmark68(points: landmark68.points.map { SIMD2<Float>($0.x * scaleX, $0.y * scaleY) })
        return FaceTarget(
            boundingBox: newBbox,
            landmark5: newL5,
            landmark68: newL68,
            angle: angle,
            score: score,
            age: age
        )
    }

    public func rescaled(by scale: Float) -> FaceTarget {
        rescaled(scaleX: scale, scaleY: scale)
    }
}
