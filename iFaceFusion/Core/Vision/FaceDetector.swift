import Foundation
import CoreGraphics
import Vision
import UIKit

public enum FaceDetectorError: LocalizedError, Equatable {
    case detectionFailed(String)
    case noFaceDetected
    case multipleFacesDetected(Int)

    public var errorDescription: String? {
        switch self {
        case .detectionFailed(let reason):
            return "Apple Vision face detection failed: \(reason)"
        case .noFaceDetected:
            return "No face was detected in the target image."
        case .multipleFacesDetected(let count):
            return "Expected exactly one face, but detected \(count) faces."
        }
    }
}

/// Raw point collections for face landmark regions extracted from Vision or synthetic fixtures.
public struct RawVisionLandmarkRegions: Sendable {
    public var faceContour: [SIMD2<Float>]?
    public var leftEyebrow: [SIMD2<Float>]?
    public var rightEyebrow: [SIMD2<Float>]?
    public var noseCrest: [SIMD2<Float>]?
    public var nose: [SIMD2<Float>]?
    public var leftEye: [SIMD2<Float>]?
    public var rightEye: [SIMD2<Float>]?
    public var outerLips: [SIMD2<Float>]?
    public var innerLips: [SIMD2<Float>]?

    public init(
        faceContour: [SIMD2<Float>]? = nil,
        leftEyebrow: [SIMD2<Float>]? = nil,
        rightEyebrow: [SIMD2<Float>]? = nil,
        noseCrest: [SIMD2<Float>]? = nil,
        nose: [SIMD2<Float>]? = nil,
        leftEye: [SIMD2<Float>]? = nil,
        rightEye: [SIMD2<Float>]? = nil,
        outerLips: [SIMD2<Float>]? = nil,
        innerLips: [SIMD2<Float>]? = nil
    ) {
        self.faceContour = faceContour
        self.leftEyebrow = leftEyebrow
        self.rightEyebrow = rightEyebrow
        self.noseCrest = noseCrest
        self.nose = nose
        self.leftEye = leftEye
        self.rightEye = rightEye
        self.outerLips = outerLips
        self.innerLips = innerLips
    }
}

/// Face detector leveraging Apple Vision Framework to detect exactly one primary face and its 68 facial landmarks.
public final class FaceDetector: Sendable {
    public static let shared = FaceDetector()

    public init() {}

    // MARK: - Face Count Validation

    /// Validates face detection count, strictly enforcing exactly one face.
    @discardableResult
    public static func validateFaceCount(_ count: Int) throws -> Int {
        if count == 0 {
            throw FaceDetectorError.noFaceDetected
        }
        if count > 1 {
            throw FaceDetectorError.multipleFacesDetected(count)
        }
        return 1
    }

    /// Selects the single detected face observation, strictly requiring count == 1 and rejecting 0 or >1.
    public static func selectSingleFace<T>(_ observations: [T]) throws -> T {
        try validateFaceCount(observations.count)
        return observations[0]
    }

    // MARK: - Contour Resampling & Normalization

    // ponytail: Linear interpolation along arc length is used for 68-landmark resampling. Upgrade to cubic spline / Catmull-Rom if sub-pixel curvature artifacts appear in high-res neural warps.
    /// Resamples an open polyline along its cumulative arc length to exactly `targetCount` points.
    public static func resampleOpenContour(_ points: [SIMD2<Float>], targetCount: Int) -> [SIMD2<Float>] {
        guard !points.isEmpty else { return [] }
        if points.count == 1 || targetCount == 1 {
            return [SIMD2<Float>](repeating: points[0], count: targetCount)
        }

        var cumDist = [Float](repeating: 0, count: points.count)
        for i in 1..<points.count {
            cumDist[i] = cumDist[i - 1] + simd_length(points[i] - points[i - 1])
        }
        let totalLength = cumDist.last!
        if totalLength <= 1e-6 {
            return [SIMD2<Float>](repeating: points[0], count: targetCount)
        }

        var resampled = [SIMD2<Float>]()
        resampled.reserveCapacity(targetCount)

        for i in 0..<targetCount {
            let targetDist = totalLength * Float(i) / Float(targetCount - 1)
            var j = 0
            while j < points.count - 2 && cumDist[j + 1] < targetDist {
                j += 1
            }
            let segLen = cumDist[j + 1] - cumDist[j]
            let fraction = segLen > 1e-6 ? (targetDist - cumDist[j]) / segLen : 0.0
            let pt = points[j] * (1.0 - fraction) + points[j + 1] * fraction
            resampled.append(pt)
        }

        return resampled
    }

    /// Splits a closed loop at its horizontal extrema (leftmost min X, rightmost max X) into upper and lower arcs,
    /// then resamples each arc to conform to Dlib clockwise landmark ordering.
    public static func resampleClosedLoop(
        points: [SIMD2<Float>],
        upperCount: Int,
        lowerCount: Int
    ) -> (upper: [SIMD2<Float>], lower: [SIMD2<Float>]) {
        let n = points.count
        guard n >= 3 else {
            let u = [SIMD2<Float>](repeating: points.first ?? .zero, count: upperCount)
            let l = [SIMD2<Float>](repeating: points.last ?? .zero, count: lowerCount)
            return (u, l)
        }

        // Find indices of minimum and maximum X coordinates
        var minIdx = 0
        var maxIdx = 0
        for i in 1..<n {
            if points[i].x < points[minIdx].x { minIdx = i }
            if points[i].x > points[maxIdx].x { maxIdx = i }
        }

        if minIdx == maxIdx {
            let u = [SIMD2<Float>](repeating: points[minIdx], count: upperCount)
            let l = [SIMD2<Float>](repeating: points[maxIdx], count: lowerCount)
            return (u, l)
        }

        // Traverse path 1 (increasing modulo n) from minIdx to maxIdx
        var path1 = [SIMD2<Float>]()
        var cur = minIdx
        while cur != maxIdx {
            path1.append(points[cur])
            cur = (cur + 1) % n
        }
        path1.append(points[maxIdx])

        // Traverse path 2 (decreasing modulo n) from minIdx to maxIdx
        var path2 = [SIMD2<Float>]()
        cur = minIdx
        while cur != maxIdx {
            path2.append(points[cur])
            cur = (cur - 1 + n) % n
        }
        path2.append(points[maxIdx])

        // In pixel space (Y down), smaller average Y corresponds to the upper contour
        let avgY1 = path1.reduce(0.0) { $0 + $1.y } / Float(path1.count)
        let avgY2 = path2.reduce(0.0) { $0 + $1.y } / Float(path2.count)

        let upperPath: [SIMD2<Float>]
        let lowerPath: [SIMD2<Float>]

        if avgY1 <= avgY2 {
            upperPath = path1
            lowerPath = Array(path2.reversed()) // from maxIdx back to minIdx
        } else {
            upperPath = path2
            lowerPath = Array(path1.reversed()) // from maxIdx back to minIdx
        }

        let resampledUpper = resampleOpenContour(upperPath, targetCount: upperCount)
        let resampledLower = resampleOpenContour(lowerPath, targetCount: lowerCount)

        return (resampledUpper, resampledLower)
    }

    // MARK: - Landmark Mapping

    // Explicit approximation: Vision's open/closed contours vary in point count across revisions.
    // Vision's noseCrest traces the nasal bone/bridge while the actual anatomical nose tip (Dlib index 30)
    // is the apex of the nose base region along the vertical symmetry axis.
    // Vision landmark traversal order is normalized geometrically to Dlib 68 conventions
    // (viewer's left-to-right, clockwise eye and lip loops starting at left X extrema).
    /// Pure landmark mapping helper translating variable Vision regions into complete Dlib 68 and semantic Vision 5.
    public static func mapLandmarks(from regions: RawVisionLandmarkRegions) throws -> (landmark68: FaceLandmark68, landmark5: FaceLandmark5) {
        // Strict validation: missing required regions throw an error rather than silently defaulting to zero
        guard let rawJaw = regions.faceContour, rawJaw.count >= 2 else {
            throw FaceDetectorError.detectionFailed("Missing or insufficient required landmark region: faceContour")
        }
        guard let rawLeftBrow = regions.leftEyebrow, rawLeftBrow.count >= 2 else {
            throw FaceDetectorError.detectionFailed("Missing or insufficient required landmark region: leftEyebrow")
        }
        guard let rawRightBrow = regions.rightEyebrow, rawRightBrow.count >= 2 else {
            throw FaceDetectorError.detectionFailed("Missing or insufficient required landmark region: rightEyebrow")
        }
        guard let rawNoseCrest = regions.noseCrest, rawNoseCrest.count >= 2 else {
            throw FaceDetectorError.detectionFailed("Missing or insufficient required landmark region: noseCrest")
        }
        guard let rawNoseBase = regions.nose, rawNoseBase.count >= 2 else {
            throw FaceDetectorError.detectionFailed("Missing or insufficient required landmark region: nose")
        }
        guard let rawLeftEye = regions.leftEye, rawLeftEye.count >= 3 else {
            throw FaceDetectorError.detectionFailed("Missing or insufficient required landmark region: leftEye")
        }
        guard let rawRightEye = regions.rightEye, rawRightEye.count >= 3 else {
            throw FaceDetectorError.detectionFailed("Missing or insufficient required landmark region: rightEye")
        }
        guard let rawOuterLips = regions.outerLips, rawOuterLips.count >= 3 else {
            throw FaceDetectorError.detectionFailed("Missing or insufficient required landmark region: outerLips")
        }
        guard let rawInnerLips = regions.innerLips, rawInnerLips.count >= 3 else {
            throw FaceDetectorError.detectionFailed("Missing or insufficient required landmark region: innerLips")
        }

        var points68 = [SIMD2<Float>](repeating: SIMD2<Float>(0, 0), count: 68)

        // 1. Jawline (Dlib 0..<17, 17 points): Start at viewer's left (min X) to viewer's right (max X)
        var jawPoints = rawJaw
        if jawPoints.first!.x > jawPoints.last!.x {
            jawPoints.reverse()
        }
        let resampledJaw = resampleOpenContour(jawPoints, targetCount: 17)
        for i in 0..<17 { points68[i] = resampledJaw[i] }

        // 2. Eyebrows (Dlib 17..<22 and 22..<27, 5 points each):
        // Sort regions left-to-right in viewer coordinates
        let brow1Center = rawLeftBrow.reduce(SIMD2<Float>(0, 0), +) / Float(rawLeftBrow.count)
        let brow2Center = rawRightBrow.reduce(SIMD2<Float>(0, 0), +) / Float(rawRightBrow.count)
        var browL = brow1Center.x <= brow2Center.x ? rawLeftBrow : rawRightBrow
        var browR = brow1Center.x <= brow2Center.x ? rawRightBrow : rawLeftBrow

        if browL.first!.x > browL.last!.x { browL.reverse() }
        if browR.first!.x > browR.last!.x { browR.reverse() }

        let resampledBrowL = resampleOpenContour(browL, targetCount: 5)
        let resampledBrowR = resampleOpenContour(browR, targetCount: 5)
        for i in 0..<5 {
            points68[17 + i] = resampledBrowL[i]
            points68[22 + i] = resampledBrowR[i]
        }

        // 3. Eyes (Dlib 36..<42 and 42..<48, 6 points each):
        // Sort regions left-to-right in viewer coordinates
        let eye1Center = rawLeftEye.reduce(SIMD2<Float>(0, 0), +) / Float(rawLeftEye.count)
        let eye2Center = rawRightEye.reduce(SIMD2<Float>(0, 0), +) / Float(rawRightEye.count)
        let eyeLPoints = eye1Center.x <= eye2Center.x ? rawLeftEye : rawRightEye
        let eyeRPoints = eye1Center.x <= eye2Center.x ? rawRightEye : rawLeftEye

        let semanticLeftEye = min(eye1Center.x, eye2Center.x) == eye1Center.x ? eye1Center : eye2Center
        let semanticRightEye = max(eye1Center.x, eye2Center.x) == eye2Center.x ? eye2Center : eye1Center

        // Resample left eye (4 upper points, 4 lower points -> 6 Dlib points)
        let (leftEyeUpper, leftEyeLower) = resampleClosedLoop(points: eyeLPoints, upperCount: 4, lowerCount: 4)
        points68[36] = leftEyeUpper[0]
        points68[37] = leftEyeUpper[1]
        points68[38] = leftEyeUpper[2]
        points68[39] = leftEyeUpper[3]
        points68[40] = leftEyeLower[1]
        points68[41] = leftEyeLower[2]

        // Resample right eye
        let (rightEyeUpper, rightEyeLower) = resampleClosedLoop(points: eyeRPoints, upperCount: 4, lowerCount: 4)
        points68[42] = rightEyeUpper[0]
        points68[43] = rightEyeUpper[1]
        points68[44] = rightEyeUpper[2]
        points68[45] = rightEyeUpper[3]
        points68[46] = rightEyeLower[1]
        points68[47] = rightEyeLower[2]

        // 4. Lips: Outer (Dlib 48..<60, 12 points) and Inner (Dlib 60..<68, 8 points)
        let (outerUpper, outerLower) = resampleClosedLoop(points: rawOuterLips, upperCount: 7, lowerCount: 7)
        for i in 0..<7 { points68[48 + i] = outerUpper[i] }
        for i in 1..<6 { points68[54 + i] = outerLower[i] }

        let (innerUpper, innerLower) = resampleClosedLoop(points: rawInnerLips, upperCount: 5, lowerCount: 5)
        for i in 0..<5 { points68[60 + i] = innerUpper[i] }
        for i in 1..<4 { points68[64 + i] = innerLower[i] }

        // Semantic lips extrema
        let semanticLeftMouth = outerUpper[0]   // Min X corner
        let semanticRightMouth = outerUpper[6]  // Max X corner

        // 5. Nose: Bridge/Crest (Dlib 27..<31, 4 points) and Base (Dlib 31..<36, 5 points)
        // Anatomical nose tip: Located on the vertical symmetry axis between eye center and mouth center.
        // Rather than guessing the truncated crest end, identify the actual apex between the lower crest and upper nose base.
        let eyeMidpoint = (semanticLeftEye + semanticRightEye) * 0.5
        let mouthMidpoint = (semanticLeftMouth + semanticRightMouth) * 0.5
        let facialAxis = mouthMidpoint - eyeMidpoint
        let axisLen = simd_length(facialAxis)
        let unitAxis = axisLen > 1e-6 ? facialAxis / axisLen : SIMD2<Float>(0, 1)

        // Find candidate in nose base closest to facial axis
        var bestNoseBasePt = rawNoseBase[0]
        var minPerpDist = Float.greatestFiniteMagnitude
        for pt in rawNoseBase {
            let offset = pt - eyeMidpoint
            let proj = simd_dot(offset, unitAxis)
            let perp = simd_length(offset - unitAxis * proj)
            if perp < minPerpDist {
                minPerpDist = perp
                bestNoseBasePt = pt
            }
        }

        var crestPoints = rawNoseCrest
        if crestPoints.first!.y > crestPoints.last!.y {
            crestPoints.reverse()
        }

        // The actual tip is the anatomical apex connecting the bridge to the columella
        let crestEnd = crestPoints.last!
        let actualNoseTip = (crestEnd + bestNoseBasePt) * 0.5

        // Resample crest to 4 points with index 30 landing at actual nose tip
        crestPoints[crestPoints.count - 1] = actualNoseTip
        let resampledCrest = resampleOpenContour(crestPoints, targetCount: 4)
        for i in 0..<4 { points68[27 + i] = resampledCrest[i] }

        var basePoints = rawNoseBase
        if basePoints.first!.x > basePoints.last!.x {
            basePoints.reverse()
        }
        let resampledBase = resampleOpenContour(basePoints, targetCount: 5)
        for i in 0..<5 { points68[31 + i] = resampledBase[i] }

        let landmark68 = FaceLandmark68(points: points68)
        let landmark5 = FaceLandmark5(points: [
            semanticLeftEye,
            semanticRightEye,
            actualNoseTip,
            semanticLeftMouth,
            semanticRightMouth
        ])

        return (landmark68, landmark5)
    }

    // MARK: - Single Face Detection

    /// Detects the single face in the provided CGImage, extracting bounding box and 68 landmarks.
    /// Strictly rejects images with 0 or >1 detected faces.
    public func detectSingleFace(in cgImage: CGImage) throws -> FaceTarget {
        let width = CGFloat(cgImage.width)
        let height = CGFloat(cgImage.height)

        let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        let request = VNDetectFaceLandmarksRequest()
        request.revision = VNDetectFaceLandmarksRequestRevision3 // iOS 15+ high accuracy landmarks

        try requestHandler.perform([request])

        guard let results = request.results else {
            throw FaceDetectorError.noFaceDetected
        }

        // Strictly enforce exactly one face; rejects > 1 faces without largest-face fallback
        let primaryFace = try Self.selectSingleFace(results)

        guard let landmarks2D = primaryFace.landmarks else {
            throw FaceDetectorError.detectionFailed("Detected face missing landmark regions")
        }

        // Convert Vision bounding box (normalized, bottom-left origin) to pixel rect (top-left origin)
        let bbox = primaryFace.boundingBox
        let pixelBbox = CGRect(
            x: bbox.origin.x * width,
            y: (1.0 - bbox.origin.y - bbox.height) * height,
            width: bbox.width * width,
            height: bbox.height * height
        )

        // Helper to convert normalized Vision region points to global pixel coordinates
        func toPixelPoints(_ region: VNFaceLandmarkRegion2D?) -> [SIMD2<Float>]? {
            guard let region = region, region.pointCount > 0 else { return nil }
            var pts = [SIMD2<Float>]()
            pts.reserveCapacity(region.pointCount)
            for i in 0..<region.pointCount {
                let normPt = region.normalizedPoints[i]
                let gx = Float((bbox.origin.x + normPt.x * bbox.width) * width)
                let gy = Float((1.0 - (bbox.origin.y + normPt.y * bbox.height)) * height)
                pts.append(SIMD2<Float>(gx, gy))
            }
            return pts
        }

        let rawRegions = RawVisionLandmarkRegions(
            faceContour: toPixelPoints(landmarks2D.faceContour),
            leftEyebrow: toPixelPoints(landmarks2D.leftEyebrow),
            rightEyebrow: toPixelPoints(landmarks2D.rightEyebrow),
            noseCrest: toPixelPoints(landmarks2D.noseCrest),
            nose: toPixelPoints(landmarks2D.nose),
            leftEye: toPixelPoints(landmarks2D.leftEye),
            rightEye: toPixelPoints(landmarks2D.rightEye),
            outerLips: toPixelPoints(landmarks2D.outerLips),
            innerLips: toPixelPoints(landmarks2D.innerLips)
        )

        let (landmark68, landmark5) = try Self.mapLandmarks(from: rawRegions)
        let angle = landmark68.estimateAngle()

        return FaceTarget(
            boundingBox: pixelBbox,
            landmark5: landmark5,
            landmark68: landmark68,
            angle: angle,
            score: primaryFace.confidence
        )
    }
}
