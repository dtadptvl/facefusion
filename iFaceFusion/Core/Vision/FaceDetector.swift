import Foundation
import CoreGraphics
import Vision
import UIKit

public enum FaceDetectorError: LocalizedError {
    case detectionFailed(String)
    case noFaceDetected

    public var errorDescription: String? {
        switch self {
        case .detectionFailed(let reason):
            return "Apple Vision face detection failed: \(reason)"
        case .noFaceDetected:
            return "No face was detected in the target image."
        }
    }
}

/// Face detector leveraging Apple Vision Framework to detect exactly one primary face and its 68 facial landmarks.
public final class FaceDetector: Sendable {
    public static let shared = FaceDetector()

    public init() {}

    /// Detects the single largest face in the provided CGImage, extracting bounding box and 68 landmarks.
    public func detectSingleFace(in cgImage: CGImage) throws -> FaceTarget {
        let width = CGFloat(cgImage.width)
        let height = CGFloat(cgImage.height)

        let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        let request = VNDetectFaceLandmarksRequest()
        request.revision = VNDetectFaceLandmarksRequestRevision3 // iOS 15+ high accuracy landmarks

        try requestHandler.perform([request])

        guard let results = request.results, !results.isEmpty else {
            throw FaceDetectorError.noFaceDetected
        }

        // Selection policy: pick the largest face by bounding box area to ensure exactly one face is targeted
        let primaryFace = results.max { a, b in
            let areaA = a.boundingBox.width * a.boundingBox.height
            let areaB = b.boundingBox.width * b.boundingBox.height
            return areaA < areaB
        }!

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

        // Map Vision landmark regions into 68-point landmark array
        var points68 = [SIMD2<Float>](repeating: SIMD2<Float>(0, 0), count: 68)

        func convertRegionPoints(_ region: VNFaceLandmarkRegion2D?, targetIndices: [Int]) {
            guard let region = region else { return }
            let count = min(region.pointCount, targetIndices.count)
            for i in 0..<count {
                let normPt = region.normalizedPoints[i]
                // Convert normalized point within face bounding box to global pixel coordinates
                let gx = (bbox.origin.x + normPt.x * bbox.width) * width
                let gy = (1.0 - (bbox.origin.y + normPt.y * bbox.height)) * height
                points68[targetIndices[i]] = SIMD2<Float>(Float(gx), Float(gy))
            }
        }

        // 1. Face contour: indices 0..<17
        convertRegionPoints(landmarks2D.faceContour, targetIndices: Array(0..<17))

        // 2. Eyebrows: left 17..<22, right 22..<27
        convertRegionPoints(landmarks2D.leftEyebrow, targetIndices: Array(17..<22))
        convertRegionPoints(landmarks2D.rightEyebrow, targetIndices: Array(22..<27))

        // 3. Nose: crest 27..<31, base 31..<36
        convertRegionPoints(landmarks2D.noseCrest, targetIndices: Array(27..<31))
        convertRegionPoints(landmarks2D.nose, targetIndices: Array(31..<36))

        // 4. Eyes: left 36..<42, right 42..<48
        convertRegionPoints(landmarks2D.leftEye, targetIndices: Array(36..<42))
        convertRegionPoints(landmarks2D.rightEye, targetIndices: Array(42..<48))

        // 5. Lips: outer 48..<60, inner 60..<68
        convertRegionPoints(landmarks2D.outerLips, targetIndices: Array(48..<60))
        convertRegionPoints(landmarks2D.innerLips, targetIndices: Array(60..<68))

        let landmark68 = FaceLandmark68(points: points68)
        let landmark5 = landmark68.toLandmark5()
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
