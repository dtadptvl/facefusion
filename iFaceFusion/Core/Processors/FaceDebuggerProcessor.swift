import Foundation
import CoreGraphics
import UIKit

/// Implements FaceFusion face_debugger processor drawing bounding boxes, orientation indicators,
/// landmark points, and mask outlines directly onto the target frame.
public final class FaceDebuggerProcessor: Sendable {
    public init() {}

    public func process(
        targetImage: ImageBuffer,
        targetFace: FaceTarget,
        settings: FaceDebuggerSettings,
        maskSettings: FaceMaskSettings
    ) -> ImageBuffer {
        let width = targetImage.width
        let height = targetImage.height

        guard let baseCG = targetImage.toCGImage() else { return targetImage }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else { return targetImage }

        // Draw original image (CoreGraphics y is bottom-to-top)
        context.draw(baseCG, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Flip coordinates for top-left drawing matching standard image coordinates
        context.saveGState()
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1.0, y: -1.0)

        let lineScale = max(1.0, CGFloat(round(Float(height) / 270.0)))

        // 1. Bounding Box & Face Angle Indicator
        if settings.items.contains(.boundingBox) {
            let bbox = targetFace.boundingBox
            context.setLineWidth(lineScale)
            context.setStrokeColor(UIColor.red.cgColor)
            context.stroke(bbox)

            // Draw orientation top-edge line
            context.setLineWidth(lineScale + 1.0)
            context.setStrokeColor(UIColor(red: 0.4, green: 0.4, blue: 1.0, alpha: 1.0).cgColor)
            context.beginPath()
            switch targetFace.angle {
            case 90:
                context.move(to: CGPoint(x: bbox.maxX, y: bbox.minY))
                context.addLine(to: CGPoint(x: bbox.maxX, y: bbox.maxY))
            case 180:
                context.move(to: CGPoint(x: bbox.minX, y: bbox.maxY))
                context.addLine(to: CGPoint(x: bbox.maxX, y: bbox.maxY))
            case 270:
                context.move(to: CGPoint(x: bbox.minX, y: bbox.minY))
                context.addLine(to: CGPoint(x: bbox.minX, y: bbox.maxY))
            default: // 0 degrees
                context.move(to: CGPoint(x: bbox.minX, y: bbox.minY))
                context.addLine(to: CGPoint(x: bbox.maxX, y: bbox.minY))
            }
            context.strokePath()
        }

        // 2. 5-point landmarks
        if settings.items.contains(.landmark5) {
            let ptRadius = lineScale * 1.5
            context.setFillColor(UIColor.red.cgColor)
            for pt in targetFace.landmark5.points {
                let rect = CGRect(x: CGFloat(pt.x) - ptRadius, y: CGFloat(pt.y) - ptRadius, width: ptRadius * 2, height: ptRadius * 2)
                context.fillEllipse(in: rect)
            }
        }

        // 3. 68-point landmarks
        if settings.items.contains(.landmark68) {
            let ptRadius = lineScale * 1.0
            context.setFillColor(UIColor.green.cgColor)
            for pt in targetFace.landmark68.points {
                let rect = CGRect(x: CGFloat(pt.x) - ptRadius, y: CGFloat(pt.y) - ptRadius, width: ptRadius * 2, height: ptRadius * 2)
                context.fillEllipse(in: rect)
            }
        }

        // 4. Face mask boundary outline
        if settings.items.contains(.faceMask) {
            let cropSize = 512
            let template = WarpTemplate.arcface128
            let targetPoints = template.targetPoints(width: Float(cropSize), height: Float(cropSize))
            let matrix = ImageGeometry.estimateSimilarityMatrix(src: targetFace.landmark5.points, dst: targetPoints)

            if let invMatrix = matrix.inverted() {
                let boxMask = FaceMask.createBoxMask(width: cropSize, height: cropSize, blur: 0, padding: maskSettings.padding)
                // Sample border corners and project back
                context.setLineWidth(lineScale)
                context.setStrokeColor(UIColor.green.cgColor)
                let c0 = invMatrix.transformPoint(SIMD2<Float>(Float(cropSize * maskSettings.padding.left / 100), Float(cropSize * maskSettings.padding.top / 100)))
                let c1 = invMatrix.transformPoint(SIMD2<Float>(Float(cropSize * (100 - maskSettings.padding.right) / 100), Float(cropSize * maskSettings.padding.top / 100)))
                let c2 = invMatrix.transformPoint(SIMD2<Float>(Float(cropSize * (100 - maskSettings.padding.right) / 100), Float(cropSize * (100 - maskSettings.padding.bottom) / 100)))
                let c3 = invMatrix.transformPoint(SIMD2<Float>(Float(cropSize * maskSettings.padding.left / 100), Float(cropSize * (100 - maskSettings.padding.bottom) / 100)))

                context.beginPath()
                context.move(to: CGPoint(x: CGFloat(c0.x), y: CGFloat(c0.y)))
                context.addLine(to: CGPoint(x: CGFloat(c1.x), y: CGFloat(c1.y)))
                context.addLine(to: CGPoint(x: CGFloat(c2.x), y: CGFloat(c2.y)))
                context.addLine(to: CGPoint(x: CGFloat(c3.x), y: CGFloat(c3.y)))
                context.closePath()
                context.strokePath()
            }
        }

        context.restoreGState()

        guard let outputCG = context.makeImage() else { return targetImage }
        return ImageBuffer(cgImage: outputCG) ?? targetImage
    }
}
