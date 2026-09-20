import Foundation
import simd

/// Standard landmark alignment templates matching FaceFusion WARP_TEMPLATE_SET.
public enum WarpTemplate: String, CaseIterable, Sendable {
    case arcface112v1 = "arcface_112_v1"
    case arcface112v2 = "arcface_112_v2"
    case arcface128 = "arcface_128"
    case dflWholeFace = "dfl_whole_face"
    case ffhq512 = "ffhq_512"
    case mtcnn512 = "mtcnn_512"
    case styleganex384 = "styleganex_384"

    /// Normalized landmark coordinates in [0, 1] relative to crop dimensions.
    public var normalizedPoints: [SIMD2<Float>] {
        switch self {
        case .arcface112v1:
            return [
                SIMD2<Float>(0.35473214, 0.45658929),
                SIMD2<Float>(0.64526786, 0.45658929),
                SIMD2<Float>(0.50000000, 0.61154464),
                SIMD2<Float>(0.37913393, 0.77687500),
                SIMD2<Float>(0.62086607, 0.77687500)
            ]
        case .arcface112v2:
            return [
                SIMD2<Float>(0.34191607, 0.46157411),
                SIMD2<Float>(0.65653393, 0.45983393),
                SIMD2<Float>(0.50022500, 0.64050536),
                SIMD2<Float>(0.37097589, 0.82469196),
                SIMD2<Float>(0.63151696, 0.82325089)
            ]
        case .arcface128:
            return [
                SIMD2<Float>(0.36167656, 0.40387734),
                SIMD2<Float>(0.63696719, 0.40235469),
                SIMD2<Float>(0.50019687, 0.56044219),
                SIMD2<Float>(0.38710391, 0.72160547),
                SIMD2<Float>(0.61507734, 0.72034453)
            ]
        case .dflWholeFace:
            return [
                SIMD2<Float>(0.35342266, 0.39285716),
                SIMD2<Float>(0.62797622, 0.39285716),
                SIMD2<Float>(0.48660713, 0.54017860),
                SIMD2<Float>(0.38839287, 0.68750011),
                SIMD2<Float>(0.59821427, 0.68750011)
            ]
        case .ffhq512:
            return [
                SIMD2<Float>(0.37691676, 0.46864664),
                SIMD2<Float>(0.62285697, 0.46912813),
                SIMD2<Float>(0.50123859, 0.61331904),
                SIMD2<Float>(0.39308822, 0.72541100),
                SIMD2<Float>(0.61150205, 0.72490465)
            ]
        case .mtcnn512:
            return [
                SIMD2<Float>(0.36562865, 0.46733799),
                SIMD2<Float>(0.63305391, 0.46585885),
                SIMD2<Float>(0.50019127, 0.61942959),
                SIMD2<Float>(0.39032951, 0.77598822),
                SIMD2<Float>(0.61178945, 0.77476328)
            ]
        case .styleganex384:
            return [
                SIMD2<Float>(0.42353745, 0.52289879),
                SIMD2<Float>(0.57725008, 0.52319972),
                SIMD2<Float>(0.50123859, 0.61331904),
                SIMD2<Float>(0.43364461, 0.68337652),
                SIMD2<Float>(0.57015325, 0.68306005)
            ]
        }
    }

    /// Computes target pixel coordinates scaled to a target width and height.
    public func targetPoints(width: Float, height: Float) -> [SIMD2<Float>] {
        let size = SIMD2<Float>(width, height)
        return normalizedPoints.map { $0 * size }
    }
}
