import Foundation

/// Detection status for selected source or target still images.
public enum FaceStatus: Equatable, Sendable {
    case idle
    case detecting
    case valid(count: Int)
    case failed(reason: String)

    public var isSuccess: Bool {
        if case .valid = self { return true }
        return false
    }

    public var statusDescription: String {
        switch self {
        case .idle:
            return "No image selected"
        case .detecting:
            return "Detecting face..."
        case .valid(let count):
            return count == 1 ? "1 face detected" : "\(count) faces detected"
        case .failed(let reason):
            return reason
        }
    }
}
