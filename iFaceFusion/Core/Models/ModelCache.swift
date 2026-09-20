import Foundation
import CryptoKit

public enum ModelCacheError: LocalizedError {
    case downloadFailed(String)
    case integrityCheckFailed(expected: String, actual: String)
    case fileSystemError(String)
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .downloadFailed(let reason):
            return "Model download failed: \(reason)"
        case .integrityCheckFailed(let expected, let actual):
            return "Model integrity verification failed: expected CRC32 \(expected), but got \(actual)"
        case .fileSystemError(let reason):
            return "File system error: \(reason)"
        case .cancelled:
            return "Model download was cancelled"
        }
    }
}

/// Thread-safe local cache manager for integrity-verified model storage and atomic lazy downloads.
public actor ModelCache {
    public static let shared = ModelCache()

    // MARK: - Progress Notification Hooks

    public static let downloadProgressNotification = Notification.Name("ModelCache.downloadProgress")
    public static let downloadCompletedNotification = Notification.Name("ModelCache.downloadCompleted")
    public static let downloadFailedNotification = Notification.Name("ModelCache.downloadFailed")

    public let cacheDirectory: URL

    public init(customCacheDirectory: URL? = nil) {
        if let customDir = customCacheDirectory {
            self.cacheDirectory = customDir
        } else {
            let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
                ?? URL(fileURLWithPath: NSTemporaryDirectory())
            self.cacheDirectory = appSupport.appendingPathComponent("iFaceFusion/Models", isDirectory: true)
        }
        try? FileManager.default.createDirectory(at: self.cacheDirectory, withIntermediateDirectories: true)
    }

    /// Destination file URL for a model in the cache directory.
    public func localURL(for modelId: String) -> URL {
        cacheDirectory.appendingPathComponent("\(modelId).onnx")
    }

    /// Checks if model file exists and passes CRC32 integrity verification.
    public func isModelAvailableLocally(_ metadata: ModelMetadata) -> Bool {
        let fileURL = localURL(for: metadata.id)
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return false }
        guard let actualCRC = try? computeCRC32(fileURL: fileURL) else { return false }
        guard actualCRC.lowercased() == metadata.expectedCRC32.lowercased() else { return false }
        if let expectedSHA = metadata.expectedSHA256 {
            guard let actualSHA = try? computeSHA256(fileURL: fileURL) else { return false }
            guard actualSHA.lowercased() == expectedSHA.lowercased() else { return false }
        }
        return true
    }

    /// Ensures the model is present on disk and integrity verified, downloading lazily if missing.
    public func ensureModelDownloaded(
        _ metadata: ModelMetadata,
        progressHandler: (@Sendable (Double) -> Void)? = nil
    ) async throws -> URL {
        let targetURL = localURL(for: metadata.id)

        // Offline reuse: verify existing cached model
        if isModelAvailableLocally(metadata) {
            progressHandler?(1.0)
            return targetURL
        } else if FileManager.default.fileExists(atPath: targetURL.path) {
            // Corrupted file: remove prior to re-downloading
            try? FileManager.default.removeItem(at: targetURL)
        }

        // Attempt download from candidate sources in priority order
        var lastError: Error? = nil
        for source in metadata.sources {
            if Task.isCancelled { throw ModelCacheError.cancelled }
            do {
                let downloadedURL = try await downloadAtomic(from: source.url, metadata: metadata, progressHandler: progressHandler)
                return downloadedURL
            } catch {
                lastError = error
                continue
            }
        }

        NotificationCenter.default.post(
            name: Self.downloadFailedNotification,
            object: metadata.id,
            userInfo: ["modelId": metadata.id, "error": lastError?.localizedDescription ?? "No download sources"]
        )
        throw lastError ?? ModelCacheError.downloadFailed("No available download source for \(metadata.id)")
    }

    /// Downloads atomically via URLSession.download(for:delegate:), validates CRC32/SHA256, and moves to target.
    private func downloadAtomic(
        from url: URL,
        metadata: ModelMetadata,
        progressHandler: (@Sendable (Double) -> Void)?
    ) async throws -> URL {
        let targetURL = localURL(for: metadata.id)
        let stagingURL = cacheDirectory.appendingPathComponent("\(metadata.id)_\(UUID().uuidString).tmp")

        defer {
            if FileManager.default.fileExists(atPath: stagingURL.path) {
                try? FileManager.default.removeItem(at: stagingURL)
            }
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = 120.0

        let progressDelegate = DownloadProgressDelegate(modelId: metadata.id, progressHandler: progressHandler)

        let tempDownloadedURL: URL
        let response: URLResponse
        do {
            let (loc, resp) = try await URLSession.shared.download(for: request, delegate: progressDelegate)
            tempDownloadedURL = loc
            response = resp
        } catch is CancellationError {
            throw ModelCacheError.cancelled
        } catch let urlErr as URLError where urlErr.code == .cancelled {
            throw ModelCacheError.cancelled
        } catch {
            throw ModelCacheError.downloadFailed(error.localizedDescription)
        }

        guard let httpResponse = response as? HTTPURLResponse, (200...299).contains(httpResponse.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            throw ModelCacheError.downloadFailed("HTTP status \(code)")
        }

        // Move system download file to cache staging URL
        do {
            if FileManager.default.fileExists(atPath: stagingURL.path) {
                try FileManager.default.removeItem(at: stagingURL)
            }
            try FileManager.default.moveItem(at: tempDownloadedURL, to: stagingURL)
        } catch {
            throw ModelCacheError.fileSystemError("Failed to stage downloaded model: \(error.localizedDescription)")
        }

        // Integrity verification: CRC32
        let actualCRC32 = try computeCRC32(fileURL: stagingURL).lowercased()
        if actualCRC32 != metadata.expectedCRC32.lowercased() {
            throw ModelCacheError.integrityCheckFailed(expected: metadata.expectedCRC32, actual: actualCRC32)
        }

        // Integrity verification: SHA256 (if specified)
        if let expectedSHA = metadata.expectedSHA256 {
            let actualSHA = try computeSHA256(fileURL: stagingURL).lowercased()
            if actualSHA != expectedSHA.lowercased() {
                throw ModelCacheError.integrityCheckFailed(expected: expectedSHA, actual: actualSHA)
            }
        }

        // Atomic move into final cache location
        if FileManager.default.fileExists(atPath: targetURL.path) {
            try? FileManager.default.removeItem(at: targetURL)
        }
        try FileManager.default.moveItem(at: stagingURL, to: targetURL)

        progressHandler?(1.0)
        NotificationCenter.default.post(
            name: Self.downloadCompletedNotification,
            object: metadata.id,
            userInfo: ["modelId": metadata.id, "targetURL": targetURL]
        )
        return targetURL
    }

    // MARK: - Checksum Computation

    private static let crcTable: [UInt32] = {
        (0..<256).map { i -> UInt32 in
            var c = UInt32(i)
            for _ in 0..<8 {
                if (c & 1) != 0 {
                    c = 0xEDB88320 ^ (c >> 1)
                } else {
                    c = c >> 1
                }
            }
            return c
        }
    }()

    /// Computes CRC32 hex string (8 characters, lowercase) matching format(zlib.crc32(content), '08x').
    public func computeCRC32(fileURL: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: fileURL)
        defer { try? handle.close() }

        var crc: UInt32 = 0xFFFFFFFF
        let bufferSize = 65536
        while true {
            let chunk = try handle.read(upToCount: bufferSize)
            guard let chunk = chunk, !chunk.isEmpty else { break }
            chunk.withUnsafeBytes { rawBuffer in
                guard let ptr = rawBuffer.bindMemory(to: UInt8.self).baseAddress else { return }
                for i in 0..<chunk.count {
                    let byte = ptr[i]
                    let tableIndex = Int((crc ^ UInt32(byte)) & 0xFF)
                    crc = Self.crcTable[tableIndex] ^ (crc >> 8)
                }
            }
        }
        crc = crc ^ 0xFFFFFFFF
        return String(format: "%08x", crc)
    }

    /// Computes SHA256 hex string.
    public func computeSHA256(fileURL: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: fileURL)
        defer { try? handle.close() }

        var hasher = SHA256()
        let bufferSize = 65536
        while true {
            let chunk = try handle.read(upToCount: bufferSize)
            guard let chunk = chunk, !chunk.isEmpty else { break }
            hasher.update(data: chunk)
        }
        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - Download Progress Delegate

private final class DownloadProgressDelegate: NSObject, URLSessionDownloadDelegate, Sendable {
    private let modelId: String
    private let progressHandler: (@Sendable (Double) -> Void)?

    init(modelId: String, progressHandler: (@Sendable (Double) -> Void)?) {
        self.modelId = modelId
        self.progressHandler = progressHandler
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let fraction = min(1.0, max(0.0, Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)))
        progressHandler?(fraction)
        NotificationCenter.default.post(
            name: ModelCache.downloadProgressNotification,
            object: modelId,
            userInfo: [
                "modelId": modelId,
                "progress": fraction,
                "totalBytesWritten": totalBytesWritten,
                "totalBytesExpected": totalBytesExpectedToWrite
            ]
        )
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        // Destination URL lifecycle managed by URLSession.download(for:delegate:)
    }
}
