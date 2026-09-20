import Foundation
import UIKit
import SwiftUI
import PhotosUI
import Photos
import Combine

public enum ComparisonMode: String, CaseIterable, Identifiable {
    case before = "Before"
    case after = "After"
    public var id: String { rawValue }
}

@MainActor
public final class PhotoEditorViewModel: ObservableObject {
    // PhotosPicker items
    @Published public var sourceItem: PhotosPickerItem? {
        didSet { loadSourceImage(from: sourceItem) }
    }
    @Published public var targetItem: PhotosPickerItem? {
        didSet { loadTargetImage(from: targetItem) }
    }

    // Selected images and face detection state
    @Published public var sourceImage: UIImage?
    @Published public var targetImage: UIImage?
    @Published public var sourceStatus: FaceStatus = .idle
    @Published public var targetStatus: FaceStatus = .idle

    // Processors & settings
    @Published public var selectedProcessors: Set<ProcessorKind> = [.faceSwapper] {
        didSet { invalidateResult() }
    }
    @Published public var settings: ProcessorSettings = ProcessorSettings() {
        didSet { invalidateResult() }
    }

    // Output & processing state
    @Published public var resultImage: UIImage?
    @Published public var comparisonMode: ComparisonMode = .after
    @Published public var isProcessing: Bool = false
    @Published public var progressFraction: Double = 0.0
    @Published public var progressMessage: String = ""

    // Alerts, messages, and sheet presentations
    @Published public var errorMessage: String?
    @Published public var saveStatusMessage: String?
    @Published public var tempShareURL: URL?
    @Published public var showAdvancedSheet: Bool = false
    @Published public var showAboutSheet: Bool = false
    @Published public var showDFMImporter: Bool = false
    @Published public var importedDFMName: String?

    // Race-condition guards and cancellation tasks
    private var sourceLoadGeneration: Int = 0
    private var targetLoadGeneration: Int = 0
    private var sourceLoadTask: Task<Void, Never>?
    private var targetLoadTask: Task<Void, Never>?
    private var processGeneration: Int = 0
    private var processingTask: Task<Void, Never>?
    private var preparedShareURLTask: Task<URL, Error>?
    private var cancellables = Set<AnyCancellable>()

    public init() {
        // Lightweight observer for ModelCache download progress notifications
        NotificationCenter.default.publisher(for: ModelCache.downloadProgressNotification)
            .receive(on: RunLoop.main)
            .sink { [weak self] notification in
                guard let self = self, self.isProcessing else { return }
                guard let progress = notification.userInfo?["progress"] as? Double,
                      let modelId = notification.userInfo?["modelId"] as? String else {
                    return
                }
                self.progressMessage = "Downloading \(modelId) (\(Int(progress * 100))%)..."
            }
            .store(in: &cancellables)
    }

    // MARK: - Photo Loading & Face Validation

    private func loadSourceImage(from item: PhotosPickerItem?) {
        sourceLoadGeneration += 1
        let currentGen = sourceLoadGeneration
        sourceLoadTask?.cancel()
        invalidateResult()

        guard let item = item else {
            sourceImage = nil
            sourceStatus = .idle
            return
        }

        sourceStatus = .detecting
        sourceLoadTask = Task {
            do {
                guard let data = try await item.loadTransferable(type: Data.self) else {
                    if currentGen == self.sourceLoadGeneration {
                        self.sourceImage = nil
                        self.sourceStatus = .failed(reason: "Could not load image data")
                    }
                    return
                }

                guard currentGen == self.sourceLoadGeneration, !Task.isCancelled else { return }

                // Offload decoding, ImageBuffer normalization, and face detection to Task.detached
                let (uprightImage, count) = try await Task.detached(priority: .userInitiated) { () -> (UIImage, Int) in
                    guard let rawImage = UIImage(data: data) else {
                        throw NSError(domain: "iFaceFusion", code: -1, userInfo: [NSLocalizedDescriptionKey: "Could not decode source image"])
                    }
                    // Normalize orientation so pixel memory is upright
                    guard let normalizedBuffer = ImageBuffer(image: rawImage),
                          let uprightCg = normalizedBuffer.toCGImage() else {
                        throw NSError(domain: "iFaceFusion", code: -1, userInfo: [NSLocalizedDescriptionKey: "Could not normalize source orientation"])
                    }
                    // Validate face detection on upright CGImage
                    _ = try FaceDetector.shared.detectSingleFace(in: uprightCg)
                    let displayImage = UIImage(cgImage: uprightCg)
                    // normalizedBuffer is released here and not retained
                    return (displayImage, 1)
                }.value

                guard currentGen == self.sourceLoadGeneration, !Task.isCancelled else { return }
                self.sourceImage = uprightImage
                self.sourceStatus = .valid(count: count)
            } catch {
                if currentGen == self.sourceLoadGeneration && !Task.isCancelled {
                    self.sourceImage = nil
                    self.sourceStatus = .failed(reason: error.localizedDescription)
                }
            }
        }
    }

    private func loadTargetImage(from item: PhotosPickerItem?) {
        targetLoadGeneration += 1
        let currentGen = targetLoadGeneration
        targetLoadTask?.cancel()
        invalidateResult()

        guard let item = item else {
            targetImage = nil
            targetStatus = .idle
            return
        }

        targetStatus = .detecting
        targetLoadTask = Task {
            do {
                guard let data = try await item.loadTransferable(type: Data.self) else {
                    if currentGen == self.targetLoadGeneration {
                        self.targetImage = nil
                        self.targetStatus = .failed(reason: "Could not load image data")
                    }
                    return
                }

                guard currentGen == self.targetLoadGeneration, !Task.isCancelled else { return }

                // Offload decoding, ImageBuffer normalization, and face detection to Task.detached
                let (uprightImage, count) = try await Task.detached(priority: .userInitiated) { () -> (UIImage, Int) in
                    guard let rawImage = UIImage(data: data) else {
                        throw NSError(domain: "iFaceFusion", code: -1, userInfo: [NSLocalizedDescriptionKey: "Could not decode target image"])
                    }
                    // Normalize orientation so pixel memory is upright
                    guard let normalizedBuffer = ImageBuffer(image: rawImage),
                          let uprightCg = normalizedBuffer.toCGImage() else {
                        throw NSError(domain: "iFaceFusion", code: -1, userInfo: [NSLocalizedDescriptionKey: "Could not normalize target orientation"])
                    }
                    // Validate face detection on upright CGImage
                    _ = try FaceDetector.shared.detectSingleFace(in: uprightCg)
                    let displayImage = UIImage(cgImage: uprightCg)
                    // normalizedBuffer is released here and not retained
                    return (displayImage, 1)
                }.value

                guard currentGen == self.targetLoadGeneration, !Task.isCancelled else { return }
                self.targetImage = uprightImage
                self.targetStatus = .valid(count: count)
            } catch {
                if currentGen == self.targetLoadGeneration && !Task.isCancelled {
                    self.targetImage = nil
                    self.targetStatus = .failed(reason: error.localizedDescription)
                }
            }
        }
    }

    // MARK: - Processor Toggles & Ordering

    public func toggleProcessor(_ kind: ProcessorKind) {
        ProcessorOrder.toggle(kind, in: &selectedProcessors)
    }

    public var orderedProcessors: [ProcessorKind] {
        ProcessorOrder.sort(selectedProcessors)
    }

    public var requiresSourceImage: Bool {
        selectedProcessors.contains(.faceSwapper)
    }

    public var canProcess: Bool {
        guard !isProcessing, !selectedProcessors.isEmpty else { return false }
        // Target image must have exactly 1 validated face (enforced for all processors)
        guard targetImage != nil, case .valid(let targetCount) = targetStatus, targetCount == 1 else {
            return false
        }
        // Source image required per user flow (faceSwapper): must have exactly 1 validated face
        if requiresSourceImage {
            guard sourceImage != nil, case .valid(let sourceCount) = sourceStatus, sourceCount == 1 else {
                return false
            }
        }
        return true
    }

    // MARK: - Processing Execution

    public func process() {
        guard canProcess, let target = targetImage else {
            if targetImage == nil {
                errorMessage = "Please select a target photo."
            } else if case .failed(let reason) = targetStatus {
                errorMessage = "Target photo: \(reason)"
            } else if targetStatus == .detecting {
                errorMessage = "Target photo face detection in progress..."
            } else if case .valid(let count) = targetStatus, count != 1 {
                errorMessage = "Target photo must have exactly 1 face detected."
            } else if requiresSourceImage {
                if sourceImage == nil {
                    errorMessage = "Face Swapper requires a source identity photo."
                } else if case .failed(let reason) = sourceStatus {
                    errorMessage = "Source photo: \(reason)"
                } else if sourceStatus == .detecting {
                    errorMessage = "Source photo face detection in progress..."
                } else if case .valid(let count) = sourceStatus, count != 1 {
                    errorMessage = "Source photo must have exactly 1 face detected."
                }
            } else if selectedProcessors.isEmpty {
                errorMessage = "Please select at least one processor."
            } else {
                errorMessage = "Please verify your input photos."
            }
            return
        }

        processGeneration += 1
        let currentGen = processGeneration

        isProcessing = true
        progressFraction = 0.0
        progressMessage = "Initializing processing engine..."
        errorMessage = nil
        saveStatusMessage = nil

        var activeSettings = settings
        activeSettings.activeProcessors = orderedProcessors
        let source = self.sourceImage

        processingTask = Task {
            do {
                let output = try await ProcessingEngine.shared.process(
                    source: source,
                    target: target,
                    settings: activeSettings,
                    progress: { @Sendable [weak self] fraction, message in
                        Task { @MainActor in
                            guard self?.processGeneration == currentGen else { return }
                            self?.progressFraction = fraction
                            self?.progressMessage = message
                        }
                    }
                )

                guard currentGen == self.processGeneration, !Task.isCancelled else {
                    self.isProcessing = false
                    return
                }

                self.resultImage = output
                self.comparisonMode = .after
                self.isProcessing = false
                self.progressMessage = "Complete"
                self.prepareShareURL(for: output)
            } catch is CancellationError {
                guard currentGen == self.processGeneration else { return }
                self.isProcessing = false
                self.progressMessage = "Cancelled"
            } catch ProcessingEngineError.processingCancelled {
                guard currentGen == self.processGeneration else { return }
                self.isProcessing = false
                self.progressMessage = "Cancelled"
            } catch {
                guard currentGen == self.processGeneration else { return }
                self.isProcessing = false
                if !Task.isCancelled {
                    self.errorMessage = error.localizedDescription
                } else {
                    self.progressMessage = "Cancelled"
                }
            }
        }
    }

    public func cancelProcessing() {
        guard isProcessing else { return }
        processingTask?.cancel()
        progressMessage = "Cancelling..."
        // ponytail: Keep isProcessing = true until ProcessingEngine returns to prevent overlapping execution.
    }

    public func invalidateResult() {
        if resultImage != nil {
            resultImage = nil
        }
        if tempShareURL != nil {
            try? FileManager.default.removeItem(at: tempShareURL!)
            tempShareURL = nil
        }
        preparedShareURLTask?.cancel()
        preparedShareURLTask = nil
        if isProcessing {
            cancelProcessing()
        }
    }

    // MARK: - Save and Share

    public func saveResultToPhotos() {
        guard resultImage != nil else { return }
        Task {
            do {
                let authStatus = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
                guard authStatus == .authorized || authStatus == .limited else {
                    self.errorMessage = "Photos access was denied. Please allow photo library access in Settings to save photos."
                    return
                }

                let fileURL: URL
                if let existing = self.tempShareURL {
                    fileURL = existing
                } else if let shareTask = self.preparedShareURLTask {
                    fileURL = try await shareTask.value
                } else if let result = self.resultImage {
                    fileURL = try await Task.detached(priority: .userInitiated) {
                        guard let data = result.pngData() else {
                            throw NSError(domain: "iFaceFusion", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to encode PNG data"])
                        }
                        let tempDir = FileManager.default.temporaryDirectory
                        let url = tempDir.appendingPathComponent("iFaceFusion_\(UUID().uuidString).png")
                        try data.write(to: url)
                        return url
                    }.value
                    self.tempShareURL = fileURL
                } else {
                    return
                }

                try await PHPhotoLibrary.shared().performChanges {
                    let creationRequest = PHAssetCreationRequest.forAsset()
                    let options = PHAssetResourceCreationOptions()
                    options.shouldMoveFile = false
                    creationRequest.addResource(with: .photo, fileURL: fileURL, options: options)
                }
                self.saveStatusMessage = "Photo successfully saved to Photos library!"
            } catch {
                self.errorMessage = "Failed to save photo: \(error.localizedDescription)"
            }
        }
    }

    private func prepareShareURL(for image: UIImage) {
        preparedShareURLTask?.cancel()
        let shareTask = Task.detached(priority: .userInitiated) { () -> URL in
            guard let data = image.pngData() else {
                throw NSError(domain: "iFaceFusion", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to encode PNG data"])
            }
            let tempDir = FileManager.default.temporaryDirectory
            let fileURL = tempDir.appendingPathComponent("iFaceFusion_\(UUID().uuidString).png")
            try data.write(to: fileURL)
            return fileURL
        }
        self.preparedShareURLTask = shareTask
        Task {
            do {
                let url = try await shareTask.value
                guard !Task.isCancelled else { return }
                self.tempShareURL = url
            } catch {
                // Silently handle if cancelled
            }
        }
    }

    // MARK: - DFM Model Import

    public func handleImportedDFM(at url: URL) {
        guard url.startAccessingSecurityScopedResource() else {
            errorMessage = "Failed to access security-scoped file URL"
            return
        }
        defer { url.stopAccessingSecurityScopedResource() }

        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        let destDir = appSupport.appendingPathComponent("iFaceFusion/DFMModels", isDirectory: true)

        do {
            try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)
            let destURL = destDir.appendingPathComponent(url.lastPathComponent)
            let stagingURL = destDir.appendingPathComponent("\(UUID().uuidString).tmp")

            defer {
                if FileManager.default.fileExists(atPath: stagingURL.path) {
                    try? FileManager.default.removeItem(at: stagingURL)
                }
            }

            // Copy to staging first to preserve original at destURL on failure
            try FileManager.default.copyItem(at: url, to: stagingURL)

            // Atomic replace or move
            if FileManager.default.fileExists(atPath: destURL.path) {
                _ = try FileManager.default.replaceItemAt(destURL, withItemAt: stagingURL)
            } else {
                try FileManager.default.moveItem(at: stagingURL, to: destURL)
            }

            settings.deepSwapper.model = destURL.path
            importedDFMName = url.lastPathComponent
        } catch {
            errorMessage = "Failed to import DFM model: \(error.localizedDescription)"
        }
    }
}
