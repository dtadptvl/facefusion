import Foundation
import UIKit
import SwiftUI
import PhotosUI
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
    private var processingTask: Task<Void, Never>?

    public init() {}

    // MARK: - Photo Loading & Face Validation

    private func loadSourceImage(from item: PhotosPickerItem?) {
        sourceLoadGeneration += 1
        let currentGen = sourceLoadGeneration
        invalidateResult()

        guard let item = item else {
            sourceImage = nil
            sourceStatus = .idle
            return
        }

        sourceStatus = .detecting
        Task {
            do {
                guard let data = try await item.loadTransferable(type: Data.self),
                      let image = UIImage(data: data) else {
                    if currentGen == self.sourceLoadGeneration {
                        self.sourceStatus = .failed(reason: "Could not decode source image")
                    }
                    return
                }

                guard currentGen == self.sourceLoadGeneration else { return }
                self.sourceImage = image

                // Validate exactly one face in source
                guard let cg = image.cgImage else {
                    self.sourceStatus = .failed(reason: "Invalid image format")
                    return
                }

                _ = try FaceDetector.shared.detectSingleFace(in: cg)
                if currentGen == self.sourceLoadGeneration {
                    self.sourceStatus = .valid(count: 1)
                }
            } catch {
                if currentGen == self.sourceLoadGeneration {
                    self.sourceStatus = .failed(reason: error.localizedDescription)
                }
            }
        }
    }

    private func loadTargetImage(from item: PhotosPickerItem?) {
        targetLoadGeneration += 1
        let currentGen = targetLoadGeneration
        invalidateResult()

        guard let item = item else {
            targetImage = nil
            targetStatus = .idle
            return
        }

        targetStatus = .detecting
        Task {
            do {
                guard let data = try await item.loadTransferable(type: Data.self),
                      let image = UIImage(data: data) else {
                    if currentGen == self.targetLoadGeneration {
                        self.targetStatus = .failed(reason: "Could not decode target image")
                    }
                    return
                }

                guard currentGen == self.targetLoadGeneration else { return }
                self.targetImage = image

                // Validate face detection if target requires face
                guard let cg = image.cgImage else {
                    self.targetStatus = .failed(reason: "Invalid image format")
                    return
                }

                _ = try FaceDetector.shared.detectSingleFace(in: cg)
                if currentGen == self.targetLoadGeneration {
                    self.targetStatus = .valid(count: 1)
                }
            } catch {
                if currentGen == self.targetLoadGeneration {
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
        guard !isProcessing, let _ = targetImage else { return false }
        if requiresSourceImage && sourceImage == nil { return false }
        return !selectedProcessors.isEmpty
    }

    // MARK: - Processing Execution

    public func process() {
        guard canProcess, let target = targetImage else {
            errorMessage = "Please select the required input photos."
            return
        }

        if requiresSourceImage && sourceImage == nil {
            errorMessage = "Face Swapper requires a source identity photo."
            return
        }

        processingTask?.cancel()
        isProcessing = true
        progressFraction = 0.0
        progressMessage = "Initializing processing engine..."
        errorMessage = nil
        saveStatusMessage = nil

        var activeSettings = settings
        activeSettings.activeProcessors = orderedProcessors

        processingTask = Task {
            do {
                let output = try await ProcessingEngine.shared.process(
                    source: self.sourceImage,
                    target: target,
                    settings: activeSettings,
                    progress: { @Sendable [weak self] fraction, message in
                        Task { @MainActor in
                            self?.progressFraction = fraction
                            self?.progressMessage = message
                        }
                    }
                )

                guard !Task.isCancelled else { return }
                self.resultImage = output
                self.comparisonMode = .after
                self.isProcessing = false
                self.prepareShareURL(for: output)
            } catch {
                guard !Task.isCancelled else { return }
                self.isProcessing = false
                self.errorMessage = error.localizedDescription
            }
        }
    }

    public func cancelProcessing() {
        processingTask?.cancel()
        processingTask = nil
        isProcessing = false
        progressMessage = "Cancelled"
    }

    public func invalidateResult() {
        if resultImage != nil {
            resultImage = nil
        }
        if tempShareURL != nil {
            try? FileManager.default.removeItem(at: tempShareURL!)
            tempShareURL = nil
        }
        if isProcessing {
            cancelProcessing()
        }
    }

    // MARK: - Save and Share

    public func saveResultToPhotos() {
        guard let result = resultImage else { return }
        // ponytail: Direct UIImageWriteToSavedPhotosAlbum suffices for photo library export. Upgrade to PHPhotoLibrary for album grouping.
        UIImageWriteToSavedPhotosAlbum(result, self, #selector(saveCompleted(_:didFinishSavingWithError:contextInfo:)), nil)
    }

    @objc private func saveCompleted(_ image: UIImage, didFinishSavingWithError error: Error?, contextInfo: UnsafeRawPointer?) {
        if let error = error {
            errorMessage = "Failed to save photo: \(error.localizedDescription)"
        } else {
            saveStatusMessage = "Photo successfully saved to Photos library!"
        }
    }

    private func prepareShareURL(for image: UIImage) {
        guard let data = image.pngData() else { return }
        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent("iFaceFusion_\(Int(Date().timeIntervalSince1970)).png")
        do {
            try data.write(to: fileURL)
            self.tempShareURL = fileURL
        } catch {
            print("Failed to write temporary share PNG: \(error)")
        }
    }

    // MARK: - DFM Model Import

    public func handleImportedDFM(at url: URL) {
        guard url.startAccessingSecurityScopedResource() else { return }
        defer { url.stopAccessingSecurityScopedResource() }

        let destDir = FileManager.default.temporaryDirectory.appendingPathComponent("dfm_models")
        try? FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)
        let destURL = destDir.appendingPathComponent(url.lastPathComponent)

        try? FileManager.default.removeItem(at: destURL)
        do {
            try FileManager.default.copyItem(at: url, to: destURL)
            settings.deepSwapper.model = destURL.path
            importedDFMName = url.lastPathComponent
        } catch {
            errorMessage = "Failed to import DFM model: \(error.localizedDescription)"
        }
    }
}
