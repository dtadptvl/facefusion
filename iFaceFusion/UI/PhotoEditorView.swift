import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

public struct PhotoEditorView: View {
    @StateObject private var viewModel = PhotoEditorViewModel()
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    public init() {}

    public var body: some View {
        NavigationStack {
            ZStack(alignment: .bottom) {
                // Main content scroll view
                ScrollView {
                    VStack(spacing: 20) {
                        // 1. Result Comparison Section (when result is available)
                        if let result = viewModel.resultImage {
                            resultComparisonSection(resultImage: result)
                        }

                        // 2. Input Cards: Source and Target
                        inputCardsSection

                        // 3. Multi-Select Processors
                        processorSelectionSection

                        // Spacer for safe area CTA button
                        Spacer()
                            .frame(height: 100)
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 8)
                }

                // 4. Safe Area Process CTA (floating bottom bar)
                processCTASection
            }
            .navigationTitle("iFaceFusion")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button {
                        viewModel.showAboutSheet = true
                    } label: {
                        Image(systemName: "info.circle")
                            .font(.system(size: 18))
                            .frame(minWidth: 44, minHeight: 44)
                            .contentShape(Rectangle())
                    }
                    .accessibilityLabel("About and Licenses")
                    .disabled(viewModel.isProcessing)
                }

                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        viewModel.showAdvancedSheet = true
                    } label: {
                        Image(systemName: "slider.horizontal.3")
                            .font(.system(size: 18))
                            .frame(minWidth: 44, minHeight: 44)
                            .contentShape(Rectangle())
                    }
                    .accessibilityLabel("Advanced Settings")
                    .disabled(viewModel.isProcessing)
                }
            }
            .sheet(isPresented: $viewModel.showAdvancedSheet) {
                AdvancedSettingsSheet(settings: $viewModel.settings)
            }
            .sheet(isPresented: $viewModel.showAboutSheet) {
                AboutLicensesSheet()
            }
            .fileImporter(
                isPresented: $viewModel.showDFMImporter,
                allowedContentTypes: [.data, UTType(filenameExtension: "dfm") ?? .data, UTType(filenameExtension: "onnx") ?? .data],
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    if let first = urls.first {
                        viewModel.handleImportedDFM(at: first)
                    }
                case .failure(let error):
                    viewModel.errorMessage = "DFM file selection error: \(error.localizedDescription)"
                }
            }
            .alert("Error", isPresented: Binding(
                get: { viewModel.errorMessage != nil },
                set: { if !$0 { viewModel.errorMessage = nil } }
            )) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(viewModel.errorMessage ?? "")
            }
            .alert("Saved", isPresented: Binding(
                get: { viewModel.saveStatusMessage != nil },
                set: { if !$0 { viewModel.saveStatusMessage = nil } }
            )) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(viewModel.saveStatusMessage ?? "")
            }
        }
    }

    // MARK: - Result Comparison Section

    @ViewBuilder
    private func resultComparisonSection(resultImage: UIImage) -> some View {
        VStack(spacing: 12) {
            // Accessible Segmented Control (not drag-only)
            Picker("Comparison View", selection: $viewModel.comparisonMode) {
                ForEach(ComparisonMode.allCases) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .accessibilityLabel("Before and After Comparison")
            .disabled(viewModel.isProcessing)

            // Image Display
            ZStack {
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(uiColor: .secondarySystemBackground))
                    .aspectRatio(1.0, contentMode: .fit)

                if viewModel.comparisonMode == .before, let target = viewModel.targetImage {
                    Image(uiImage: target)
                        .resizable()
                        .scaledToFit()
                        .cornerRadius(12)
                        .accessibilityLabel("Original target image before processing")
                } else {
                    Image(uiImage: resultImage)
                        .resizable()
                        .scaledToFit()
                        .cornerRadius(12)
                        .accessibilityLabel("Processed result image")
                }
            }
            .frame(maxHeight: 360)
            .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: viewModel.comparisonMode)

            // Save and Share Action Buttons
            HStack(spacing: 16) {
                Button {
                    viewModel.saveResultToPhotos()
                } label: {
                    Label("Save to Photos", systemImage: "square.and.arrow.down")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .frame(maxWidth: .infinity, minHeight: 44)
                        .background(Color(uiColor: .secondarySystemBackground))
                        .foregroundColor(.primary)
                        .cornerRadius(10)
                }
                .accessibilityHint("Saves full resolution PNG to your Photos library")
                .disabled(viewModel.isProcessing)

                if let shareURL = viewModel.tempShareURL {
                    ShareLink(item: shareURL) {
                        Label("Share", systemImage: "square.and.arrow.up")
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .frame(maxWidth: .infinity, minHeight: 44)
                            .background(Color(uiColor: .secondarySystemBackground))
                            .foregroundColor(.primary)
                            .cornerRadius(10)
                    }
                    .accessibilityHint("Opens share sheet for processed full-resolution PNG")
                    .disabled(viewModel.isProcessing)
                } else {
                    Button {} label: {
                        Label("Share", systemImage: "square.and.arrow.up")
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .frame(maxWidth: .infinity, minHeight: 44)
                            .background(Color(uiColor: .secondarySystemBackground))
                            .foregroundColor(.secondary)
                            .cornerRadius(10)
                    }
                    .disabled(true)
                }
            }
        }
        .padding(12)
        .background(Color(uiColor: .systemBackground))
        .cornerRadius(16)
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.purple.opacity(0.3), lineWidth: 1))
    }

    // MARK: - Input Cards Section

    private var inputCardsSection: some View {
        HStack(spacing: 12) {
            // Source Card (Only if faceSwapper is selected)
            if viewModel.requiresSourceImage {
                PhotosPicker(selection: $viewModel.sourceItem, matching: .images) {
                    inputCard(
                        title: "Source Face",
                        subtitle: "Identity",
                        image: viewModel.sourceImage,
                        placeholderIcon: "person.crop.circle.badge.plus",
                        status: viewModel.sourceStatus
                    )
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Select Source Face Photo")
                .disabled(viewModel.isProcessing)
            }

            // Target Card (Always required)
            PhotosPicker(selection: $viewModel.targetItem, matching: .images) {
                inputCard(
                    title: "Target Photo",
                    subtitle: "Canvas",
                    image: viewModel.targetImage,
                    placeholderIcon: "photo.badge.plus",
                    status: viewModel.targetStatus
                )
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Select Target Canvas Photo")
            .disabled(viewModel.isProcessing)
        }
    }

    private func inputCard(
        title: String,
        subtitle: String,
        image: UIImage?,
        placeholderIcon: String,
        status: FaceStatus
    ) -> some View {
        VStack(spacing: 8) {
            ZStack {
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(uiColor: .secondarySystemBackground))
                    .frame(height: 140)

                if let img = image {
                    Image(uiImage: img)
                        .resizable()
                        .scaledToFill()
                        .frame(height: 140)
                        .clipped()
                        .cornerRadius(12)
                } else {
                    VStack(spacing: 6) {
                        Image(systemName: placeholderIcon)
                            .font(.system(size: 32))
                            .foregroundColor(.purple)
                        Text("Tap to Select")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }

            VStack(spacing: 2) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(.primary)
                Text(subtitle)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            // Face Validation Badge
            faceStatusBadge(status: status)
        }
        .frame(maxWidth: .infinity)
        .padding(8)
        .background(Color(uiColor: .systemBackground))
        .cornerRadius(14)
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color.secondary.opacity(0.2), lineWidth: 1))
    }

    @ViewBuilder
    private func faceStatusBadge(status: FaceStatus) -> some View {
        switch status {
        case .idle:
            Text("Ready")
                .font(.caption2)
                .foregroundColor(.secondary)
        case .detecting:
            HStack(spacing: 4) {
                ProgressView()
                    .scaleEffect(0.6)
                Text("Detecting...")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        case .valid(let count):
            HStack(spacing: 4) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
                Text("\(count) face")
                    .font(.caption2)
                    .foregroundColor(.green)
            }
        case .failed(let reason):
            HStack(spacing: 4) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.red)
                Text(reason)
                    .font(.caption2)
                    .foregroundColor(.red)
                    .lineLimit(1)
            }
        }
    }

    // MARK: - Multi-Select Processors Section

    private var processorSelectionSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Active Processors")
                    .font(.headline)
                    .foregroundColor(.primary)
                Spacer()
                Text("\(viewModel.selectedProcessors.count) selected")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            // Processor Chips Flow / Grid
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 8)], spacing: 8) {
                processorChip(kind: .faceSwapper, icon: "person.2.swap", title: "Face Swapper")
                processorChip(kind: .deepSwapper, icon: "cpu", title: "Deep Swapper")
                processorChip(kind: .ageModifier, icon: "calendar", title: "Age Modifier")
                processorChip(kind: .expressionRestorer, icon: "face.smiling", title: "Expression")
                processorChip(kind: .faceEditor, icon: "slider.horizontal.2.square", title: "Face Editor")
                processorChip(kind: .faceEnhancer, icon: "sparkles", title: "Face Enhancer")
                processorChip(kind: .frameColorizer, icon: "paintpalette", title: "Colorizer")
                processorChip(kind: .frameEnhancer, icon: "arrow.up.left.and.arrow.down.right.magnifyingglass", title: "4x Upscaler")
                processorChip(kind: .backgroundRemover, icon: "person.crop.artframe", title: "Bg Remover")
                processorChip(kind: .faceDebugger, icon: "ladybug", title: "Face Debugger")
            }

            // 4x Upscale memory warning banner
            if viewModel.selectedProcessors.contains(.frameEnhancer) {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                    Text("4x Upscale increases peak memory footprint. Default preserves resolution.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(8)
                .background(Color.orange.opacity(0.1))
                .cornerRadius(8)
            }

            // Deep Swapper DFM import button
            if viewModel.selectedProcessors.contains(.deepSwapper) {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Deep Swapper Model")
                            .font(.caption)
                            .fontWeight(.medium)
                        Text(viewModel.importedDFMName ?? viewModel.settings.deepSwapper.model)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                    }
                    Spacer()
                    Button {
                        viewModel.showDFMImporter = true
                    } label: {
                        Label("Import .dfm", systemImage: "doc.badge.plus")
                            .font(.caption)
                            .padding(.horizontal, 10)
                            .frame(minHeight: 44)
                            .background(Color.purple.opacity(0.15))
                            .foregroundColor(.purple)
                            .cornerRadius(8)
                    }
                    .disabled(viewModel.isProcessing)
                }
                .padding(8)
                .background(Color(uiColor: .secondarySystemBackground))
                .cornerRadius(8)
            }
        }
        .padding(12)
        .background(Color(uiColor: .systemBackground))
        .cornerRadius(16)
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.secondary.opacity(0.2), lineWidth: 1))
    }

    private func processorChip(kind: ProcessorKind, icon: String, title: String) -> some View {
        let isSelected = viewModel.selectedProcessors.contains(kind)
        return Button {
            viewModel.toggleProcessor(kind)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 14))
                Text(title)
                    .font(.subheadline)
                    .fontWeight(isSelected ? .semibold : .regular)
                    .lineLimit(1)
                Spacer(minLength: 0)
                if isSelected {
                    Image(systemName: "checkmark")
                        .font(.system(size: 10, weight: .bold))
                }
            }
            .padding(.horizontal, 10)
            .frame(minHeight: 44)
            .background(isSelected ? Color.purple : Color(uiColor: .secondarySystemBackground))
            .foregroundColor(isSelected ? .white : .primary)
            .cornerRadius(10)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(title), \(isSelected ? "selected" : "not selected")")
        .accessibilityAddTraits(isSelected ? .isSelected : [])
        .disabled(viewModel.isProcessing)
    }

    // MARK: - Safe Area Process CTA

    private var processCTASection: some View {
        VStack(spacing: 8) {
            if viewModel.isProcessing {
                // Active Progress View
                VStack(spacing: 6) {
                    HStack {
                        Text(viewModel.progressMessage)
                            .font(.footnote)
                            .foregroundColor(.primary)
                            .lineLimit(1)
                        Spacer()
                        Text("\(Int(viewModel.progressFraction * 100))%")
                            .font(.footnote)
                            .monospacedDigit()
                            .foregroundColor(.secondary)
                    }

                    ProgressView(value: viewModel.progressFraction)
                        .tint(.purple)

                    Button(role: .cancel) {
                        viewModel.cancelProcessing()
                    } label: {
                        Text(viewModel.progressMessage == "Cancelling..." ? "Cancelling..." : "Cancel Processing")
                            .font(.footnote)
                            .foregroundColor(.red)
                            .frame(minHeight: 36)
                    }
                    .disabled(viewModel.progressMessage == "Cancelling...")
                }
                .padding(12)
                .background(Color(uiColor: .systemBackground))
                .cornerRadius(12)
                .shadow(color: Color.black.opacity(0.1), radius: 8, x: 0, y: 4)
            } else {
                // Primary Process Button
                Button {
                    viewModel.process()
                } label: {
                    HStack(spacing: 8) {
                        Image(systemName: "wand.and.stars")
                            .font(.system(size: 18, weight: .semibold))
                        Text("Process Photo")
                            .font(.headline)
                            .fontWeight(.bold)
                    }
                    .frame(maxWidth: .infinity, minHeight: 50)
                    .background(viewModel.canProcess ? Color.purple : Color.purple.opacity(0.4))
                    .foregroundColor(.white)
                    .cornerRadius(14)
                }
                .disabled(!viewModel.canProcess || viewModel.isProcessing)
                .accessibilityLabel("Process Photo")
                .accessibilityHint(viewModel.canProcess ? "Executes selected processors on target photo" : "Select required photos to enable processing")
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(
            Color(uiColor: .systemBackground)
                .opacity(0.95)
                .ignoresSafeArea(edges: .bottom)
        )
    }
}
