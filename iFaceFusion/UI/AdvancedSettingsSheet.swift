import SwiftUI

public struct AdvancedSettingsSheet: View {
    @Binding public var settings: ProcessorSettings
    @Environment(\.dismiss) private var dismiss

    public init(settings: Binding<ProcessorSettings>) {
        self._settings = settings
    }

    public var body: some View {
        NavigationStack {
            Form {
                // MARK: - Face Swapper
                Section(header: Label("Face Swapper", systemImage: "person.2.swap")) {
                    Picker("Model", selection: $settings.faceSwapper.model) {
                        Text("HyperSwap 1a (256px)").tag("hyperswap_1a_256")
                        Text("InSwapper (128px)").tag("inswapper_128")
                    }
                    .accessibilityLabel("Face Swapper Model")

                    Picker("Pixel Boost", selection: $settings.faceSwapper.pixelBoost) {
                        Text("256x256").tag("256x256")
                        Text("512x512").tag("512x512")
                        Text("1024x1024").tag("1024x1024")
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Swap Weight")
                            Spacer()
                            Text(String(format: "%.2f", settings.faceSwapper.weight))
                                .foregroundColor(.secondary)
                        }
                        Slider(value: $settings.faceSwapper.weight, in: 0.0...1.0, step: 0.05)
                            .accessibilityLabel("Face Swapper Weight")
                    }
                    .frame(minHeight: 44)
                }

                // MARK: - Face Enhancer
                Section(header: Label("Face Enhancer", systemImage: "sparkles")) {
                    Picker("Model", selection: $settings.faceEnhancer.model) {
                        Text("GFPGAN 1.4").tag("gfpgan_1.4")
                        Text("CodeFormer").tag("codeformer")
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Blend")
                            Spacer()
                            Text("\(settings.faceEnhancer.blend)%")
                                .foregroundColor(.secondary)
                        }
                        Slider(value: Binding(
                            get: { Double(settings.faceEnhancer.blend) },
                            set: { settings.faceEnhancer.blend = Int($0) }
                        ), in: 0...100, step: 1)
                        .accessibilityLabel("Face Enhancer Blend")
                    }
                    .frame(minHeight: 44)

                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Weight")
                            Spacer()
                            Text(String(format: "%.2f", settings.faceEnhancer.weight))
                                .foregroundColor(.secondary)
                        }
                        Slider(value: $settings.faceEnhancer.weight, in: 0.0...1.0, step: 0.05)
                            .accessibilityLabel("Face Enhancer Weight")
                    }
                    .frame(minHeight: 44)
                }

                // MARK: - Frame Enhancer (Upscaler)
                Section(header: Label("Frame Enhancer (Upscaler)", systemImage: "arrow.up.left.and.arrow.down.right.magnifyingglass")) {
                    Picker("Model", selection: $settings.frameEnhancer.model) {
                        Text("SPAN Kendata 4x").tag("span_kendata_x4")
                        Text("Real-ESRGAN 4x").tag("real_esrgan_x4")
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Blend")
                            Spacer()
                            Text("\(settings.frameEnhancer.blend)%")
                                .foregroundColor(.secondary)
                        }
                        Slider(value: Binding(
                            get: { Double(settings.frameEnhancer.blend) },
                            set: { settings.frameEnhancer.blend = Int($0) }
                        ), in: 0...100, step: 1)
                        .accessibilityLabel("Frame Enhancer Blend")
                    }
                    .frame(minHeight: 44)

                    // Explicit 4x memory warning
                    HStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                        Text("4x Upscale increases memory usage significantly. May encounter memory limits on large images.")
                            .font(.footnote)
                            .foregroundColor(.secondary)
                    }
                    .padding(.vertical, 4)
                }

                // MARK: - Frame Colorizer
                Section(header: Label("Frame Colorizer", systemImage: "paintpalette")) {
                    Picker("Model", selection: $settings.frameColorizer.model) {
                        Text("DDColor").tag("ddcolor")
                    }

                    Picker("Inference Size", selection: $settings.frameColorizer.size) {
                        Text("256x256").tag("256x256")
                        Text("512x512").tag("512x512")
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Blend")
                            Spacer()
                            Text("\(settings.frameColorizer.blend)%")
                                .foregroundColor(.secondary)
                        }
                        Slider(value: Binding(
                            get: { Double(settings.frameColorizer.blend) },
                            set: { settings.frameColorizer.blend = Int($0) }
                        ), in: 0...100, step: 1)
                        .accessibilityLabel("Frame Colorizer Blend")
                    }
                    .frame(minHeight: 44)
                }

                // MARK: - Age Modifier
                Section(header: Label("Age Modifier", systemImage: "calendar")) {
                    Picker("Model", selection: $settings.ageModifier.model) {
                        Text("FRAN (Disney Research)").tag("fran")
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Age Direction")
                            Spacer()
                            Text("\(settings.ageModifier.direction > 0 ? "+" : "")\(settings.ageModifier.direction)")
                                .foregroundColor(.secondary)
                        }
                        Slider(value: Binding(
                            get: { Double(settings.ageModifier.direction) },
                            set: { settings.ageModifier.direction = Int($0) }
                        ), in: -100...100, step: 5)
                        .accessibilityLabel("Age Direction")
                    }
                    .frame(minHeight: 44)
                }

                // MARK: - Expression Restorer
                Section(header: Label("Expression Restorer", systemImage: "face.smiling")) {
                    Picker("Model", selection: $settings.expressionRestorer.model) {
                        Text("LivePortrait").tag("live_portrait")
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Factor")
                            Spacer()
                            Text("\(settings.expressionRestorer.factor)%")
                                .foregroundColor(.secondary)
                        }
                        Slider(value: Binding(
                            get: { Double(settings.expressionRestorer.factor) },
                            set: { settings.expressionRestorer.factor = Int($0) }
                        ), in: 0...100, step: 5)
                        .accessibilityLabel("Expression Restoration Factor")
                    }
                    .frame(minHeight: 44)
                }

                // MARK: - Face Editor
                Section(header: Label("Face Editor (LivePortrait)", systemImage: "slider.horizontal.2.square")) {
                    editorSlider(label: "Smile", value: $settings.faceEditor.mouthSmile)
                    editorSlider(label: "Eyebrows", value: $settings.faceEditor.eyebrowDirection)
                    editorSlider(label: "Eye Open", value: $settings.faceEditor.eyeOpenRatio)
                    editorSlider(label: "Lip Open", value: $settings.faceEditor.lipOpenRatio)
                    editorSlider(label: "Gaze Horizontal", value: $settings.faceEditor.eyeGazeHorizontal)
                    editorSlider(label: "Gaze Vertical", value: $settings.faceEditor.eyeGazeVertical)
                    editorSlider(label: "Head Pitch", value: $settings.faceEditor.headPitch)
                    editorSlider(label: "Head Yaw", value: $settings.faceEditor.headYaw)
                    editorSlider(label: "Head Roll", value: $settings.faceEditor.headRoll)
                }

                // MARK: - Deep Swapper
                Section(header: Label("Deep Swapper", systemImage: "cpu")) {
                    Text("Model: \(settings.deepSwapper.model)")
                        .font(.footnote)
                        .foregroundColor(.secondary)

                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Morph")
                            Spacer()
                            Text("\(settings.deepSwapper.morph)%")
                                .foregroundColor(.secondary)
                        }
                        Slider(value: Binding(
                            get: { Double(settings.deepSwapper.morph) },
                            set: { settings.deepSwapper.morph = Int($0) }
                        ), in: 0...100, step: 5)
                        .accessibilityLabel("Deep Swapper Morph")
                    }
                    .frame(minHeight: 44)
                }

                // MARK: - Face Mask & Blending
                Section(header: Label("Face Mask", systemImage: "theatermasks")) {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Mask Blur")
                            Spacer()
                            Text(String(format: "%.2f", settings.mask.blur))
                                .foregroundColor(.secondary)
                        }
                        Slider(value: $settings.mask.blur, in: 0.0...1.0, step: 0.05)
                            .accessibilityLabel("Face Mask Blur")
                    }
                    .frame(minHeight: 44)

                    Toggle("Box Mask", isOn: maskTypeBinding(.box))
                    Toggle("Occlusion Mask", isOn: maskTypeBinding(.occlusion))
                    Toggle("Region Mask", isOn: maskTypeBinding(.region))
                }

                // MARK: - Face Debugger
                Section(header: Label("Face Debugger", systemImage: "ladybug")) {
                    Toggle("Bounding Box", isOn: debuggerItemBinding(.boundingBox))
                    Toggle("Face Mask Overlay", isOn: debuggerItemBinding(.faceMask))
                    Toggle("5-Point Landmarks", isOn: debuggerItemBinding(.landmark5))
                    Toggle("68-Point Landmarks", isOn: debuggerItemBinding(.landmark68))
                }

                // Reset Action
                Section {
                    Button(role: .destructive) {
                        settings = ProcessorSettings()
                    } label: {
                        HStack {
                            Spacer()
                            Text("Reset All Settings to Defaults")
                            Spacer()
                        }
                    }
                    .frame(minHeight: 44)
                }
            }
            .navigationTitle("Advanced Controls")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                        .fontWeight(.semibold)
                        .frame(minWidth: 44, minHeight: 44)
                }
            }
        }
    }

    private func editorSlider(label: String, value: Binding<Float>) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                Spacer()
                Text(String(format: "%.2f", value.wrappedValue))
                    .foregroundColor(.secondary)
            }
            Slider(value: value, in: -1.0...1.0, step: 0.05)
                .accessibilityLabel("Face Editor \(label)")
        }
        .frame(minHeight: 44)
    }

    private func maskTypeBinding(_ type: FaceMaskType) -> Binding<Bool> {
        Binding(
            get: { settings.mask.types.contains(type) },
            set: { isSelected in
                if isSelected {
                    settings.mask.types.insert(type)
                } else {
                    settings.mask.types.remove(type)
                }
            }
        )
    }

    private func debuggerItemBinding(_ item: DebuggerItem) -> Binding<Bool> {
        Binding(
            get: { settings.faceDebugger.items.contains(item) },
            set: { isSelected in
                if isSelected {
                    settings.faceDebugger.items.insert(item)
                } else {
                    settings.faceDebugger.items.remove(item)
                }
            }
        )
    }
}
