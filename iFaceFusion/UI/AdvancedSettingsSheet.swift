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
                    // ponytail: GFPGAN ONNX model does not accept a weight tensor input (unlike CodeFormer); weight slider omitted to avoid non-functional UI.
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

                // MARK: - Background Remover
                Section(header: Label("Background Remover", systemImage: "person.crop.artframe")) {
                    Picker("Model", selection: $settings.backgroundRemover.model) {
                        Text("MODNet").tag("modnet")
                    }

                    Toggle("Solid Background Fill", isOn: isSolidFillBinding)
                    if isSolidFillBinding.wrappedValue {
                        ColorPicker("Fill Color", selection: fillColorBinding, supportsOpacity: false)
                            .accessibilityLabel("Background Fill Color")
                    } else {
                        Text("Background will be exported with transparent alpha (default).")
                            .font(.footnote)
                            .foregroundColor(.secondary)
                    }

                    Toggle("Despill Color", isOn: isDespillEnabledBinding)
                    if isDespillEnabledBinding.wrappedValue {
                        ColorPicker("Despill Color", selection: despillColorBinding, supportsOpacity: false)
                            .accessibilityLabel("Background Despill Color")
                    }
                }

                // MARK: - Age Modifier
                Section(header: Label("Age Modifier", systemImage: "calendar")) {
                    Stepper("Current age (manual): \(settings.ageModifier.sourceAge)", value: $settings.ageModifier.sourceAge, in: 0...100)
                    Text("Set the target person's current age; this value is not automatically estimated.")
                        .font(.footnote).foregroundStyle(.secondary)
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

                    VStack(alignment: .leading, spacing: 6) {
                        Text("Restoration Areas")
                            .font(.subheadline)
                            .fontWeight(.medium)
                        ForEach(FaceMaskArea.allCases, id: \.self) { area in
                            Toggle(area.rawValue, isOn: expressionAreaBinding(area))
                        }
                    }
                }

                // MARK: - Face Editor (14 LivePortrait Knobs)
                Section(header: Label("Face Editor (LivePortrait)", systemImage: "slider.horizontal.2.square")) {
                    editorSlider(label: "Smile", value: $settings.faceEditor.mouthSmile)
                    editorSlider(label: "Mouth Grim", value: $settings.faceEditor.mouthGrim)
                    editorSlider(label: "Mouth Pout", value: $settings.faceEditor.mouthPout)
                    editorSlider(label: "Mouth Purse", value: $settings.faceEditor.mouthPurse)
                    editorSlider(label: "Mouth Horizontal", value: $settings.faceEditor.mouthPositionHorizontal)
                    editorSlider(label: "Mouth Vertical", value: $settings.faceEditor.mouthPositionVertical)
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

                    Picker("Input Size", selection: $settings.deepSwapper.inputSize) {
                        Text("Auto (Infer from filename)").tag(Int?.none)
                        Text("224x224").tag(Int?.some(224))
                        Text("256x256").tag(Int?.some(256))
                        Text("320x320").tag(Int?.some(320))
                        Text("384x384").tag(Int?.some(384))
                        Text("448x448").tag(Int?.some(448))
                        Text("512x512").tag(Int?.some(512))
                    }
                    .accessibilityLabel("Deep Swapper Input Size")

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

                    VStack(alignment: .leading, spacing: 6) {
                        Text("Mask Types")
                            .font(.subheadline)
                            .fontWeight(.medium)
                        ForEach(FaceMaskType.allCases, id: \.self) { type in
                            Toggle("\(type.rawValue.capitalized) Mask", isOn: maskTypeBinding(type))
                        }
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        Text("Mask Padding (%)")
                            .font(.subheadline)
                            .fontWeight(.medium)
                        Stepper("Top: \(settings.mask.padding.top)%", value: $settings.mask.padding.top, in: 0...100)
                        Stepper("Right: \(settings.mask.padding.right)%", value: $settings.mask.padding.right, in: 0...100)
                        Stepper("Bottom: \(settings.mask.padding.bottom)%", value: $settings.mask.padding.bottom, in: 0...100)
                        Stepper("Left: \(settings.mask.padding.left)%", value: $settings.mask.padding.left, in: 0...100)
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        Text("Mask Areas")
                            .font(.subheadline)
                            .fontWeight(.medium)
                        ForEach(FaceMaskArea.allCases, id: \.self) { area in
                            Toggle(area.rawValue, isOn: maskAreaBinding(area))
                        }
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        Text("Mask Regions")
                            .font(.subheadline)
                            .fontWeight(.medium)
                        ForEach(FaceMaskRegion.allCases, id: \.self) { region in
                            Toggle(region.rawValue, isOn: maskRegionBinding(region))
                        }
                    }
                }

                // MARK: - Face Debugger
                Section(header: Label("Face Debugger", systemImage: "ladybug")) {
                    ForEach(DebuggerItem.allCases, id: \.self) { item in
                        Toggle(item.rawValue, isOn: debuggerItemBinding(item))
                    }
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

    private func maskAreaBinding(_ area: FaceMaskArea) -> Binding<Bool> {
        Binding(
            get: { settings.mask.areas.contains(area) },
            set: { isSelected in
                if isSelected {
                    settings.mask.areas.insert(area)
                } else {
                    settings.mask.areas.remove(area)
                }
            }
        )
    }

    private func maskRegionBinding(_ region: FaceMaskRegion) -> Binding<Bool> {
        Binding(
            get: { settings.mask.regions.contains(region) },
            set: { isSelected in
                if isSelected {
                    settings.mask.regions.insert(region)
                } else {
                    settings.mask.regions.remove(region)
                }
            }
        )
    }

    private func expressionAreaBinding(_ area: FaceMaskArea) -> Binding<Bool> {
        Binding(
            get: { settings.expressionRestorer.areas.contains(area) },
            set: { isSelected in
                if isSelected {
                    settings.expressionRestorer.areas.insert(area)
                } else {
                    settings.expressionRestorer.areas.remove(area)
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

    private var isSolidFillBinding: Binding<Bool> {
        Binding(
            get: {
                settings.backgroundRemover.fillColor.count >= 4 && settings.backgroundRemover.fillColor[3] > 0
            },
            set: { isSolid in
                if isSolid {
                    if settings.backgroundRemover.fillColor.count < 4 || settings.backgroundRemover.fillColor[3] == 0 {
                        settings.backgroundRemover.fillColor = [255, 255, 255, 255]
                    }
                } else {
                    settings.backgroundRemover.fillColor = [0, 0, 0, 0]
                }
            }
        )
    }

    private var fillColorBinding: Binding<Color> {
        Binding(
            get: {
                let rgba = settings.backgroundRemover.fillColor
                if rgba.count >= 4 && rgba[3] > 0 {
                    return Color(
                        red: Double(rgba[0]) / 255.0,
                        green: Double(rgba[1]) / 255.0,
                        blue: Double(rgba[2]) / 255.0
                    )
                }
                return Color.white
            },
            set: { newColor in
                let uiColor = UIColor(newColor)
                var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
                if uiColor.getRed(&r, green: &g, blue: &b, alpha: &a) {
                    settings.backgroundRemover.fillColor = [
                        UInt8(clamping: Int(round(r * 255))),
                        UInt8(clamping: Int(round(g * 255))),
                        UInt8(clamping: Int(round(b * 255))),
                        255
                    ]
                } else {
                    settings.backgroundRemover.fillColor = [255, 255, 255, 255]
                }
            }
        )
    }

    private var isDespillEnabledBinding: Binding<Bool> {
        Binding(
            get: {
                settings.backgroundRemover.despillColor.count >= 4 && settings.backgroundRemover.despillColor[3] > 0
            },
            set: { isEnabled in
                if isEnabled {
                    if settings.backgroundRemover.despillColor.count < 4 || settings.backgroundRemover.despillColor[3] == 0 {
                        settings.backgroundRemover.despillColor = [0, 255, 0, 255]
                    }
                } else {
                    settings.backgroundRemover.despillColor = [0, 0, 0, 0]
                }
            }
        )
    }

    private var despillColorBinding: Binding<Color> {
        Binding(
            get: {
                let rgba = settings.backgroundRemover.despillColor
                if rgba.count >= 4 && rgba[3] > 0 {
                    return Color(
                        red: Double(rgba[0]) / 255.0,
                        green: Double(rgba[1]) / 255.0,
                        blue: Double(rgba[2]) / 255.0
                    )
                }
                return Color.green
            },
            set: { newColor in
                let uiColor = UIColor(newColor)
                var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
                if uiColor.getRed(&r, green: &g, blue: &b, alpha: &a) {
                    settings.backgroundRemover.despillColor = [
                        UInt8(clamping: Int(round(r * 255))),
                        UInt8(clamping: Int(round(g * 255))),
                        UInt8(clamping: Int(round(b * 255))),
                        255
                    ]
                } else {
                    settings.backgroundRemover.despillColor = [0, 255, 0, 255]
                }
            }
        )
    }
}
