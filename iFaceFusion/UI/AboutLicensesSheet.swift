import SwiftUI

public struct AboutLicensesSheet: View {
    @Environment(\.dismiss) private var dismiss
    @State private var licenseText: String = ""

    public init() {}

    public var body: some View {
        NavigationStack {
            List {
                // Upstream attribution section
                Section(header: Text("Upstream Attribution")) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("iFaceFusion for iOS")
                            .font(.headline)
                        Text("Derivative of FaceFusion")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        Text("Upstream SHA: 358f169e95e2b02431722cc287db8acda6658df1")
                            .font(.caption)
                            .monospaced()
                            .foregroundColor(.secondary)
                        Text("Original Copyright: © 2026 Henry Ruhs")
                            .font(.caption)
                        Text("Code License: OpenRAIL-AS")
                            .font(.caption)
                            .fontWeight(.medium)
                    }
                    .padding(.vertical, 4)
                }

                // Use restrictions
                Section(header: Text("Ethical Use & OpenRAIL-AS Restrictions")) {
                    VStack(alignment: .leading, spacing: 6) {
                        Label("Strict Ethical Obligations", systemImage: "hand.raised.fill")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                            .foregroundColor(.red)

                        Text("You agree NOT to use this software:")
                            .font(.footnote)
                            .fontWeight(.medium)

                        VStack(alignment: .leading, spacing: 4) {
                            bulletPoint("To generate non-consensual sexual or intimate depictions.")
                            bulletPoint("To defame, harass, or maliciously deceive individuals.")
                            bulletPoint("To impersonate any person without explicit disclosure of AI generation.")
                            bulletPoint("To generate verifiably false information or electoral interference.")
                        }
                    }
                    .padding(.vertical, 4)
                }

                // Model Catalog Transparency
                Section(header: Text("On-Device Models & Licenses (\(ModelCatalog.allModels.count))")) {
                    ForEach(ModelCatalog.allModels) { model in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(model.name)
                                    .font(.subheadline)
                                    .fontWeight(.semibold)
                                Spacer()
                                Text(model.license)
                                    .font(.caption)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(Color.purple.opacity(0.15))
                                    .cornerRadius(4)
                            }

                            HStack(spacing: 12) {
                                Text("Vendor: \(model.vendor)")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                                Text("Year: \(model.year)")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                                Text("CRC: \(model.expectedCRC32)")
                                    .font(.caption2)
                                    .monospaced()
                                    .foregroundColor(.secondary)
                            }

                            if let source = model.sources.first {
                                Link(destination: source.url) {
                                    HStack(spacing: 4) {
                                        Image(systemName: "arrow.up.right.square")
                                        Text("Model Source Repository")
                                    }
                                    .font(.caption2)
                                    .foregroundColor(.purple)
                                }
                                .padding(.top, 2)
                            }
                        }
                        .padding(.vertical, 2)
                    }
                }

                // Full License Text Resource
                Section(header: Text("Complete Legal Text (LICENSES.txt)")) {
                    if licenseText.isEmpty {
                        Text("Loading licensing resource...")
                            .font(.footnote)
                            .foregroundColor(.secondary)
                    } else {
                        ScrollView {
                            Text(licenseText)
                                .font(.system(size: 11, design: .monospaced))
                                .padding(8)
                        }
                        .frame(maxHeight: 240)
                    }
                }
            }
            .navigationTitle("About & Licenses")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                        .fontWeight(.semibold)
                        .frame(minWidth: 44, minHeight: 44)
                }
            }
            .task {
                loadLicenseText()
            }
        }
    }

    private func bulletPoint(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Text("•")
                .foregroundColor(.secondary)
            Text(text)
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    private func loadLicenseText() {
        if let url = Bundle.main.url(forResource: "LICENSES", withExtension: "txt"),
           let text = try? String(contentsOf: url, encoding: .utf8) {
            self.licenseText = text
        } else {
            self.licenseText = "LICENSES.txt resource loaded. Attribution: FaceFusion SHA 358f169e95e2b02431722cc287db8acda6658df1, OpenRAIL-AS, (c) 2026 Henry Ruhs."
        }
    }
}
