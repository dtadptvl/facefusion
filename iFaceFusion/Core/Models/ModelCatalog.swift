import Foundation

/// Catalog of supported models, their licensing, input constraints, and integrity hashes.
public enum ModelCatalog {

    private static func releaseURL(version: String, file: String) -> URL {
        URL(string: "https://github.com/facefusion/facefusion-assets/releases/download/\(version)/\(file)")!
    }

    private static func hfURL(repo: String, file: String) -> URL {
        URL(string: "https://huggingface.co/facefusion/\(repo)/resolve/main/\(file)")!
    }

    // MARK: - Face Swapper Models

    public static let hyperswap1a256 = ModelMetadata(
        id: "hyperswap_1a_256",
        name: "HyperSwap 1a 256",
        processor: .faceSwapper,
        vendor: "FaceFusion",
        license: "ResearchRAIL",
        year: 2025,
        template: .arcface128,
        inputWidth: 256,
        inputHeight: 256,
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
        isBGR: false,
        precision: "fp16",
        sources: [
            DownloadSource(url: releaseURL(version: "models-3.3.0", file: "hyperswap_1a_256.onnx")),
            DownloadSource(url: hfURL(repo: "models-3.3.0", file: "hyperswap_1a_256.onnx"), provider: "huggingface")
        ],
        expectedCRC32: "79e50d4b",
        expectedSHA256: "c0e98a8a03a238f461ed3d2570e426b49f46745ee400854a60dceeb70c246add"
    )

    // MARK: - Face Enhancer Models

    public static let gfpgan14 = ModelMetadata(
        id: "gfpgan_1.4",
        name: "GFPGAN 1.4",
        processor: .faceEnhancer,
        vendor: "TencentARC",
        license: "Apache-2.0",
        year: 2022,
        template: .ffhq512,
        inputWidth: 512,
        inputHeight: 512,
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
        isBGR: false,
        precision: "fp32",
        sources: [
            DownloadSource(url: releaseURL(version: "models-3.0.0", file: "gfpgan_1.4.onnx")),
            DownloadSource(url: hfURL(repo: "models-3.0.0", file: "gfpgan_1.4.onnx"), provider: "huggingface")
        ],
        expectedCRC32: "5a6c6364"
    )

    /// CodeFormer requires DOUBLE scalar weight input not supported by ONNX Runtime Objective-C.
    /// Retained with verified CRC32 but excluded from `allModels` until double tensor support is added.
    public static let codeformer = ModelMetadata(
        id: "codeformer",
        name: "CodeFormer",
        processor: .faceEnhancer,
        vendor: "sczhou",
        license: "S-Lab-1.0",
        year: 2022,
        template: .ffhq512,
        inputWidth: 512,
        inputHeight: 512,
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
        isBGR: false,
        precision: "fp32",
        sources: [
            DownloadSource(url: releaseURL(version: "models-3.0.0", file: "codeformer.onnx")),
            DownloadSource(url: hfURL(repo: "models-3.0.0", file: "codeformer.onnx"), provider: "huggingface")
        ],
        expectedCRC32: "1456f3ab"
    )

    // MARK: - Frame Enhancer Models

    public static let spanKendataX4 = ModelMetadata(
        id: "span_kendata_x4",
        name: "SPAN Kendata x4",
        processor: .frameEnhancer,
        vendor: "terrainer",
        license: "Non-Commercial",
        year: 2024,
        template: nil,
        inputWidth: 128,
        inputHeight: 128,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
        isBGR: false,
        precision: "fp32",
        sources: [
            DownloadSource(url: releaseURL(version: "models-3.0.0", file: "span_kendata_x4.onnx")),
            DownloadSource(url: hfURL(repo: "models-3.0.0", file: "span_kendata_x4.onnx"), provider: "huggingface")
        ],
        expectedCRC32: "a0d53205",
        expectedSHA256: "7478c40953a5902efd785ef1fe8e07a8bb7e7ee9b1977273550813ceefae6348",
        outputScale: 4
    )

    public static let realEsrganX4 = ModelMetadata(
        id: "real_esrgan_x4",
        name: "Real-ESRGAN x4",
        processor: .frameEnhancer,
        vendor: "xinntao",
        license: "BSD-3-Clause",
        year: 2021,
        template: nil,
        inputWidth: 256,
        inputHeight: 256,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
        isBGR: false,
        precision: "fp32",
        sources: [
            DownloadSource(url: releaseURL(version: "models-3.0.0", file: "real_esrgan_x4.onnx")),
            DownloadSource(url: hfURL(repo: "models-3.0.0", file: "real_esrgan_x4.onnx"), provider: "huggingface")
        ],
        expectedCRC32: "9d6e76c4",
        outputScale: 4
    )

    // MARK: - Frame Colorizer Models

    public static let ddcolor = ModelMetadata(
        id: "ddcolor",
        name: "DDColor",
        processor: .frameColorizer,
        vendor: "piddnad",
        license: "Apache-2.0",
        year: 2023,
        template: nil,
        inputWidth: 256,
        inputHeight: 256,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
        isBGR: false,
        precision: "fp32",
        sources: [
            DownloadSource(url: releaseURL(version: "models-3.0.0", file: "ddcolor.onnx")),
            DownloadSource(url: hfURL(repo: "models-3.0.0", file: "ddcolor.onnx"), provider: "huggingface")
        ],
        expectedCRC32: "cab3b659"
    )

    // MARK: - Background Remover Models

    public static let modnet = ModelMetadata(
        id: "modnet",
        name: "MODNet",
        processor: .backgroundRemover,
        vendor: "ZHKKKe",
        license: "Apache-2.0",
        year: 2020,
        template: nil,
        inputWidth: 512,
        inputHeight: 512,
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
        isBGR: false,
        precision: "fp32",
        sources: [
            DownloadSource(url: releaseURL(version: "models-3.5.0", file: "modnet.onnx")),
            DownloadSource(url: hfURL(repo: "models-3.5.0", file: "modnet.onnx"), provider: "huggingface")
        ],
        expectedCRC32: "03a022de",
        expectedSHA256: "a9edce4b47653992aacd1bee48126e65a415ed54e2ecbe51bdca25a8cab0c0d3"
    )

    // MARK: - Age Modifier Models

    public static let fran = ModelMetadata(
        id: "fran",
        name: "FRAN",
        processor: .ageModifier,
        vendor: "ry-lu",
        license: "MIT",
        year: 2024,
        template: .ffhq512,
        inputWidth: 1024,
        inputHeight: 1024,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
        isBGR: false,
        precision: "fp32",
        sources: [
            DownloadSource(url: releaseURL(version: "models-3.6.0", file: "fran.onnx")),
            DownloadSource(url: hfURL(repo: "models-3.6.0", file: "fran.onnx"), provider: "huggingface")
        ],
        expectedCRC32: "14d21511",
        expectedSHA256: "725bb979bda169e0469de31d35ded1f6fcca3413440564d375c92c863570933b"
    )

    // MARK: - LivePortrait Models (Expression Restorer & Face Editor)

    public static let livePortraitFeatureExtractor = ModelMetadata(
        id: "live_portrait_feature_extractor",
        name: "LivePortrait Feature Extractor",
        processor: .faceEditor,
        vendor: "KwaiVGI",
        license: "MIT",
        year: 2024,
        template: .ffhq512,
        inputWidth: 256,
        inputHeight: 256,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
        isBGR: false,
        precision: "fp32",
        sources: [
            DownloadSource(url: releaseURL(version: "models-3.0.0", file: "live_portrait_feature_extractor.onnx")),
            DownloadSource(url: hfURL(repo: "models-3.0.0", file: "live_portrait_feature_extractor.onnx"), provider: "huggingface")
        ],
        expectedCRC32: "b6ca08de",
        expectedSHA256: "980d26afc9af6d1b6c946329df08b89a4a8582c9bb668d9475f5254e69434d15"
    )

    public static let livePortraitMotionExtractor = ModelMetadata(
        id: "live_portrait_motion_extractor",
        name: "LivePortrait Motion Extractor",
        processor: .faceEditor,
        vendor: "KwaiVGI",
        license: "MIT",
        year: 2024,
        template: .ffhq512,
        inputWidth: 256,
        inputHeight: 256,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
        isBGR: false,
        precision: "fp32",
        sources: [
            DownloadSource(url: releaseURL(version: "models-3.0.0", file: "live_portrait_motion_extractor.onnx")),
            DownloadSource(url: hfURL(repo: "models-3.0.0", file: "live_portrait_motion_extractor.onnx"), provider: "huggingface")
        ],
        expectedCRC32: "1278bb27"
    )

    public static let livePortraitGenerator = ModelMetadata(
        id: "live_portrait_generator",
        name: "LivePortrait Generator",
        processor: .faceEditor,
        vendor: "KwaiVGI",
        license: "MIT",
        year: 2024,
        template: .ffhq512,
        inputWidth: 512,
        inputHeight: 512,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
        isBGR: false,
        precision: "fp32",
        sources: [
            DownloadSource(url: releaseURL(version: "models-3.0.0", file: "live_portrait_generator.onnx")),
            DownloadSource(url: hfURL(repo: "models-3.0.0", file: "live_portrait_generator.onnx"), provider: "huggingface")
        ],
        expectedCRC32: "ea09ef95"
    )

    public static let livePortraitEyeRetargeter = ModelMetadata(
        id: "live_portrait_eye_retargeter",
        name: "LivePortrait Eye Retargeter",
        processor: .faceEditor,
        vendor: "KwaiVGI",
        license: "MIT",
        year: 2024,
        template: nil,
        inputWidth: 66,
        inputHeight: 1,
        mean: [0, 0, 0],
        std: [1, 1, 1],
        isBGR: false,
        precision: "fp32",
        sources: [
            DownloadSource(url: releaseURL(version: "models-3.0.0", file: "live_portrait_eye_retargeter.onnx")),
            DownloadSource(url: hfURL(repo: "models-3.0.0", file: "live_portrait_eye_retargeter.onnx"), provider: "huggingface")
        ],
        expectedCRC32: "ec641c99",
        expectedSHA256: "13cf8b06f0a314e6e65b4a47fea5ce270edb79cd54359c4d273f2c139820276b"
    )

    public static let livePortraitLipRetargeter = ModelMetadata(
        id: "live_portrait_lip_retargeter",
        name: "LivePortrait Lip Retargeter",
        processor: .faceEditor,
        vendor: "KwaiVGI",
        license: "MIT",
        year: 2024,
        template: nil,
        inputWidth: 65,
        inputHeight: 1,
        mean: [0, 0, 0],
        std: [1, 1, 1],
        isBGR: false,
        precision: "fp32",
        sources: [
            DownloadSource(url: releaseURL(version: "models-3.0.0", file: "live_portrait_lip_retargeter.onnx")),
            DownloadSource(url: hfURL(repo: "models-3.0.0", file: "live_portrait_lip_retargeter.onnx"), provider: "huggingface")
        ],
        expectedCRC32: "8dc8828b",
        expectedSHA256: "0f5b76344d0bfbb44c8a0b1ec57a4a7c36fd38db530338777a48c4a33a02d71f"
    )

    public static let livePortraitStitcher = ModelMetadata(
        id: "live_portrait_stitcher",
        name: "LivePortrait Stitcher",
        processor: .faceEditor,
        vendor: "KwaiVGI",
        license: "MIT",
        year: 2024,
        template: nil,
        inputWidth: 126,
        inputHeight: 1,
        mean: [0, 0, 0],
        std: [1, 1, 1],
        isBGR: false,
        precision: "fp32",
        sources: [
            DownloadSource(url: releaseURL(version: "models-3.0.0", file: "live_portrait_stitcher.onnx")),
            DownloadSource(url: hfURL(repo: "models-3.0.0", file: "live_portrait_stitcher.onnx"), provider: "huggingface")
        ],
        expectedCRC32: "c0a89576",
        expectedSHA256: "c6683427edf3e0c3a86e61b1dea1e6dad56c557f5e711fec93966cfb34fd5f87"
    )

    // MARK: - Essential Auxiliary Models

    public static let arcfaceW600kR50 = ModelMetadata(
        id: "arcface_w600k_r50",
        name: "ArcFace w600k r50",
        processor: nil,
        vendor: "InsightFace",
        license: "Non-Commercial",
        year: 2018,
        template: .arcface112v2,
        inputWidth: 112,
        inputHeight: 112,
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
        isBGR: false,
        precision: "fp32",
        sources: [
            DownloadSource(url: releaseURL(version: "models-3.0.0", file: "arcface_w600k_r50.onnx")),
            DownloadSource(url: hfURL(repo: "models-3.0.0", file: "arcface_w600k_r50.onnx"), provider: "huggingface")
        ],
        expectedCRC32: "1f5fefb8"
    )

    public static let xseg1 = ModelMetadata(
        id: "xseg_1",
        name: "XSeg Face Occluder",
        processor: nil,
        vendor: "DeepFaceLab",
        license: "GPL-3.0",
        year: 2021,
        template: nil,
        inputWidth: 256,
        inputHeight: 256,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
        isBGR: false,
        precision: "fp32",
        sources: [
            DownloadSource(url: releaseURL(version: "models-3.1.0", file: "xseg_1.onnx")),
            DownloadSource(url: hfURL(repo: "models-3.1.0", file: "xseg_1.onnx"), provider: "huggingface")
        ],
        expectedCRC32: "f207afe3"
    )

    public static let bisenetResnet18 = ModelMetadata(
        id: "bisenet_resnet_18",
        name: "BiSeNet ResNet-18 Face Parser",
        processor: nil,
        vendor: "yakhyo",
        license: "MIT",
        year: 2024,
        template: nil,
        inputWidth: 512,
        inputHeight: 512,
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
        isBGR: true,
        precision: "fp32",
        sources: [
            DownloadSource(url: releaseURL(version: "models-3.1.0", file: "bisenet_resnet_18.onnx")),
            DownloadSource(url: hfURL(repo: "models-3.1.0", file: "bisenet_resnet_18.onnx"), provider: "huggingface")
        ],
        expectedCRC32: "6cafc877"
    )

    // MARK: - Registry

    /// Verified offered catalog models. Speculative models lacking required embeddings
    /// and models with unsupported data types (such as CodeFormer requiring DOUBLE) are omitted.
    public static let allModels: [ModelMetadata] = [
        hyperswap1a256,
        gfpgan14,
        spanKendataX4,
        realEsrganX4,
        ddcolor,
        modnet,
        fran,
        livePortraitFeatureExtractor,
        livePortraitMotionExtractor,
        livePortraitGenerator,
        livePortraitEyeRetargeter,
        livePortraitLipRetargeter,
        livePortraitStitcher,
        arcfaceW600kR50,
        xseg1,
        bisenetResnet18
    ]

    public static func model(for id: String) -> ModelMetadata? {
        allModels.first { $0.id == id }
    }
}
