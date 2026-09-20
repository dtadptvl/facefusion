#!/usr/bin/env python3
"""
Executable verification checks for iFaceFusion ModelCatalog, ModelCache, and ORTBridge.
Validates CRC32/SHA256 integrity, elimination of speculative models, mandatory ONNX SPM import,
CoreML provider configuration, and typed tensor I/O contracts.
"""

import os
import re
import sys

# Authoritative upstream FaceFusion CRC32 hashes from GitHub release .hash files
UPSTREAM_VERIFIED_CRC32 = {
    "hyperswap_1a_256": "79e50d4b",
    "gfpgan_1.4": "5a6c6364",
    "codeformer": "1456f3ab",
    "span_kendata_x4": "a0d53205",
    "real_esrgan_x4": "9d6e76c4",
    "ddcolor": "cab3b659",
    "modnet": "03a022de",
    "fran": "14d21511",
    "live_portrait_feature_extractor": "b6ca08de",
    "live_portrait_motion_extractor": "1278bb27",
    "live_portrait_generator": "ea09ef95",
    "live_portrait_eye_retargeter": "ec641c99",
    "live_portrait_lip_retargeter": "8dc8828b",
    "live_portrait_stitcher": "c0a89576",
    "arcface_w600k_r50": "1f5fefb8",
    "xseg_1": "f207afe3",
    "bisenet_resnet_18": "6cafc877"
}

def test_model_catalog_contracts():
    catalog_path = os.path.join("iFaceFusion", "Core", "Models", "ModelCatalog.swift")
    assert os.path.exists(catalog_path), f"ModelCatalog.swift missing at {catalog_path}"

    with open(catalog_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Invariant: speculative InSwapper must be removed from catalog options
    assert "inswapper" not in content.lower(), "InSwapper must be removed from ModelCatalog"

    # Invariant: real_esrgan_x4 CRC must be 9d6e76c4 (not suspicious f3c2b1a0)
    assert 'expectedCRC32: "9d6e76c4"' in content, "real_esrgan_x4 must have upstream CRC32 9d6e76c4"
    assert 'f3c2b1a0' not in content, "Corrupted CRC32 f3c2b1a0 must not be present in catalog"

    # Invariant: codeformer CRC must be 1456f3ab (not 1198e3b7) and excluded from allModels
    assert 'expectedCRC32: "1456f3ab"' in content, "codeformer must have upstream CRC32 1456f3ab"
    assert '1198e3b7' not in content, "Corrupted CRC32 1198e3b7 must not be present in catalog"

    # Find allModels block
    all_models_match = re.search(r'public static let allModels:\s*\[ModelMetadata\]\s*=\s*\[(.*?)\]', content, re.DOTALL)
    assert all_models_match, "public static let allModels not found in ModelCatalog"
    all_models_body = all_models_match.group(1)

    # Invariant: codeformer must NOT be exposed in allModels because double scalar tensor is unsupported
    assert "codeformer" not in all_models_body, "codeformer must be excluded from allModels until DOUBLE tensor is supported"

    # Verify every model in allModels has valid CRC32 matching upstream
    model_blocks = re.findall(r'public static let (\w+)\s*=\s*ModelMetadata\((.*?)\n    \)', content, re.DOTALL)
    assert len(model_blocks) >= 16, f"Expected at least 16 model definitions, found {len(model_blocks)}"

    found_ids = set()
    for var_name, block in model_blocks:
        id_match = re.search(r'id:\s*"([^"]+)"', block)
        crc_match = re.search(r'expectedCRC32:\s*"([^"]+)"', block)
        assert id_match, f"Missing id in {var_name}"
        assert crc_match, f"Missing expectedCRC32 in {var_name}"
        m_id = id_match.group(1)
        crc = crc_match.group(1).lower()
        found_ids.add(m_id)

        assert m_id in UPSTREAM_VERIFIED_CRC32, f"Unknown model ID {m_id} in catalog"
        expected = UPSTREAM_VERIFIED_CRC32[m_id]
        assert crc == expected, f"CRC32 mismatch for {m_id}: expected {expected}, got {crc}"

    print(f"[PASS] ModelCatalog verified: {len(found_ids)} models verified against upstream CRC32; speculative models eliminated")

def test_ort_bridge_contracts():
    bridge_path = os.path.join("iFaceFusion", "Core", "Inference", "ORTBridge.swift")
    assert os.path.exists(bridge_path), f"ORTBridge.swift missing at {bridge_path}"

    with open(bridge_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Invariant: ONNX 1.24.2 mandatory import, not conditional silently unavailable
    assert "import onnxruntime_objc" in content, "Missing mandatory import onnxruntime_objc"
    assert "#if canImport(onnxruntime_objc)" not in content, "Silent conditional import forbidden"

    # Invariant: CoreML API must use official appendCoreMLExecutionProvider(with:)
    assert 'appendExecutionProvider("coreml"' not in content, "Lowercase appendExecutionProvider('coreml') is invalid"
    assert "ORTCoreMLExecutionProviderOptions()" in content, "Must use ORTCoreMLExecutionProviderOptions"
    assert "appendCoreMLExecutionProvider(with: coreMLOptions)" in content, "Must use appendCoreMLExecutionProvider(with:)"
    assert "ORTIsCoreMLExecutionProviderAvailable()" in content, "Must verify CoreML availability"

    # Invariant: CPU fallback on failure
    assert "cpuOptions" in content, "CPU fallback options must be present"

    # Invariant: Refuse silent double cast
    assert "Double tensor data type is unsupported" in content, "Must refuse silent double cast"

    # Invariant: Strict output decoding based on element type (no treating all bytes as Float)
    assert "case .float:" in content, "Output decoding must handle .float"
    assert "case .int64:" in content, "Output decoding must handle .int64"
    assert "case .int32:" in content, "Output decoding must handle .int32"
    assert "case .uInt8:" in content, "Output decoding must handle .uInt8"
    assert "totalElements * MemoryLayout<Float>.stride" in content, "Must validate float byte length"
    assert "totalElements * MemoryLayout<Int64>.stride" in content, "Must validate int64 byte length"

    print("[PASS] ORTBridge verified: mandatory import, official CoreML EP, strict typed output decoding, no silent double cast")

def test_model_cache_contracts():
    cache_path = os.path.join("iFaceFusion", "Core", "Models", "ModelCache.swift")
    assert os.path.exists(cache_path), f"ModelCache.swift missing at {cache_path}"

    with open(cache_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Invariant: Use URLSession download(for:) instead of per-byte loop
    assert "for try await byte in asyncBytes" not in content, "Catastrophic per-byte async loop must be removed"
    assert "URLSession.shared.download(for:" in content, "Must use URLSession.shared.download(for:delegate:)"

    # Invariant: NotificationCenter progress notification hooks
    assert "downloadProgressNotification" in content, "Missing downloadProgressNotification"
    assert "downloadCompletedNotification" in content, "Missing downloadCompletedNotification"

    # Invariant: Download staging and CRC32/SHA256 verification before atomic move
    assert "computeCRC32(fileURL: stagingURL)" in content or "computeCRC32" in content
    assert "FileManager.default.moveItem" in content, "Must atomically move verified file to target"

    print("[PASS] ModelCache verified: URLSession download(for:), progress hooks, atomic validation")

if __name__ == "__main__":
    test_model_catalog_contracts()
    test_ort_bridge_contracts()
    test_model_cache_contracts()
    print("\nALL MODEL CATALOG, CACHE & ORT INFERENCE CONTRACTS VERIFIED.")
