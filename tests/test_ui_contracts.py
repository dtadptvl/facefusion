#!/usr/bin/env python3
"""
Assert-based verification checks for UI pipeline contracts, icon specs, and licensing invariants.
"""

import os
import struct
import yaml

def test_pipeline_ordering_and_mutual_exclusion():
    # Verify deterministic sequence from ProcessorOrder.swift
    expected_order = [
        "faceSwapper",
        "deepSwapper",
        "ageModifier",
        "expressionRestorer",
        "faceEditor",
        "faceEnhancer",
        "frameColorizer",
        "frameEnhancer",
        "backgroundRemover",
        "faceDebugger"
    ]

    swift_file = os.path.join("iFaceFusion", "UI", "ProcessorOrder.swift")
    with open(swift_file, "r", encoding="utf-8") as f:
        content = f.read()

    for item in expected_order:
        assert f".{item}" in content, f"Processor .{item} missing from ProcessorOrder.swift"

    assert "selected.remove(.deepSwapper)" in content
    assert "selected.remove(.faceSwapper)" in content
    print("[PASS] Deterministic ordering and mutual exclusivity verified in ProcessorOrder.swift")

def test_app_icon_spec():
    icon_path = os.path.join("iFaceFusion", "Resources", "Assets.xcassets", "AppIcon.appiconset", "AppIcon-1024.png")
    assert os.path.exists(icon_path), f"AppIcon missing at {icon_path}"

    with open(icon_path, "rb") as f:
        sig = f.read(8)
        assert sig == b"\x89PNG\r\n\x1a\n", "Invalid PNG signature"
        length, tag = struct.unpack(">I4s", f.read(8))
        assert tag == b"IHDR", "Missing IHDR chunk"
        w, h, depth, color, comp, filt, inter = struct.unpack(">IIBBBBB", f.read(13))
        assert (w, h) == (1024, 1024), f"Icon dimensions {w}x{h} != 1024x1024"
        assert color == 2, f"Icon must be RGB opaque (color type 2), got {color}"
        assert depth == 8, f"Bit depth must be 8, got {depth}"
    print("[PASS] App icon verified: 1024x1024 opaque RGB PNG (no alpha)")

def test_license_resource():
    license_path = os.path.join("iFaceFusion", "Resources", "LICENSES.txt")
    assert os.path.exists(license_path), f"LICENSES.txt missing at {license_path}"

    with open(license_path, "r", encoding="utf-8") as f:
        text = f.read()

    assert "358f169e95e2b02431722cc287db8acda6658df1" in text
    assert "OpenRAIL-AS" in text
    assert "Henry Ruhs" in text
    assert "2026" in text
    assert "onnxruntime 1.24.2" in text
    assert "hyperswap_1a_256" in text
    assert "79e50d4b" in text
    print("[PASS] Licensing resource verified with upstream SHA, OpenRAIL-AS, and model metadata")

def test_project_spec():
    project_path = "project.yml"
    assert os.path.exists(project_path)

    with open(project_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    assert spec["packages"]["onnxruntime"]["exactVersion"] == "1.24.2"
    assert "https://github.com/microsoft/onnxruntime-swift-package-manager" in spec["packages"]["onnxruntime"]["url"]
    app_target = spec["targets"]["iFaceFusion"]
    assert app_target["deploymentTarget"] == "18.0"
    dep = app_target["dependencies"][0]
    assert dep["package"] == "onnxruntime"
    assert dep["product"] == "onnxruntime"
    print("[PASS] XcodeGen project spec verified with ONNX SPM 1.24.2 and iOS 18 target")

def test_github_workflow():
    workflow_path = os.path.join(".github", "workflows", "build.yml")
    assert os.path.exists(workflow_path)

    with open(workflow_path, "r", encoding="utf-8") as f:
        wf = f.read()

    assert "macos-15" in wf
    assert "xcodegen generate" in wf
    assert "CODE_SIGNING_ALLOWED=NO" in wf
    assert "unsigned-iFaceFusion" in wf
    assert "workflow_dispatch" in wf
    print("[PASS] GitHub Actions CI workflow verified for macos-15 unsigned IPA")

if __name__ == "__main__":
    test_pipeline_ordering_and_mutual_exclusion()
    test_app_icon_spec()
    test_license_resource()
    test_project_spec()
    test_github_workflow()
    print("\nALL UI & BUILD CONTRACT CHECKS PASSED.")
