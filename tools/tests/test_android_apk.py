import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from embeddinggemma_tools.android import audit_apk, main
from embeddinggemma_tools.player import MODEL_REVISION

COMMIT = "a" * 40
PREFIX = "assets/EmbeddingGemmaValidation/"
FILES = ("model-fp32.sentis", "model-float16.sentis", "tokenizer.json", "reference.json", "search-reference.json")


def inputs(tmp_path):
    payload = {name: ("fixture " + name).encode() for name in FILES}
    receipt = {"success": True, "model_revision": MODEL_REVISION, "model_source": "b" * 40,
               "search_source": "c" * 40, "model_origin_sha256": "d" * 64,
               "unity_preparation_version": "6000.3.16f1",
               "files": [{"name": name, "bytes": len(payload[name]), "sha256": hashlib.sha256(payload[name]).hexdigest()} for name in FILES]}
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps(receipt), encoding="utf-8")
    build = tmp_path / "build.json"
    build_data = {"success": True, "injected_build": False, "source_commit": COMMIT,
                  "build_result": "Succeeded", "errors": 0, "build_options": "Development",
                  "il2cpp_compiler": "Release", "build_info": {"codeCommit": COMMIT, "target": "Android",
                      "unityVersion": "6000.3.16f1", "sentisVersion": "2.6.1", "scriptingBackend": "IL2CPP",
                      "stripping": "High", "sourceSha256": {"runtime.cs": "e" * 64}},
                  "target_settings": {"architectures": "ARM64", "application_identifier": "com.ayutaz.embeddinggemma.validation",
                      "min_sdk": 26, "target_sdk": "AndroidApiLevelAuto", "custom_keystore": False,
                      "apk_expansion": False, "apk_per_cpu": False, "app_bundle": False, "export_project": False,
                      "automatic_graphics_api": False, "graphics_apis": ["Vulkan", "OpenGLES3"]}}
    build.write_text(json.dumps(build_data), encoding="utf-8")
    entries = {PREFIX + name: value for name, value in payload.items()}
    entries[PREFIX + "bundle.json"] = bundle.read_bytes()
    entries.update({"lib/arm64-v8a/libil2cpp.so": b"native fixture", "lib/arm64-v8a/libunity.so": b"native fixture"})
    apk = tmp_path / "Validation.apk"
    return apk, bundle, build, entries, build_data


def write_apk(path, entries):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)


def test_audits_all_selected_apk_bytes_and_never_claims_runtime_execution(tmp_path):
    apk, bundle, build, entries, _ = inputs(tmp_path)
    write_apk(apk, entries)
    result = audit_apk(apk, bundle, build, COMMIT)
    assert result["success"] is True and result["apk_payload_verified"] is True
    assert result["unity_runtime_executed"] is False and result["gpu_verified"] is False
    assert result["source_commit"] == COMMIT
    assert [row["name"] for row in result["files"]] == list(FILES)
    assert result["abis"] == ["arm64-v8a"]
    assert result["apk_sha256"] == hashlib.sha256(apk.read_bytes()).hexdigest()


@pytest.mark.parametrize("damage", ["corruption", "missing", "extra_model", "extra_abi", "missing_native", "manifest"])
def test_rejects_changed_missing_or_duplicate_model_layout_before_success(tmp_path, damage):
    apk, bundle, build, entries, _ = inputs(tmp_path)
    if damage == "corruption":
        name = PREFIX + "model-fp32.sentis"
        entries[name] = b"x" + entries[name][1:]  # Same length; metadata-only validation must fail.
    elif damage == "missing":
        del entries[PREFIX + "tokenizer.json"]
    elif damage == "extra_model":
        entries["assets/EmbeddingGemmaTextSearch/model-fp32.sentis"] = b"duplicate model"
    elif damage == "extra_abi":
        entries["lib/x86_64/libunity.so"] = b"other ABI"
    elif damage == "missing_native":
        del entries["lib/arm64-v8a/libil2cpp.so"]
    else:
        data = json.loads(entries[PREFIX + "bundle.json"])
        data["model_source"] = "f" * 40
        entries[PREFIX + "bundle.json"] = json.dumps(data).encode()
    write_apk(apk, entries)
    with pytest.raises(ValueError):
        audit_apk(apk, bundle, build, COMMIT)


def test_rejects_duplicate_zip_names(tmp_path):
    apk, bundle, build, entries, _ = inputs(tmp_path)
    write_apk(apk, entries)
    with zipfile.ZipFile(apk, "a") as archive, pytest.warns(UserWarning):
        archive.writestr(PREFIX + "tokenizer.json", entries[PREFIX + "tokenizer.json"])
    with pytest.raises(ValueError):
        audit_apk(apk, bundle, build, COMMIT)


@pytest.mark.parametrize("damage", ["injected", "failed", "commit", "profile", "model_revision", "descriptor"])
def test_requires_real_pinned_android_build_and_fixed_bundle_descriptors(tmp_path, damage):
    apk, bundle, build, entries, data = inputs(tmp_path)
    if damage == "injected":
        data["injected_build"] = True
    elif damage == "failed":
        data["success"] = False
    elif damage == "commit":
        data["build_info"]["codeCommit"] = "f" * 40
    elif damage == "profile":
        data["target_settings"]["app_bundle"] = True
    else:
        receipt = json.loads(bundle.read_text())
        if damage == "model_revision":
            receipt["model_revision"] = "f" * 40
        else:
            receipt["files"][0]["name"] = "../model-fp32.sentis"
        bundle.write_text(json.dumps(receipt), encoding="utf-8")
        entries[PREFIX + "bundle.json"] = bundle.read_bytes()
    build.write_text(json.dumps(data), encoding="utf-8")
    write_apk(apk, entries)
    with pytest.raises(ValueError):
        audit_apk(apk, bundle, build, COMMIT)


def test_cli_returns_small_failure_receipt_without_runtime_or_gpu_claim(tmp_path, capsys):
    apk, bundle, build, entries, _ = inputs(tmp_path)
    del entries[PREFIX + "tokenizer.json"]
    write_apk(apk, entries)
    output = tmp_path / "audit.json"
    assert main(["--apk", str(apk), "--bundle", str(bundle), "--build", str(build), "--commit", COMMIT, "--output", str(output)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads(output.read_text())
    assert not result["success"] and not result["gpu_verified"] and not result["unity_runtime_executed"]


def test_cli_preserves_an_existing_result(tmp_path):
    apk, bundle, build, entries, _ = inputs(tmp_path)
    write_apk(apk, entries)
    output = tmp_path / "audit.json"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="existing"):
        main(["--apk", str(apk), "--bundle", str(bundle), "--build", str(build), "--commit", COMMIT, "--output", str(output)])
    assert output.read_text() == "keep"
