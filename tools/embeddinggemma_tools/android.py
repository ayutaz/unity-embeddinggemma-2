"""Audit APK payload and build provenance; never claim device or GPU execution."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

from .player import MODEL_FILES, MODEL_REVISION, _object

PREFIX = "assets/EmbeddingGemmaValidation/"
FILES = MODEL_FILES + ("reference.json", "search-reference.json")
# Observed from the pinned 6000.3.16f1 Android BuildPlayerOptions receipt.
# Keep the complete representation; do not admit arbitrary extra build tasks.
ANDROID_DEVELOPMENT_OPTIONS = "ForceOptimizeScriptCompilation, Il2CPP, CompressTextures, StripDebugSymbols, ShaderLivelinkSupport, Development"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hex(value, size):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{%d}" % size, value) is not None


def audit_apk(apk, bundle, build, expected_commit):
    apk, bundle, build = map(Path, (apk, bundle, build))
    _require(_hex(expected_commit, 40), "A fixed source commit is required")
    receipt, info = _object(build), _object(bundle)
    provenance, settings = receipt.get("build_info", {}), receipt.get("target_settings", {})
    _require(receipt.get("success") is True and receipt.get("injected_build") is False
             and receipt.get("source_commit") == expected_commit and receipt.get("build_result") == "Succeeded"
             and type(receipt.get("errors")) is int and receipt["errors"] == 0
             and receipt.get("build_options") in {"Development", "None", ANDROID_DEVELOPMENT_OPTIONS}
             and receipt.get("il2cpp_compiler") == "Release", "Successful real APK build receipt is required")
    _require(isinstance(provenance, dict) and provenance.get("codeCommit") == expected_commit
             and provenance.get("target") == "Android" and provenance.get("unityVersion") == "6000.3.16f1"
             and provenance.get("sentisVersion") == "2.6.1" and provenance.get("scriptingBackend") == "IL2CPP"
             and provenance.get("stripping") == "High", "Pinned Android IL2CPP provenance is required")
    sources = provenance.get("sourceSha256")
    _require(isinstance(sources, dict) and sources and all(isinstance(name, str) and _hex(value, 64)
             for name, value in sources.items()), "Source hashes are required")
    expected_settings = {"architectures": "ARM64", "application_identifier": "com.ayutaz.embeddinggemma.validation",
                         "min_sdk": 26, "target_sdk": "AndroidApiLevelAuto", "custom_keystore": False,
                         "apk_expansion": False, "apk_per_cpu": False, "app_bundle": False,
                         "export_project": False, "automatic_graphics_api": False, "graphics_apis": ["Vulkan", "OpenGLES3"]}
    _require(isinstance(settings, dict) and all(settings.get(key) == value and type(settings.get(key)) is type(value)
             for key, value in expected_settings.items()), "Expected single ARM64 APK profile is required")
    _require(info.get("success") is True and info.get("model_revision") == MODEL_REVISION
             and info.get("unity_preparation_version") == "6000.3.16f1" and _hex(info.get("model_source"), 40)
             and _hex(info.get("search_source"), 40) and _hex(info.get("model_origin_sha256"), 64), "Pinned prepared bundle is required")
    files = info.get("files")
    _require(isinstance(files, list) and len(files) == 5 and all(isinstance(row, dict) for row in files)
             and [row.get("name") for row in files] == list(FILES)
             and all(type(row.get("bytes")) is int and row["bytes"] > 0 and _hex(row.get("sha256"), 64)
                     for row in files), "Exactly five fixed bundle file descriptors are required")
    checked = []
    with zipfile.ZipFile(apk) as archive:
        entries = [entry for entry in archive.infolist() if not entry.is_dir()]
        names = [entry.filename for entry in entries]
        _require(len(names) == len(set(names)), "Duplicate APK ZIP names are not allowed")
        required = {PREFIX + name for name in ("bundle.json",) + FILES}
        _require({name for name in names if name.startswith(PREFIX)} == required, "APK bundle contents differ")
        extra_models = [name for name in names if name.lower().endswith((".sentis", ".pt2", ".onnx", ".safetensors")) and name not in required]
        _require(not extra_models, "Unexpected or duplicated APK models: " + ", ".join(extra_models))
        abis = sorted({name.split("/")[1] for name in names if name.startswith("lib/") and len(name.split("/")) >= 3})
        _require(abis == ["arm64-v8a"], "APK must contain only the intended ARM64 ABI")
        for name in ("lib/arm64-v8a/libil2cpp.so", "lib/arm64-v8a/libunity.so"):
            _require(name in names and archive.getinfo(name).file_size > 0, "Required native library missing: " + name)
        _require(archive.getinfo(PREFIX + "bundle.json").file_size <= 65536, "APK bundle manifest too large")
        _require(json.loads(archive.read(PREFIX + "bundle.json")) == info, "APK manifest differs from the audited source bundle")
        for row in files:
            entry = archive.getinfo(PREFIX + row["name"])
            _require(entry.file_size == row["bytes"], "APK uncompressed length differs: " + row["name"])
            with archive.open(entry) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            _require(digest == row["sha256"], "APK full SHA-256 differs: " + row["name"])
            checked.append({"name": row["name"], "bytes": entry.file_size, "compressed_bytes": entry.compress_size,
                            "compression": entry.compress_type, "sha256": digest})
    with apk.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"success": True, "apk_payload_verified": True, "unity_runtime_executed": False, "gpu_verified": False,
            "source_commit": expected_commit, "apk_bytes": apk.stat().st_size, "apk_sha256": digest,
            "abis": abis, "files": checked, "build": receipt, "bundle": info,
            "scope": "APK payload hashes and build receipt only; signature, installation, jar runtime, device precision and performance are not verified."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("apk", "bundle", "build"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        raise ValueError("Never overwrite an existing result")
    try:
        report = audit_apk(args.apk, args.bundle, args.build, args.commit)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, zipfile.BadZipFile) as error:
        report = {"success": False, "apk_payload_verified": False, "unity_runtime_executed": False,
                  "gpu_verified": False, "error": str(error)}
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
