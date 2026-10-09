import json
from pathlib import Path
import uuid

import pytest

from embeddinggemma_tools.package import PACKAGE_NAME, audit_package, create_consumer

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    (root / "package.json").write_text(json.dumps({
        "name": PACKAGE_NAME, "version": "0.1.0-pre.1", "unity": "6000.3", "unityRelease": "16f1",
        "dependencies": {"com.unity.ai.inference": "2.6.1", "com.unity.nuget.newtonsoft-json": "3.2.2"},
    }))
    for name in ("README.md", "CHANGELOG.md", "LICENSE.md", "Documentation~/index.md", "Samples~/README.md"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Package documentation\n")
    for name in ("TextEmbedder", "TextPrompts", "TextTokenizer", "TextModelFile"):
        path = root / "Runtime" / f"{name}.cs"
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"namespace EmbeddingGemma {{ public class {name} {{}} }}")
    for name, data in {
        "Runtime/EmbeddingGemma.Runtime.asmdef": {"name": "EmbeddingGemma.Runtime", "references": [
            "Unity.InferenceEngine", "Unity.InferenceEngine.Tokenization", "Unity.Newtonsoft.Json"]},
        "Tests/Editor/EmbeddingGemma.Package.Editor.Tests.asmdef": {"name": "EmbeddingGemma.Package.Editor.Tests",
            "references": ["EmbeddingGemma.Runtime", "Unity.InferenceEngine", "Unity.InferenceEngine.Tokenization"],
            "includePlatforms": ["Editor"], "optionalUnityReferences": ["TestAssemblies"], "autoReferenced": False},
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    for path in list(root.rglob("*")):
        if path.suffix in {".cs", ".asmdef"}:
            Path(str(path) + ".meta").write_text(f"fileFormatVersion: 2\nguid: {uuid.uuid4().hex}\n")
    return root


def test_valid_package_and_missing_package(package, tmp_path):
    assert audit_package(package)["success"] is True
    assert audit_package(tmp_path / "missing")["success"] is False


@pytest.mark.parametrize("name", ["model.sentis", "model.pt2", "weights.safetensors", "native.dll", "native.so", "native.dylib"])
def test_models_and_native_plugins_are_rejected(package, name):
    (package / name).write_bytes(b"not distributed")
    result = audit_package(package)
    assert result["success"] is False
    assert any(name in error for error in result["errors"])


@pytest.mark.parametrize("mutation", ["missing_dependency", "editor_reference", "missing_meta", "duplicate_guid", "missing_license", "bad_version"])
def test_invalid_distribution_fails_audit(package, mutation):
    manifest = package / "package.json"
    if mutation == "missing_dependency":
        data = json.loads(manifest.read_text())
        del data["dependencies"]["com.unity.nuget.newtonsoft-json"]
        manifest.write_text(json.dumps(data))
    elif mutation == "editor_reference":
        (package / "Runtime/TextEmbedder.cs").write_text("using UnityEditor; class EditorDependency {}")
    elif mutation == "missing_meta":
        (package / "Runtime/TextEmbedder.cs.meta").unlink()
    elif mutation == "duplicate_guid":
        (package / "Runtime/TextEmbedder.cs.meta").write_text((package / "Runtime/TextTokenizer.cs.meta").read_text())
    elif mutation == "missing_license":
        (package / "LICENSE.md").unlink()
    elif mutation == "bad_version":
        data = json.loads(manifest.read_text())
        data["version"] = "latest"
        manifest.write_text(json.dumps(data))
    assert audit_package(package)["success"] is False


def test_new_consumer_declares_only_package_and_test_dependencies(package, tmp_path):
    project = tmp_path / "consumer"
    create_consumer(project, package)
    manifest = json.loads((project / "Packages/manifest.json").read_text())
    assert manifest["dependencies"] == {
        PACKAGE_NAME: "file:" + package.resolve().as_posix(), "com.unity.test-framework": "1.6.0"}
    assert manifest["testables"] == [PACKAGE_NAME]
    assert (project / "ProjectSettings/ProjectVersion.txt").read_text() == "m_EditorVersion: 6000.3.16f1\n"
    assert (project / "Assets").is_dir()
    assert not (project / "Assets/EmbeddingGemma").exists()


def test_consumer_can_pin_git_commit_and_add_editor_automation(package, tmp_path):
    project = tmp_path / "consumer"
    create_consumer(project, package, git_revision="a" * 40, automation=True)
    manifest = json.loads((project / "Packages/manifest.json").read_text())
    assert manifest["dependencies"][PACKAGE_NAME].endswith(f"?path=/Packages/{PACKAGE_NAME}#" + "a" * 40)
    assert manifest["dependencies"]["io.github.hatayama.uloopmcp"] == "3.14.0"
    assert manifest["scopedRegistries"][0]["scopes"] == ["io.github.hatayama.uloopmcp"]


def test_existing_project_is_never_overwritten(package, tmp_path):
    project = tmp_path / "consumer"
    project.mkdir()
    sentinel = project / "existing.txt"
    sentinel.write_text("keep me")
    with pytest.raises(ValueError, match="empty"):
        create_consumer(project, package)
    assert sentinel.read_text() == "keep me"


def test_invalid_package_or_git_revision_does_not_create_project(package, tmp_path):
    project = tmp_path / "consumer"
    with pytest.raises(ValueError):
        create_consumer(project, package, git_revision="main")
    assert not project.exists()
    (package / "package.json").write_text("invalid json")
    with pytest.raises(ValueError):
        create_consumer(project, package)
    assert not project.exists()


def test_repository_package_can_be_consumed_without_asset_runtime():
    assert audit_package(ROOT / "Packages" / PACKAGE_NAME)["success"] is True
    assert not (ROOT / "Assets/EmbeddingGemma/Runtime").exists()
    manifest = json.loads((ROOT / "Packages/manifest.json").read_text())
    assert PACKAGE_NAME in manifest["testables"]
    package = json.loads((ROOT / "Packages" / PACKAGE_NAME / "package.json").read_text())
    assert manifest["dependencies"][PACKAGE_NAME] == package["version"]
