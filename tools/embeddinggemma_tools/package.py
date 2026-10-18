"""Audit the text UPM distribution and prepare an isolated Unity consumer project."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

PACKAGE_NAME = "com.ayutaz.embeddinggemma"
ROOT = Path(__file__).resolve().parents[2]
DEPENDENCIES = {"com.unity.ai.inference": "2.6.1", "com.unity.nuget.newtonsoft-json": "3.2.2"}


def public_path(path,label):
    resolved=Path(path).resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else '<'+label+'>'


def audit_git_consumer(project, revision):
    """Audit the requested URL and UPM lock; this is not Editor execution evidence."""
    project = Path(project)
    report = {"success": False, "project": public_path(project,'CONSUMER_PROJECT'), "requested_revision": revision,
              "resolved_revision": None, "unity_executed": False, "errors": []}
    errors = report["errors"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        errors.append("revision must be a full lowercase commit SHA")
        return report
    url = f"https://github.com/ayutaz/unity-embeddinggemma-2.git?path=/Packages/{PACKAGE_NAME}#{revision}"

    def dependencies(name):
        try:
            data = json.loads((project / "Packages" / name).read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict) or not isinstance(data.get("dependencies"), dict):
                raise ValueError("expected an object with dependencies")
            return data["dependencies"]
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: {type(exc).__name__}")
            return {}

    requested = dependencies("manifest.json")
    locked = dependencies("packages-lock.json")
    if requested.get(PACKAGE_NAME) != url:
        errors.append("manifest.json: package must request the exact repository, subfolder and commit")
    entry = locked.get(PACKAGE_NAME)
    if not isinstance(entry, dict):
        errors.append("packages-lock.json: missing resolved package")
    else:
        report["resolved_revision"] = entry.get("hash")
        if entry.get("source") != "git" or entry.get("version") != url or entry.get("hash") != revision:
            errors.append("packages-lock.json: Git source, requested URL or resolved commit mismatch")
        if entry.get("dependencies") != DEPENDENCIES:
            errors.append("packages-lock.json: package direct dependencies changed")
    for name, version in DEPENDENCIES.items():
        dependency = locked.get(name)
        if not isinstance(dependency, dict) or dependency.get("version") != version or dependency.get("source") != "registry":
            errors.append(f"packages-lock.json: expected registry dependency {name}@{version}")
    report["success"] = not errors
    return report


def audit_package(package):
    package = Path(package)
    errors = []

    def read_json(name):
        try:
            value = json.loads((package / name).read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("expected object")
            return value
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: {type(exc).__name__}")
            return {}

    manifest = read_json("package.json")
    if manifest.get("name") != PACKAGE_NAME:
        errors.append("package.json: unexpected package name")
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", str(manifest.get("version", ""))):
        errors.append("package.json: expected semantic version")
    if manifest.get("unity") != "6000.3" or manifest.get("unityRelease") != "16f1":
        errors.append("package.json: expected tested Unity 6000.3.16f1 minimum")
    if manifest.get("dependencies") != DEPENDENCIES:
        errors.append("package.json: declare only the pinned direct Runtime dependencies")
    for name in ("README.md", "CHANGELOG.md", "LICENSE.md", "Documentation~/index.md", "Samples~/README.md",
                 "Runtime/TextEmbedder.cs", "Runtime/TextPrompts.cs", "Runtime/TextTokenizer.cs", "Runtime/TextModelFile.cs"):
        if not (package / name).is_file():
            errors.append(f"missing {name}")
    runtime = read_json("Runtime/EmbeddingGemma.Runtime.asmdef")
    if runtime.get("name") != "EmbeddingGemma.Runtime" or set(runtime.get("references", [])) != {
            "Unity.InferenceEngine", "Unity.InferenceEngine.Tokenization", "Unity.Newtonsoft.Json"}:
        errors.append("Runtime assembly must preserve its name and declared references")
    if runtime.get("includePlatforms") or runtime.get("excludePlatforms"):
        errors.append("Runtime assembly must remain usable in Editor and Player")
    tests = read_json("Tests/Editor/EmbeddingGemma.Package.Editor.Tests.asmdef")
    if (tests.get("name") != "EmbeddingGemma.Package.Editor.Tests"
            or tests.get("includePlatforms") != ["Editor"]
            or tests.get("optionalUnityReferences") != ["TestAssemblies"]
            or tests.get("autoReferenced") is not False):
        errors.append("Package tests must be an explicit Editor test assembly")
    forbidden = {".dll", ".so", ".dylib", ".bundle", ".a", ".safetensors", ".pt2", ".onnx", ".onnx_data", ".sentis", ".tflite", ".bin"}
    guids = set()
    for path in package.rglob("*"):
        if path.suffix.lower() in forbidden:
            errors.append(f"forbidden distribution asset: {path.relative_to(package)}")
        if not path.is_file():
            continue
        if path.suffix in {".cs", ".asmdef"} and not Path(str(path) + ".meta").is_file():
            errors.append(f"missing meta: {path.relative_to(package)}")
        if path.suffix == ".meta":
            match = re.search(r"(?m)^guid: ([0-9a-f]{32})$", path.read_text(encoding="utf-8"))
            if match is None or match[1] in guids:
                errors.append(f"invalid or duplicate GUID: {path.relative_to(package)}")
            else:
                guids.add(match[1])
        if path.suffix == ".cs" and path.is_relative_to(package / "Runtime"):
            source = path.read_text(encoding="utf-8")
            if re.search(r"\bUnityEditor\b", source):
                errors.append(f"Runtime Editor dependency: {path.relative_to(package)}")
    return {"success": not errors, "package": manifest.get("name"), "version": manifest.get("version"), "errors": errors}


def create_consumer(project, package, *, git_revision=None, automation=False, sample=False, media_sample=False, test_framework_source=None):
    project, package = Path(project), Path(package).resolve()
    report = audit_package(package)
    if not report["success"]:
        raise ValueError("package audit failed: " + "; ".join(report["errors"]))
    if git_revision is not None and not re.fullmatch(r"[0-9a-f]{40}", git_revision):
        raise ValueError("git_revision must be a full lowercase commit SHA")
    sample_path = package / "Samples~/TextSearch"
    if sample and not (sample_path / "TextSearch.unity").is_file():
        raise ValueError("TextSearch sample must be available before creating its consumer")
    media_path = package / "Samples~/MultimodalSearch"
    if media_sample and not (media_path / "MultimodalSearch.unity").is_file():
        raise ValueError("MultimodalSearch sample must be available before creating its consumer")
    if project.exists() and (not project.is_dir() or any(project.iterdir())):
        raise ValueError("consumer project must be empty; existing projects are never overwritten")
    framework_receipt = None
    if test_framework_source is not None:
        test_framework_source = Path(test_framework_source).resolve()
        configuration = json.loads((test_framework_source / "package.json").read_text(encoding="utf-8"))
        if configuration.get("name") != "com.unity.test-framework" or configuration.get("version") != "1.6.0":
            raise ValueError("embedded source must be the fixed com.unity.test-framework 1.6.0 package")
        if not (test_framework_source / "LICENSE.md").is_file():
            raise ValueError("embedded framework must retain its license")
        if project.resolve().is_relative_to(test_framework_source) or test_framework_source.is_relative_to(project.resolve()):
            raise ValueError("framework source and destination must not contain each other")
        entries = {}
        for path in sorted(test_framework_source.rglob("*")):
            if path.is_symlink():
                raise ValueError("embedded source must not contain symbolic links")
            if path.is_file():
                with path.open("rb") as stream:
                    checksum = hashlib.file_digest(stream, "sha256").hexdigest()
                entries[path.relative_to(test_framework_source).as_posix()] = dict(bytes=path.stat().st_size, sha256=checksum)
        framework_receipt = dict(schema_version=1, source_package=configuration["name"], source_version=configuration["version"],
                                 source_files=entries, unity_compiled=False,
                                 missing_script_metadata=[name + ".meta" for name in entries
                                                          if name.endswith(".cs") and name + ".meta" not in entries])
    source = (f"https://github.com/ayutaz/unity-embeddinggemma-2.git?path=/Packages/{PACKAGE_NAME}#{git_revision}"
              if git_revision else "file:" + package.as_posix())
    manifest = {"dependencies": {PACKAGE_NAME: source, "com.unity.test-framework": "1.6.0"}, "testables": [PACKAGE_NAME]}
    if sample or media_sample:
        manifest["dependencies"]["com.unity.modules.unitywebrequest"] = "1.0.0"
    if media_sample:
        manifest["dependencies"]["com.unity.modules.audio"] = "1.0.0"
    if automation:
        manifest["dependencies"]["io.github.hatayama.uloopmcp"] = "3.14.0"
        manifest["scopedRegistries"] = [{"name": "package.openupm.com", "url": "https://package.openupm.com",
                                         "scopes": ["io.github.hatayama.uloopmcp"]}]
    for directory in ("Assets", "Packages", "ProjectSettings"):
        (project / directory).mkdir(parents=True, exist_ok=True)
    if framework_receipt is not None:
        # Embedded packages are mutable. Unity's importer generates missing
        # metadata on first import; neither the Editor nor PackageCache is edited.
        shutil.copytree(test_framework_source, project / "Packages/com.unity.test-framework")
        (project / "ProjectSettings/EmbeddingGemmaTestFramework.json").write_text(
            json.dumps(framework_receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    (project / "Packages/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    (project / "ProjectSettings/ProjectVersion.txt").write_text("m_EditorVersion: 6000.3.21f1\n", encoding="utf-8", newline="\n")
    if automation:
        # Isolate validation from an ambient Accelerator endpoint in Editor preferences.
        # Ordinary consumer projects retain Unity's defaults.
        (project / "ProjectSettings/EditorSettings.asset").write_text(
            "%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n--- !u!159 &1\n"
            "EditorSettings:\n  m_ObjectHideFlags: 0\n  serializedVersion: 15\n"
            "  m_CacheServerMode: 2\n  m_CacheServerEnableDownload: 0\n"
            "  m_CacheServerEnableUpload: 0\n", encoding="utf-8", newline="\n")
    if sample:
        shutil.copytree(sample_path, project / "Assets/EmbeddingGemmaTextSearch")
    if media_sample:
        shutil.copytree(media_path, project / "Assets/EmbeddingGemmaMultimodalSearch")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=ROOT / "Packages" / PACKAGE_NAME)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--consumer", type=Path)
    action.add_argument("--verify-git-consumer", type=Path, help="Audit requested and resolved Git commit; does not run Unity")
    parser.add_argument("--git-revision")
    parser.add_argument("--automation", action="store_true")
    parser.add_argument("--sample", action="store_true", help="Import TextSearch into a new empty consumer")
    parser.add_argument("--media-sample", action="store_true", help="Import Image and Audio Search into a new empty consumer")
    parser.add_argument("--test-framework-source", type=Path,
                        help="Embed a local official Test Framework 1.6.0 source in a new consumer; does not patch cache")
    args = parser.parse_args(argv)
    if args.test_framework_source is not None and args.consumer is None:
        parser.error("--test-framework-source requires --consumer")
    if args.verify_git_consumer:
        report = audit_git_consumer(args.verify_git_consumer, args.git_revision)
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["success"] else 1
    report = audit_package(args.package)
    if report["success"] and args.consumer:
        try:
            create_consumer(args.consumer, args.package, git_revision=args.git_revision, automation=args.automation,
                            sample=args.sample, media_sample=args.media_sample, test_framework_source=args.test_framework_source)
            report["consumer"] = public_path(args.consumer,'CONSUMER_PROJECT')
        except (OSError, ValueError) as exc:
            report["success"] = False
            report["errors"].append(str(exc))
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
