"""Local Unity CLI Loop harness. Run from tools with uv; evidence stays in artifacts/."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess


def _payload(stdout):
    # launch can print progress before its final JSON; accept one final object only.
    decoder = json.JSONDecoder()
    for index, char in enumerate(stdout):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(stdout[index:])
        except ValueError:
            continue
        if isinstance(value, dict) and not stdout[index + end:].strip():
            return value
    raise ValueError("CLI did not return a JSON result")


def run_harness(project: Path, output: Path, cli: str, suite: str, *,
                runner=subprocess.run, launch=False, timeout=900) -> dict:
    scopes = {"m1": ("M1ReferenceTests", 3, "m1_reference_passed"),
              "runtime": ("TextEmbedderReferenceTests", 2, "runtime_reference_passed"),
              "completion": ("M1CompletionTests", 1, "m1_completion_passed"),
              "search": ("TextSearchReferenceTests", 4, "search_reference_passed")}
    if suite not in ("compile", *scopes) or not 1 <= timeout <= 1200:
        raise ValueError("suite must be compile/m1/runtime/completion/search and timeout between 1 and 1200 seconds")
    project = project.resolve()
    version_file = project / "ProjectSettings/ProjectVersion.txt"
    version = next(line.split(":", 1)[1].strip() for line in version_file.read_text().splitlines()
                   if line.startswith("m_EditorVersion:"))
    output.mkdir(parents=True, exist_ok=True)
    report = {"success": False, "m1_reference_passed": False, "runtime_reference_passed": False, "m1_completion_passed": False, "search_reference_passed": False, "suite": suite,
              "project": str(project), "unity_version": version, "cli": cli,
              "started_at": datetime.now(timezone.utc).isoformat(), "steps": []}
    summary = output / "summary.json"

    def save():
        summary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Clear stale success before attempting a command, including failed launches.
    save()
    environment = dict(os.environ)
    if os.name == "nt":
        environment.setdefault("ALLUSERSPROFILE", environment.get("ProgramData", r"C:\ProgramData"))

    def execute(command, arguments=(), wait=timeout):
        args = [cli, "--project-path", str(project), command, *arguments]
        evidence = {"command": args}
        try:
            result = runner(args, cwd=project, env=environment, shell=False,
                            text=True, encoding="utf-8", errors="replace",
                            capture_output=True, timeout=wait + 30)
            evidence.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except subprocess.TimeoutExpired as exc:
            def text(value):
                return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")
            evidence.update(error="timeout", stdout=text(exc.stdout), stderr=text(exc.stderr))
            raise RuntimeError(f"{command}: timeout after {wait + 30}s") from exc
        finally:
            (output / f"{command}.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
            report["steps"].append(command)
        try:
            payload = _payload(result.stdout)
        except ValueError:
            if result.returncode == 0:
                raise
            # uloop emits structured connection failures on stderr. Preserve the
            # actual reason; an accepted request must never be retried here.
            payload = _payload(result.stderr)
        if command == "run-tests":
            report["tests"] = payload
            if payload.get("XmlPath"):
                xml = Path(payload["XmlPath"]).resolve()
                if xml.is_relative_to((project / ".uloop/outputs/TestResults").resolve()) and xml.is_file():
                    shutil.copyfile(xml, output / "test-results.xml")
        if result.returncode != 0:
            error = payload.get("Error", {})
            reason = error.get("ErrorCode", "") if isinstance(error, dict) else ""
            raise RuntimeError(f"{command}: exit {result.returncode} {reason}; see {command}.json")
        return payload

    try:
        if launch:
            ready = execute("launch")
            if ready.get("Success") is not True or ready.get("Ready") is not True:
                raise RuntimeError("Unity launch did not confirm CLI readiness")
        compiled = execute("compile", ["--stop-on-external-scene-changes", "--timeout-seconds", str(timeout)])
        if compiled.get("Success") is not True or compiled.get("ErrorCount") != 0:
            raise RuntimeError("Unity compilation did not pass")
        if suite in scopes:
            fixture, count, passed_flag = scopes[suite]
            tests = execute("run-tests", ["--test-mode", "EditMode", "--filter-type", "class",
                            "--filter-value", f"EmbeddingGemma.Tests.{fixture}", "--unsaved-changes", "fail",
                            "--timeout-seconds", str(timeout)])
            report["tests"] = tests
            passed = (tests.get("Success") is True and tests.get("TestCount") == count
                      and tests.get("PassedCount") == count and tests.get("FailedCount") == 0
                      and tests.get("SkippedCount") == 0 and tests.get("InconclusiveCount", 0) == 0)
            if not passed:
                raise RuntimeError(f"{suite} requires {count} reference tests passed, no skips")
            report[passed_flag] = True
        report["success"] = True
    except (OSError, ValueError, RuntimeError) as exc:
        report["error"] = str(exc)
    finally:
        try:
            execute("get-logs", ["--max-count", "200", "--include-stack-trace"], wait=60)
        except (OSError, ValueError, RuntimeError) as exc:
            report["logs_error"] = str(exc)
            report["success"] = False
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        save()
    return report


def main(argv=None):
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=project)
    parser.add_argument("--uloop", default=str(project / "artifacts/uloop/bin/uloop.exe"))
    parser.add_argument("--suite", choices=["compile", "m1", "runtime", "completion", "search"], default="m1")
    parser.add_argument("--launch", action="store_true", help="Open the Editor and wait for readiness first")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output", type=Path, default=project / "artifacts/unity-harness" /
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    args = parser.parse_args(argv)
    report = run_harness(args.project, args.output, args.uloop, args.suite,
                         launch=args.launch, timeout=args.timeout)
    print(json.dumps({"success": report["success"], "m1_reference_passed": report["m1_reference_passed"],
                      "runtime_reference_passed": report.get("runtime_reference_passed", False),
                      "m1_completion_passed": report.get("m1_completion_passed", False),
                      "search_reference_passed": report.get("search_reference_passed", False),
                      "evidence": str(args.output.resolve()), "error": report.get("error")}, ensure_ascii=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
