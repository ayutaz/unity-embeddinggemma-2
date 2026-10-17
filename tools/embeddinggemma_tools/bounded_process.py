"""Run one owned process with a wall-clock bound and durable terminal evidence."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET


def write_json_atomic(output, report):
    """Keep atomic receipts while tolerating brief Windows sharing conflicts."""
    output = Path(output)
    pending = output.with_suffix(output.suffix + ".pending")
    pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for attempt in range(10):
        try:
            pending.replace(output)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in {5, 32, 33} or attempt == 9:
                raise
            time.sleep(0.05)


def read_unity_execution_counts(path):
    """Reject empty/skipped Unity runs independently of the process exit code."""
    try:
        root = ET.parse(path).getroot()
        counts = {key:int(root.attrib[key]) for key in ('total','passed','failed','skipped')}
    except (OSError, ET.ParseError, KeyError, ValueError) as error:
        raise ValueError('Unity execution XML is missing or invalid') from error
    if (root.tag != 'test-run' or counts['total'] <= 0 or counts['skipped'] != 0
            or any(value < 0 for value in counts.values())
            or counts['total'] != counts['passed'] + counts['failed']):
        raise ValueError('Unity tests did not execute completely without skips')
    return counts


def run_bounded(command, output, *, timeout=300, env=None, unity_results=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if timeout <= 0 or not command:
        raise ValueError("positive timeout and command required")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = dict(command=command, timeout_seconds=timeout, success=False,
                  terminal=False, timed_out=False, started_utc=datetime.now(timezone.utc).isoformat())
    def save():
        write_json_atomic(output, report)
    save()
    process = None
    try:
        process = subprocess.Popen(command, env=env, shell=False)
        report["pid"] = process.pid
        save()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            report["timed_out"] = True
            # This exact handle belongs to this invocation; never kill by name.
            process.kill()
            process.wait(timeout=30)
        report.update(terminal=True, exit_code=process.returncode,
                      success=not report["timed_out"] and process.returncode == 0)
        if unity_results is not None:
            report['unity_results'] = str(unity_results)
            try:
                report['unity_counts'] = read_unity_execution_counts(unity_results)
                report['success'] = report['success'] and report['unity_counts']['failed'] == 0
            except ValueError as error:
                report['success'] = False
                report['unity_evidence_error'] = str(error)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if process is not None:
                if process.poll() is None:
                    # Receipt errors and interruption must not leave this exact child alive.
                    # Reaping this child does not make a supervision failure pass.
                    report["success"] = False
                    process.kill()
                    process.wait(timeout=30)
                    report["supervisor_terminated_child"] = True
                report.update(terminal=process.poll() is not None, exit_code=process.returncode)
        finally:
            report.update(elapsed_seconds=time.monotonic() - started,
                          completed_utc=datetime.now(timezone.utc).isoformat())
            save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--unity-results", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    environment = dict(os.environ)
    if os.name == "nt":
        environment.setdefault("ALLUSERSPROFILE", r"C:\ProgramData")
    report = run_bounded(command, args.output, timeout=args.timeout, env=environment, unity_results=args.unity_results)
    print(json.dumps(report))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
