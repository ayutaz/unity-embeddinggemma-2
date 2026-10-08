"""Fail the required PR check unless every required dependency succeeded."""

import json
import os

REQUIRED_JOBS = ("python", "workflow-lint", "model-reference", "package")


def validate_results(jobs):
    if not isinstance(jobs, dict):
        return list(REQUIRED_JOBS)
    return [name for name in REQUIRED_JOBS
            if not isinstance(jobs.get(name), dict) or jobs[name].get("result") != "success"]


def main():
    try:
        jobs = json.loads(os.environ["NEEDS_JSON"])
    except (KeyError, ValueError):
        jobs = None
    failed = validate_results(jobs)
    print(json.dumps({"success": not failed, "unsuccessful_jobs": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
