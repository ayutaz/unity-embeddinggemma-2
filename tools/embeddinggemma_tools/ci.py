"""Fail the required PR check unless every required dependency succeeded."""

import argparse
import json
import os
from pathlib import Path
import re
import sys

from .ci_scope import SCOPE_FIELDS, event_scope

REQUIRED_JOBS = ("python", "workflow-lint", "model-reference", "package")


def validate_results(jobs):
    if not isinstance(jobs, dict):
        return list(REQUIRED_JOBS)
    return [name for name in REQUIRED_JOBS
            if not isinstance(jobs.get(name), dict) or jobs[name].get("result") != "success"]


SCOPED_JOBS = ('changes', 'documentation', *REQUIRED_JOBS)


def validate_scoped_results(jobs, expected_scope):
    if (not isinstance(jobs, dict) or not isinstance(expected_scope, dict)
            or set(expected_scope) != set(SCOPE_FIELDS)
            or any(not isinstance(value, str) for value in expected_scope.values())
            or any(re.fullmatch(r'[0-9a-f]{40}', expected_scope[key]) is None
                   for key in ('base_sha', 'head_sha', 'source_sha'))
            or re.fullmatch(r'[0-9a-f]{64}', expected_scope['change_digest']) is None
            or expected_scope.get('mode') not in ('docs', 'full')):
        return list(SCOPED_JOBS)
    expected = dict.fromkeys(SCOPED_JOBS, 'success')
    if expected_scope['mode'] == 'docs':
        expected.update({key: 'skipped' for key in ('python', 'model-reference', 'package')})
    else:
        expected['documentation'] = 'skipped'
    failed = [name for name, result in expected.items()
              if not isinstance(jobs.get(name), dict) or jobs[name].get('result') != result]
    if set(jobs) - set(SCOPED_JOBS):
        failed.append('unexpected-dependencies')
    if (not isinstance(jobs.get('changes'), dict)
            or jobs['changes'].get('outputs') != expected_scope):
        if 'changes' not in failed:
            failed.append('changes')
    return failed


def main(argv=()):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scoped', action='store_true')
    parser.add_argument('--root', type=Path, default=Path('..'))
    args = parser.parse_args(argv)
    try:
        jobs = json.loads(os.environ["NEEDS_JSON"])
    except (KeyError, ValueError):
        jobs = None
    if args.scoped:
        try:
            scope = event_scope(args.root)
        except (OSError, ValueError, TypeError):
            scope = None
        failed = validate_scoped_results(jobs, scope)
    else:
        failed = validate_results(jobs)
    print(json.dumps({"success": not failed, "unsuccessful_jobs": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
