"""Publish relative skip locations while preserving outcomes and exact reasons."""
from pathlib import Path

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome=yield
    report=outcome.get_result()
    if report.skipped and isinstance(report.longrepr,tuple) and len(report.longrepr)==3:
        filename,line,reason=report.longrepr
        try:
            location=Path(filename).resolve().relative_to(Path(item.config.rootpath).resolve()).as_posix()
        except ValueError:
            location='<external-skip-location>'
        report.longrepr=(location,line,reason)
