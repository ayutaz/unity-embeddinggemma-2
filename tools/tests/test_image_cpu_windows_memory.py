"""Whole-process Windows peak memory must remain measured or explicitly unknown."""
import sys

import pytest

from embeddinggemma_tools import cpu_progress, image_cpu_reference


@pytest.mark.parametrize('observed,expected', [
    (987654321, 987654321), (None, None), (0, None), (-1, None), (False, None), (1.5, None),
])
def test_windows_peak_uses_existing_OS_counter_without_inventing_missing_values(monkeypatch, observed, expected):
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(cpu_progress, 'own_process_memory', lambda: {'rss_bytes': 42, 'peak_rss_bytes': observed})
    assert image_cpu_reference._peak_rss_bytes() == expected


def test_native_supported_host_reports_a_positive_whole_process_peak():
    value = image_cpu_reference._peak_rss_bytes()
    if sys.platform in ('win32', 'linux'):
        assert type(value) is int and value > 0
    else:
        assert value is None
