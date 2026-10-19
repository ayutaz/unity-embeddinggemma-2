"""Windows comparison reports need hardware identity without user/path fields."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from embeddinggemma_tools import cpu_execution_context as context


class Key:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def registry(monkeypatch, values, *, denied=False):
    def open_key(root, path):
        assert path == r'HARDWARE\DESCRIPTION\System\CentralProcessor\0'
        if denied:
            raise OSError('unavailable')
        return Key()
    fake = SimpleNamespace(HKEY_LOCAL_MACHINE=object(), OpenKey=open_key,
                           QueryValueEx=lambda key, name: (values.get(name), 1))
    monkeypatch.setitem(sys.modules, 'winreg', fake)


def test_windows_registry_reports_only_processor_identity(monkeypatch):
    registry(monkeypatch, {'ProcessorNameString': ' Synthetic CPU ', 'VendorIdentifier': 'SyntheticVendor',
                           'Identifier': 'Synthetic64 Family 6 Model 7 Stepping 8'})
    assert context._windows_cpu_model() == {'model name': 'Synthetic CPU', 'vendor_id': 'SyntheticVendor',
                                            'cpu family': '6', 'model': '7', 'stepping': '8'}


@pytest.mark.parametrize('value', [None, 0, '  '])
def test_missing_registry_identity_is_unknown_not_an_invented_model(monkeypatch, value):
    registry(monkeypatch, {'ProcessorNameString': value, 'VendorIdentifier': value, 'Identifier': value})
    assert context._windows_cpu_model() == {}


def test_unavailable_registry_is_kept_unknown(monkeypatch):
    registry(monkeypatch, {}, denied=True)
    assert context._windows_cpu_model() == {}


def test_default_windows_context_uses_native_identity_after_proc_is_unavailable(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'win32')
    original = Path.read_text
    def read(path, *args, **kwargs):
        if path == Path('/proc/cpuinfo'):
            raise OSError('unavailable')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(context, '_windows_cpu_model', lambda: {'model name': 'Synthetic CPU'}, raising=False)
    value = context.cpu_execution_context(environ={})
    assert value['cpu_model_available'] is True and value['cpu_model'] == {'model name': 'Synthetic CPU'}


def test_explicit_missing_cpuinfo_does_not_silently_replace_the_requested_source(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'platform', 'win32')
    def unexpected():
        raise AssertionError('Explicit CPU metadata input must remain authoritative')
    monkeypatch.setattr(context, '_windows_cpu_model', unexpected, raising=False)
    value = context.cpu_execution_context(cpuinfo_path=tmp_path / 'missing', environ={})
    assert value['cpu_model_available'] is False and value['cpu_model'] == {}
