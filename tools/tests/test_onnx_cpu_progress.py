import json
import pathlib

import pytest

from embeddinggemma_tools import onnx_cpu_comparison as cpu


def test_own_memory_snapshot_records_positive_observations_or_explicit_missing():
    value = cpu.own_process_memory()
    assert set(value) == {'rss_bytes', 'peak_rss_bytes'}
    assert all(v is None or type(v) is int and v > 0 for v in value.values())
    assert any(v is not None for v in value.values())


def test_journal_flushes_phase_and_memory_before_native_work(tmp_path, capsys):
    path = tmp_path / 'progress.json'
    with cpu.ProgressJournal(path, source='a' * 40, case_id='silence',
            memory=lambda: dict(rss_bytes=123, peak_rss_bytes=456)) as journal:
        journal.record('session_create_begin', part=2)
        current = json.loads(path.read_text())
        assert not current['terminal'] and current['events'][-1]['phase'] == 'session_create_begin'
        assert current['events'][-1]['rss_bytes'] == 123
        assert 'session_create_begin' in capsys.readouterr().out
    result = json.loads(path.read_text())
    assert result['terminal'] and result['success']
    assert not result['complete_performance_accepted'] and not result['GPU_validated']


def test_journal_preserves_phase_and_safe_failure_without_exception_details(tmp_path, capsys):
    path = tmp_path / 'failed.json'
    with pytest.raises(RuntimeError):
        with cpu.ProgressJournal(path, source='a' * 40, case_id='silence') as journal:
            journal.record('load_part_begin', part=1)
            raise RuntimeError('unpublished-synthetic-detail')
    result = json.loads(path.read_text())
    assert result['terminal'] and not result['success']
    assert result['events'][-2]['phase'] == 'load_part_begin'
    assert result['error_class'] == 'RuntimeError'
    assert 'unpublished-synthetic-detail' not in path.read_text() + capsys.readouterr().out


def test_journal_missing_memory_is_null_and_never_a_zero_measurement(tmp_path):
    path = tmp_path / 'unknown.json'
    with cpu.ProgressJournal(path, source='a' * 40, case_id='silence',
            memory=lambda: dict(rss_bytes=None, peak_rss_bytes=None)) as journal:
        journal.record('load_part_begin', part=0)
    assert all(e['rss_bytes'] is None and e['peak_rss_bytes'] is None
               for e in json.loads(path.read_text())['events'])


@pytest.mark.parametrize('pending', [False, True])
def test_journal_never_overwrites_an_existing_or_pending_evidence_file(tmp_path, pending):
    path = tmp_path / 'preserved.json'
    existing = path.with_suffix('.json.pending') if pending else path
    existing.write_bytes(b'preserved')
    with pytest.raises(FileExistsError):
        cpu.ProgressJournal(path, source='a' * 40, case_id='silence')
    assert existing.read_bytes() == b'preserved'


def test_journal_rejects_unregistered_phase_text_before_it_can_be_published(tmp_path):
    path = tmp_path / 'safe.json'
    with cpu.ProgressJournal(path, source='a' * 40, case_id='silence') as journal:
        with pytest.raises(ValueError):
            journal.record('unpublished-synthetic-detail', part=0)
    assert 'unpublished-synthetic-detail' not in path.read_text()
