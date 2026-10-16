import hashlib
import json

import pytest

from embeddinggemma_tools.editor_profile import main, summarize_editor_log


PROFILE = '''Begin MonoManager ReloadAssembly
Domain Reload Profiling: 4000ms
\tBeginReloadAssembly (1000ms)
\t\tCreateAndSetChildDomain (600ms)
\tFinalizeReload (2800ms)
\t\tSetupLoadedEditorAssemblies (1200ms)
\t\t\tProcessInitializeOnLoadMethodAttributes (700ms)
Asset Pipeline Refresh (id=abc): Total: 4.2 seconds - Initiated by RefreshV2(ForceDomainReload)
'''


def test_completed_profile_preserves_nested_timings_without_adding_them():
    result = summarize_editor_log(PROFILE)
    assert result['completed_reload_count'] == 1
    assert result['incomplete_reload_count'] == 0
    assert result['completed_reload_ms'] == {'min': 4000, 'median': 4000, 'max': 4000}
    profile = result['profiles'][0]
    assert profile['total_ms'] == 4000
    assert profile['stages_ms']['FinalizeReload'] == 2800
    assert profile['stages_ms']['CreateAndSetChildDomain'] == 600
    assert profile['stages_ms']['ProcessInitializeOnLoadMethodAttributes'] == 700
    assert result['asset_refresh_seconds'] == {'count': 1, 'min': 4.2, 'median': 4.2, 'max': 4.2}


def test_unfinished_reload_is_not_a_completed_or_slow_reload():
    result = summarize_editor_log(PROFILE + 'Begin MonoManager ReloadAssembly\nReloading Domain\n')
    assert result['completed_reload_count'] == 1
    assert result['incomplete_reload_count'] == 1
    assert result['completed_reload_ms']['max'] == 4000
    assert result['observation_complete'] is False


def test_repeated_begin_keeps_abandoned_reload_incomplete():
    result = summarize_editor_log('Begin MonoManager ReloadAssembly\n' + PROFILE)
    assert result['incomplete_reload_count'] == 1
    assert result['completed_reload_count'] == 1


def test_truncated_log_reports_unpaired_profile():
    result = summarize_editor_log('Domain Reload Profiling: 50ms\n')
    assert result['completed_reload_count'] == 1
    assert result['unpaired_profile_count'] == 1
    assert result['observation_complete'] is False


@pytest.mark.parametrize('text', ['', 'Reloading Domain\n', 'Domain Reload Profiling: NaNms\n',
                                  'Begin MonoManager ReloadAssembly\nDomain Reload Profiling: -1ms\n'])
def test_no_valid_profile_cannot_establish_completion(text):
    result = summarize_editor_log(text)
    assert result['completed_reload_ms'] is None
    assert result['observation_complete'] is False


def test_even_median_and_crlf():
    text = (PROFILE + PROFILE.replace('4000ms', '5000ms')).replace('\n', '\r\n')
    assert summarize_editor_log(text)['completed_reload_ms']['median'] == 4500


def test_stage_block_ends_before_unrelated_numeric_logs():
    result = summarize_editor_log(PROFILE + '\tFinalizeReload (999999ms)\n')
    assert result['profiles'][0]['stages_ms']['FinalizeReload'] == 2800


def test_diagnostic_counts_are_not_runtime_passes():
    result = summarize_editor_log(PROFILE + '''Start Indexing on Editor startup
  at UnityEditor.Search.SearchDatabase+<EnumerateAll>d__80.MoveNext ()
  at UnityEditor.Search.SearchDatabase.GetDefaultSearchDatabase ()
Leak Detected : Persistent allocates 1 individual allocations.
''')
    assert result['search_database_enumeration_exception_count'] == 1
    assert result['persistent_allocation_diagnostic_count'] == 1
    assert result['unity_execution_performed'] is False
    assert result['performance_improvement_verified'] is False
    assert result['m2_release_verified'] is False


def test_unknown_stage_names_and_private_log_text_are_not_exported():
    text = PROFILE.replace('\tBeginReloadAssembly', '\tC:/Users/private-project (123ms)\n\tBeginReloadAssembly')
    text += 'access token: SECRET\nC:/Users/private-project\n'
    serialized = json.dumps(summarize_editor_log(text))
    assert 'private-project' not in serialized
    assert 'SECRET' not in serialized


@pytest.mark.parametrize('text,exit_code', [(PROFILE, 0), ('', 1),
                                         (PROFILE + 'Begin MonoManager ReloadAssembly\n', 1)])
def test_cli_scopes_completion_and_hashes_exact_snapshot(tmp_path, capsys, text, exit_code):
    path = tmp_path / 'private-Editor.log'
    raw = text.encode('utf-8')
    path.write_bytes(raw)
    assert main(['--log', str(path)]) == exit_code
    result = json.loads(capsys.readouterr().out)
    assert result['log_sha256'] == hashlib.sha256(raw).hexdigest()
    assert str(path) not in json.dumps(result)


def test_missing_log_is_a_structured_cli_failure(tmp_path, capsys):
    assert main(['--log', str(tmp_path / 'missing.log')]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result['observation_complete'] is False
    assert result['error_type'] == 'FileNotFoundError'
