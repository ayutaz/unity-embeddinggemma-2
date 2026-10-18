"""Startup/refresh observations remain separate from model and search timings."""
import json
import pytest
from embeddinggemma_tools import editor_profile


def info(pid=12, **changes):
    values = dict(type='ProjectInfo', processId=pid, projectLoad=23.5, projectInit=21.3,
        templateInit=0.0, packageManagerInit=.7, assetDatabaseInit=2.0,
        globalIlluminationInit=.01, assembliesLoad=2.1, unityExtensionsInit=.02,
        assetDatabaseRefresh=14.9, sceneOpening=1.6)
    values.update(changes)
    return '##utp:' + json.dumps(values)


def test_exact_process_and_known_metrics_without_private_payload():
    result = editor_profile.summarize_editor_startup(info(123) + '\n' + info(private='SECRET'), 12)
    assert result['project_info_available'] is True
    assert result['project_info_seconds']['projectLoad'] == 23.5
    assert 'SECRET' not in json.dumps(result)
    assert result['search_database_duration_seconds'] is None
    assert result['timings_additive'] is False


def test_missing_is_unavailable_but_reported_zero_is_valid():
    absent = editor_profile.summarize_editor_startup(info(123), 12)
    assert absent['project_info_available'] is False and absent['project_info_seconds'] is None
    zero = editor_profile.summarize_editor_startup(info(projectLoad=0), 12)
    assert zero['project_info_available'] is True and zero['project_info_seconds']['projectLoad'] == 0


@pytest.mark.parametrize('bad', [-1, float('nan'), float('inf'), True, '23.5'])
def test_invalid_metric_cannot_become_a_valid_startup_duration(bad):
    result = editor_profile.summarize_editor_startup(info(projectLoad=bad), 12)
    assert result['project_info_available'] is False
    assert result['project_info_seconds']['projectLoad'] is None


def test_refresh_import_counts_are_bound_to_their_own_summary():
    text = '''Asset Pipeline Refresh (id=a): Total: 14.944 seconds - Initiated by InitialRefreshV2(ForceSynchronousImport)
\tSummary:
\t\tImports: total=0 (actual=0, local cache=0, cache server=0)
Asset Pipeline Refresh (id=b): Total: 2.5 seconds - Initiated by RefreshV2(Default)
\tSummary:
\t\tImports: total=4 (actual=1, local cache=2, cache server=1)
'''
    result = editor_profile.summarize_editor_startup(text, 12)
    first, second = result['asset_refreshes']
    assert first['seconds'] == 14.944 and first['imports']['actual'] == 0
    assert first['initiator'] == 'InitialRefreshV2'
    assert second['imports'] == dict(total=4, actual=1, local_cache=2, cache_server=1)
    assert result['model_load_seconds'] is None


def test_unrelated_import_line_does_not_fill_a_missing_refresh_summary():
    text = '''Asset Pipeline Refresh (id=a): Total: 1 seconds - Initiated by /private/SECRET()
Unrelated message
\t\tImports: total=9 (actual=9, local cache=0, cache server=0)
'''
    result = editor_profile.summarize_editor_startup(text, 12)
    assert result['asset_refreshes'][0]['imports'] is None
    assert result['asset_refreshes'][0]['initiator'] is None
    assert 'SECRET' not in json.dumps(result)


def test_malformed_telemetry_is_counted_and_duplicate_matching_records_rejected():
    result = editor_profile.summarize_editor_startup('##utp:{invalid\n' + info(), 12)
    assert result['malformed_telemetry_count'] == 1 and result['project_info_available'] is True
    with pytest.raises(ValueError):
        editor_profile.summarize_editor_startup(info() + '\n' + info(), 12)
    with pytest.raises(ValueError):
        editor_profile.summarize_editor_startup(info(), True)
