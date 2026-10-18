"""Summarize saved Unity Editor reload timings without executing Unity."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median


# Export only known profiler labels. Arbitrary log text can contain private paths
# or licensing information. These nested measurements overlap; never sum them.
STAGES = frozenset({
    'BeginReloadAssembly', 'ExecutionOrderSort', 'DisableScriptedObjects',
    'BackupInstance', 'ReleaseScriptingObjects', 'CreateAndSetChildDomain',
    'RebuildCommonClasses', 'RebuildNativeTypeToScriptingClass',
    'initialDomainReloadingComplete', 'LoadAllAssembliesAndSetupDomain',
    'LoadAssemblies', 'RebuildTransferFunctionScriptingTraits', 'AnalyzeDomain',
    'TypeCache.Refresh', 'TypeCache.ScanAssembly', 'BuildScriptInfoCaches',
    'ResolveRequiredComponents', 'FinalizeReload', 'ReleaseScriptCaches',
    'RebuildScriptCaches', 'SetupLoadedEditorAssemblies', 'LogAssemblyErrors',
    'InitializePlatformSupportModulesInManaged', 'SetLoadedEditorAssemblies',
    'BeforeProcessingInitializeOnLoad', 'ProcessInitializeOnLoadAttributes',
    'ProcessInitializeOnLoadMethodAttributes', 'AfterProcessingInitializeOnLoad',
    'EditorAssembliesLoaded', 'ExecutionOrderSort2', 'AwakeInstancesAfterBackupRestoration',
})


def _stats(values):
    return {'min': min(values), 'median': median(values), 'max': max(values)} if values else None


def summarize_editor_startup(text, pid):
    """Keep startup telemetry and adjacent import counts separate from inference."""
    if type(pid) is not int or pid <= 0:
        raise ValueError('A positive exact process ID is required')
    metrics = ('projectLoad', 'projectInit', 'templateInit', 'packageManagerInit',
               'assetDatabaseInit', 'globalIlluminationInit', 'assembliesLoad',
               'unityExtensionsInit', 'assetDatabaseRefresh', 'sceneOpening')
    project = None
    malformed = 0
    refreshes = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith('##utp:'):
            try:
                value = json.loads(line[6:])
            except json.JSONDecodeError:
                malformed += 1
                continue
            if (isinstance(value, dict) and value.get('type') == 'ProjectInfo'
                    and type(value.get('processId')) is int and value['processId'] == pid):
                if project is not None:
                    raise ValueError('Duplicate matching ProjectInfo records')
                project = {}
                for name in metrics:
                    seconds = value.get(name)
                    project[name] = seconds if (type(seconds) in (int, float)
                        and math.isfinite(seconds) and seconds >= 0) else None
        match = re.fullmatch(r'Asset Pipeline Refresh .*?: Total: (\d+(?:\.\d+)?) seconds - Initiated by (.*)', line)
        if not match:
            continue
        initiator = re.match(r'(InitialRefreshV2|RefreshV2|StopAssetImportingV2)\(', match[2])
        imports = None
        if index + 2 < len(lines) and lines[index + 1].strip() == 'Summary:':
            counts = re.fullmatch(r'\s+Imports: total=(\d+) \(actual=(\d+), local cache=(\d+), cache server=(\d+)\)', lines[index + 2])
            if counts:
                imports = dict(zip(('total', 'actual', 'local_cache', 'cache_server'), map(int, counts.groups())))
        seconds = float(match[1])
        refreshes.append(dict(line=index + 1, seconds=seconds if math.isfinite(seconds) else None,
                              initiator=initiator[1] if initiator else None, imports=imports))
    return dict(pid=pid, project_info_seconds=project,
        project_info_available=project is not None and all(v is not None for v in project.values()),
        malformed_telemetry_count=malformed, asset_refreshes=refreshes,
        timings_additive=False, search_database_duration_seconds=None, model_load_seconds=None,
        scope='Exact-PID Editor startup telemetry and adjacent refresh import counts. '
              'Nested startup, refresh and reload durations overlap; do not sum them. '
              'Not model inference, clean-cache import qualification or search database timing.')


def summarize_editor_log(text):
    profiles, refresh = [], []
    pending = None
    active_profile = None
    incomplete = unpaired = search_errors = allocation = 0
    for number, line in enumerate(text.splitlines(), 1):
        if line == 'Begin MonoManager ReloadAssembly':
            incomplete += pending is not None
            pending = number
        matched = re.fullmatch(r'Domain Reload Profiling: (\d+)ms', line)
        if matched:
            unpaired += pending is None
            active_profile = {'begin_line': pending, 'profile_line': number,
                              'total_ms': int(matched[1]), 'stages_ms': {}}
            profiles.append(active_profile)
            pending = None
        else:
            stage = re.fullmatch(r'\s+([A-Za-z][A-Za-z0-9.]*) \((\d+)ms\)', line)
            if active_profile is not None and stage:
                if stage[1] in STAGES:
                    active_profile['stages_ms'][stage[1]] = int(stage[2])
            else:
                active_profile = None
        matched = re.match(r'Asset Pipeline Refresh .*?: Total: (\d+(?:\.\d+)?) seconds\b', line)
        if matched:
            refresh.append(float(matched[1]))
        search_errors += 'at UnityEditor.Search.SearchDatabase+<EnumerateAll>' in line
        allocation += line.startswith('Leak Detected : Persistent allocates ')
    incomplete += pending is not None
    refresh_stats = _stats(refresh)
    return {
        'schema_version': 1,
        'completed_reload_count': len(profiles), 'incomplete_reload_count': incomplete,
        'unpaired_profile_count': unpaired,
        'observation_complete': bool(profiles) and incomplete == 0 and unpaired == 0,
        'completed_reload_ms': _stats([profile['total_ms'] for profile in profiles]),
        'profiles': profiles,
        'asset_refresh_seconds': {'count': len(refresh), **refresh_stats} if refresh_stats else None,
        'search_database_enumeration_exception_count': search_errors,
        'persistent_allocation_diagnostic_count': allocation,
        'unity_execution_performed': False, 'performance_improvement_verified': False,
        'm2_release_verified': False,
        'scope': 'Historical log markers only. Nested stage timings overlap and are not additive. '
                 'An incomplete reload has no duration estimate and does not prove a live hang. '
                 'A complete log observation does not establish error-free Unity, runtime acceptance, '
                 'improved performance, or the cause of earlier hangs.',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        raw = args.log.read_bytes()
        result = summarize_editor_log(raw.decode('utf-8-sig', errors='replace'))
    except OSError as error:
        print(json.dumps({'observation_complete': False, 'error_type': type(error).__name__}))
        return 1
    result['log_sha256'] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, indent=2))
    return 0 if result['observation_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
