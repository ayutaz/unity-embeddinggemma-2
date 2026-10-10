import copy
import hashlib
import json

import pytest

from embeddinggemma_tools.android_runs import audit_android_runs, main
from embeddinggemma_tools.player import MODEL_REVISION


def fixture(tmp_path):
    reference = tmp_path / 'reference.json'
    search = tmp_path / 'search-reference.json'
    cases = [{'id': 'case-%02d' % i} for i in range(15)]
    ranking = [{'document_id': 'doc-%d' % i, 'score': 0.9 - i * 0.1} for i in range(6)]
    queries = [{'id': 'query-%d' % i, 'ranking': copy.deepcopy(ranking)} for i in range(4)]
    meta = {'model_revision': MODEL_REVISION, 'source_commit': 'b' * 40}
    reference.write_text(json.dumps({'metadata': meta, 'cases': cases}), encoding='utf-8')
    search_meta = dict(meta, source_commit='c' * 40)
    search.write_text(json.dumps({'metadata': search_meta, 'documents': [{'id': 'doc-%d' % i} for i in range(6)], 'queries': queries}), encoding='utf-8')
    descriptors = [{'name': name, 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
                   for name, body in [('model-fp32.sentis', b'fp32'), ('model-float16.sentis', b'fp16'), ('tokenizer.json', b'tokens'),
                                      ('reference.json', reference.read_bytes()), ('search-reference.json', search.read_bytes())]]
    bundle = {'success': True, 'model_revision': MODEL_REVISION, 'model_source': 'b' * 40, 'search_source': 'c' * 40, 'files': descriptors}
    build = {'codeCommit': 'a' * 40, 'target': 'Android', 'unityVersion': '6000.3.16f1', 'sentisVersion': '2.6.1',
             'scriptingBackend': 'IL2CPP', 'stripping': 'High', 'sourceSha256': {'runtime.cs': 'd' * 64}}
    apk = tmp_path / 'apk-audit.json'
    apk.write_text(json.dumps({'success': True, 'apk_payload_verified': True, 'source_commit': 'a' * 40, 'apk_sha256': 'e' * 64,
                              'abis': ['arm64-v8a'], 'build': {'success': True, 'injected_build': False, 'build_info': build}, 'bundle': bundle}), encoding='utf-8')
    hashes = {row['name']: 'dotnet_sha256' for row in descriptors}
    reports = []
    for index in range(2):
        conditions = []
        for precision in ('fp32', 'float16'):
            for backend in ('CPU', 'GPUCompute'):
                conditions.append({'success': True, 'workerReleased': True, 'error': None, 'precision': precision,
                    'requestedBackend': backend, 'actualBackend': backend, 'minimumCosine': 1.0,
                    'caseCount': 15, 'documentCount': 6, 'queryCount': 4,
                    'cases': [{'id': row['id'], 'cosine': 1.0} for row in cases],
                    'queries': [{'id': row['id'], 'cosine': 1.0, 'ranking': copy.deepcopy(ranking)} for row in queries],
                    'modelLoadMilliseconds': 1.0, 'workerCreationMilliseconds': 2.0, 'firstInferenceMilliseconds': 3.0,
                    'releaseMilliseconds': 0.1, 'warmInferenceMilliseconds': [1.0, 1.1, 1.2]})
        result = {'schema_version': 1, 'success': True, 'isEditor': False, 'gpuVerified': True, 'phase': 'completed', 'error': None,
            'validationMode': 'real_model', 'runId': ('1' if index == 0 else '2') * 32,
            'startedUtc': '2026-10-10T04:0%d:00Z' % index, 'completedUtc': '2026-10-10T04:0%d:30Z' % index,
            'unity': '6000.3.16f1', 'os': 'Android OS 15', 'gpu': 'Fixture GPU', 'graphicsApi': 'Vulkan',
            'tokenRows': 25, 'build': copy.deepcopy(build), 'bundle': copy.deepcopy(bundle),
            'auditMilliseconds': 10.0, 'tokenizerMilliseconds': 1.0, 'hashBackends': copy.deepcopy(hashes),
            'staging': {'source': 'jar:file:///data/app/test/base.apk!/assets/EmbeddingGemmaValidation',
                'directory': '/storage/emulated/0/Android/data/com.ayutaz.embeddinggemma.validation/files/EmbeddingGemmaValidation/BundleCache/active',
                'error': None, 'cacheRejectedError': None, 'transport': 'UnityWebRequest.DownloadHandlerFile',
                'transferCompleted': True, 'auditPassed': True, 'cacheReused': bool(index), 'transferredFiles': 6 if index == 0 else 1,
                'milliseconds': 20.0, 'auditMilliseconds': 10.0, 'hashBackends': copy.deepcopy(hashes)},
            'conditions': conditions, 'memoryScope': 'Application samples, GPU memory unknown', 'timingScope': 'Query warmup then three inferences',
            'memory': [{'stage': stage, 'managedHeapBytes': 123, 'unityAllocatedBytes': 456, 'processWorkingSetBytes': None}
                for stage in ['after_cache_audit', 'after_token_audit'] + [prefix + precision + ':' + backend
                    for precision in ('fp32', 'float16') for backend in ('CPU', 'GPUCompute')
                    for prefix in ('before_', 'worker_created_', 'released_')]]}
        path = tmp_path / ('run%d.json' % (index + 1))
        path.write_text(json.dumps(result), encoding='utf-8')
        reports.append(path)
    return reports, apk, reference, search


def mutate(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_audits_two_reports_without_claiming_an_adb_execution(tmp_path):
    result = audit_android_runs(*fixture(tmp_path))
    assert result['success'] and result['player_reports_consistent']
    assert result['android_execution_performed_by_auditor'] is False
    assert result['independent_process_exits_verified'] is False
    assert result['reported_gpu_pass'] is True
    assert result['run_ids'] == ['1' * 32, '2' * 32]
    assert result['source_commit'] == 'a' * 40
    assert len(result['runs']) == 2


@pytest.mark.parametrize('damage', ['failed', 'injected', 'editor', 'windows', 'commit', 'runtime_hash', 'missing_condition',
    'duplicate_condition', 'cpu_fallback', 'unreleased', 'missing_case', 'case_cosine', 'nan_cosine', 'ranking', 'score',
    'missing_query', 'warm_count', 'nan_timing', 'negative_memory', 'no_jar', 'cache_count', 'cache_audit', 'tokens',
    'incomplete', 'graphics', 'gpu_missing', 'same_run', 'overlap', 'bundle', 'hash_backend', 'injected_hash', 'missing_memory_stage'])
def test_rejects_partial_or_inconsistent_device_reports(tmp_path, damage):
    args = fixture(tmp_path)
    def change(row):
        condition = row['conditions'][0]
        if damage == 'failed': row['success'] = False
        elif damage == 'injected': row['validationMode'] = 'injected_contract'
        elif damage == 'editor': row['isEditor'] = True
        elif damage == 'windows': row['build']['target'] = 'StandaloneWindows64'
        elif damage == 'commit': row['build']['codeCommit'] = 'f' * 40
        elif damage == 'runtime_hash': row['build']['sourceSha256']['runtime.cs'] = 'f' * 64
        elif damage == 'missing_condition': row['conditions'].pop()
        elif damage == 'duplicate_condition': row['conditions'][1] = copy.deepcopy(condition)
        elif damage == 'cpu_fallback': row['conditions'][1]['actualBackend'] = 'CPU'
        elif damage == 'unreleased': condition['workerReleased'] = False
        elif damage == 'missing_case': condition['cases'].pop()
        elif damage == 'case_cosine': condition['cases'][0]['cosine'] = 0.98
        elif damage == 'nan_cosine': condition['minimumCosine'] = float('nan')
        elif damage == 'ranking': condition['queries'][0]['ranking'].reverse()
        elif damage == 'score': condition['queries'][0]['ranking'][0]['score'] += 0.1
        elif damage == 'missing_query': condition['queries'].pop()
        elif damage == 'warm_count': condition['warmInferenceMilliseconds'] = [1.0]
        elif damage == 'nan_timing': condition['modelLoadMilliseconds'] = float('nan')
        elif damage == 'negative_memory': row['memory'][0]['managedHeapBytes'] = -1
        elif damage == 'no_jar': row['staging']['source'] = 'file:///models'
        elif damage == 'cache_count': row['staging']['transferredFiles'] = 6
        elif damage == 'cache_audit': row['staging']['auditPassed'] = False
        elif damage == 'tokens': row['tokenRows'] = 15
        elif damage == 'incomplete': row['completedUtc'] = None
        elif damage == 'graphics': row['graphicsApi'] = 'Direct3D12'
        elif damage == 'gpu_missing': row['gpu'] = ''
        elif damage == 'same_run': row['runId'] = '1' * 32
        elif damage == 'overlap': row['startedUtc'] = '2026-10-10T04:00:01Z'
        elif damage == 'bundle': row['bundle']['model_revision'] = 'f' * 40
        elif damage == 'injected_hash': row['hashBackends']['tokenizer.json'] = 'injected_accelerated'
        elif damage == 'missing_memory_stage': row['memory'].pop()
        else: row['hashBackends'].pop('tokenizer.json')
    mutate(args[0][1], change)
    with pytest.raises(ValueError):
        audit_android_runs(*args)


@pytest.mark.parametrize('damage', ['apk_failed', 'injected_build', 'reference', 'search', 'run_count'])
def test_requires_payload_audit_and_the_exact_packaged_references(tmp_path, damage):
    args = fixture(tmp_path)
    if damage == 'apk_failed': mutate(args[1], lambda row: row.update(success=False))
    elif damage == 'injected_build': mutate(args[1], lambda row: row['build'].update(injected_build=True))
    elif damage == 'reference': args[2].write_text('{}')
    elif damage == 'search': args[3].write_text('{}')
    else: args[0].pop()
    with pytest.raises(ValueError):
        audit_android_runs(*args)


@pytest.mark.parametrize('failed', [False, True])
def test_cli_saves_consistent_success_or_failure_without_claiming_execution(tmp_path, capsys, failed):
    reports, apk, reference, search = fixture(tmp_path)
    if failed:
        mutate(reports[0], lambda row: row.update(success=False))
    output = tmp_path / 'report-audit.json'
    args = ['--runs', *map(str, reports), '--apk-audit', str(apk), '--reference', str(reference), '--search-reference', str(search), '--output', str(output)]
    assert main(args) == (1 if failed else 0)
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads(output.read_text())
    assert result['success'] is (not failed)
    assert result['android_execution_performed_by_auditor'] is False
    if failed:
        assert result['reported_gpu_pass'] is False


def test_cli_preserves_existing_receipt(tmp_path):
    reports, apk, reference, search = fixture(tmp_path)
    output = tmp_path / 'existing.json'
    output.write_text('keep')
    with pytest.raises(ValueError, match='overwrite'):
        main(['--runs', *map(str, reports), '--apk-audit', str(apk), '--reference', str(reference), '--search-reference', str(search), '--output', str(output)])
    assert output.read_text() == 'keep'


@pytest.mark.parametrize('damage', ['nonfinite_extra', 'overflow_extra', 'duplicate_field'])
def test_cli_rejects_ambiguous_or_nonfinite_json_and_saves_failure(tmp_path, capsys, damage):
    reports, apk, reference, search = fixture(tmp_path)
    if damage == 'nonfinite_extra':
        mutate(reports[0], lambda row: row.update(extra=float('nan')))
    elif damage == 'overflow_extra':
        reports[0].write_text('{"extra": 1e999, ' + reports[0].read_text()[1:])
    else:
        reports[0].write_text('{"success": false, ' + reports[0].read_text()[1:])
    output = tmp_path / 'rejected.json'
    assert main(['--runs', *map(str, reports), '--apk-audit', str(apk), '--reference', str(reference), '--search-reference', str(search), '--output', str(output)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads(output.read_text())
    assert result['success'] is False and result['reported_gpu_pass'] is False
