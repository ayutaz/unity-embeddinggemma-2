"""Audit collected Android Player reports, not device execution itself."""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from .android import FILES, _hex, _require
from .player import MODEL_REVISION


def _object(path):
    def reject_constant(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    def finite_float(value):
        parsed = float(value)
        _require(math.isfinite(parsed), 'JSON number is outside the finite range')
        return parsed
    def unique_fields(pairs):
        value = {}
        for key, entry in pairs:
            _require(key not in value, 'Duplicate JSON field: ' + key)
            value[key] = entry
        return value
    value = json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=reject_constant,
                       parse_float=finite_float, object_pairs_hook=unique_fields)
    _require(isinstance(value, dict), 'Expected a JSON object: ' + path.name)
    return value


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _nonnegative(value):
    return _number(value) and value >= 0


def _date(value):
    _require(isinstance(value, str), 'A completed UTC timestamp is required')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    _require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0, 'UTC timestamps are required')
    return parsed


def _rows(value, count):
    _require(isinstance(value, list) and len(value) == count and all(isinstance(row, dict) for row in value), 'All fixed reference rows are required')
    ids = [row.get('id') for row in value]
    _require(all(isinstance(name, str) and name for name in ids) and len(set(ids)) == count, 'Unique reference IDs are required')
    return ids


def _reference(path, descriptors, group, count, source):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    _require(path.stat().st_size == descriptors[path.name]['bytes'] and digest == descriptors[path.name]['sha256'], 'Reference differs from APK payload: ' + path.name)
    value = _object(path)
    _require(value.get('metadata', {}).get('source_commit') == source and value['metadata'].get('model_revision') == MODEL_REVISION, 'Reference provenance differs')
    _rows(value.get(group), count)
    return value


def audit_android_runs(reports, apk_audit, reference, search):
    reports = list(reports)
    _require(len(reports) == 2, 'Exactly two Player reports are required')
    apk = _object(Path(apk_audit))
    _require(apk.get('success') is True and apk.get('apk_payload_verified') is True and apk.get('abis') == ['arm64-v8a']
             and _hex(apk.get('source_commit'), 40) and _hex(apk.get('apk_sha256'), 64), 'A successful ARM64 APK payload audit is required')
    build_receipt = apk.get('build', {})
    build = build_receipt.get('build_info', {})
    _require(build_receipt.get('success') is True and build_receipt.get('injected_build') is False
             and build.get('target') == 'Android' and build.get('codeCommit') == apk['source_commit']
             and build.get('unityVersion') == '6000.3.16f1' and build.get('sentisVersion') == '2.6.1'
             and build.get('scriptingBackend') == 'IL2CPP' and build.get('stripping') == 'High'
             and isinstance(build.get('sourceSha256'), dict) and build['sourceSha256']
             and all(_hex(value, 64) for value in build['sourceSha256'].values()), 'Pinned real Android build provenance is required')
    bundle = apk.get('bundle', {})
    files = bundle.get('files')
    _require(bundle.get('success') is True and bundle.get('model_revision') == MODEL_REVISION
             and _hex(bundle.get('model_source'), 40) and _hex(bundle.get('search_source'), 40)
             and isinstance(files, list) and len(files) == 5 and all(isinstance(row, dict) for row in files)
             and [row.get('name') for row in files] == list(FILES)
             and all(type(row.get('bytes')) is int and row['bytes'] > 0 and _hex(row.get('sha256'), 64) for row in files), 'The fixed APK bundle is required')
    descriptors = {row['name']: row for row in files}
    _require(Path(reference).name == 'reference.json' and Path(search).name == 'search-reference.json', 'Fixed reference filenames are required')
    m1 = _reference(reference, descriptors, 'cases', 15, bundle['model_source'])
    search = _reference(search, descriptors, 'queries', 4, bundle['search_source'])
    document_ids = _rows(search.get('documents'), 6)
    for query in search['queries']:
        ranks = query.get('ranking')
        _require(isinstance(ranks, list) and len(ranks) == 6 and all(isinstance(row, dict) for row in ranks)
                 and sorted(row.get('document_id', '') for row in ranks) == sorted(document_ids)
                 and all(_number(row.get('score')) for row in ranks), 'The complete fixed document rankings are required')
    checked, previous_end, ids = [], None, []
    identity = None
    for index, path in enumerate(reports):
        run = _object(Path(path))
        _require(type(run.get('schema_version')) is int and run['schema_version'] == 1 and run.get('success') is True
                 and run.get('isEditor') is False and run.get('gpuVerified') is True and run.get('phase') == 'completed'
                 and run.get('error') is None and run.get('validationMode') == 'real_model' and run.get('unity') == '6000.3.16f1'
                 and type(run.get('tokenRows')) is int and run['tokenRows'] == 25, 'A completed real-model Player report with all tokens is required')
        _require(run.get('build') == build and run.get('bundle') == bundle, 'Player build/source hashes or bundle differ from the APK audit')
        _require(_hex(run.get('runId'), 32) and run['runId'] not in ids, 'Distinct fixed-format run IDs are required')
        start, end = _date(run.get('startedUtc')), _date(run.get('completedUtc'))
        _require(end >= start and (previous_end is None or start >= previous_end), 'Player reports must complete in order without overlap')
        previous_end = end
        current_identity = (run.get('os'), run.get('gpu'), run.get('graphicsApi'))
        _require(all(isinstance(value, str) and value.strip() for value in current_identity)
                 and current_identity[0].startswith('Android') and current_identity[2] in ('Vulkan', 'OpenGLES3'), 'Android OS/GPU and an intended graphics API are required')
        _require(identity is None or current_identity == identity, 'The two reports must use the same reported OS/GPU/API')
        identity = current_identity
        stage = run.get('staging', {})
        source = stage.get('source')
        _require(isinstance(source, str) and source.startswith('jar:file://') and source.endswith('!/assets/EmbeddingGemmaValidation')
                 and stage.get('transport') == 'UnityWebRequest.DownloadHandlerFile'
                 and stage.get('transferCompleted') is True and stage.get('auditPassed') is True
                 and stage.get('error') is None and stage.get('cacheRejectedError') is None
                 and stage.get('cacheReused') is bool(index) and type(stage.get('transferredFiles')) is int
                 and stage['transferredFiles'] == (6 if index == 0 else 1), 'Cold jar extraction then audited warm cache are required')
        for hashes in (run.get('hashBackends'), stage.get('hashBackends')):
            _require(isinstance(hashes, dict) and set(hashes) == set(FILES)
                     and all(value == 'dotnet_sha256' for value in hashes.values()), 'All five Android production full-hash backend records are required')
        _require(all(_nonnegative(run.get(name)) for name in ('auditMilliseconds', 'tokenizerMilliseconds'))
                 and all(_nonnegative(stage.get(name)) for name in ('milliseconds', 'auditMilliseconds')), 'Finite nonnegative staging/audit/token timings are required')
        conditions = run.get('conditions')
        expected = [(precision, backend) for precision in ('fp32', 'float16') for backend in ('CPU', 'GPUCompute')]
        _require(isinstance(conditions, list) and len(conditions) == 4 and all(isinstance(row, dict) for row in conditions)
                 and [(row.get('precision'), row.get('requestedBackend')) for row in conditions] == expected, 'All four distinct precision/backend conditions are required')
        for condition in conditions:
            precision, requested = condition['precision'], condition['requestedBackend']
            threshold, tolerance = (0.999, 0.002) if precision == 'fp32' else (0.99, 0.02)
            _require(condition.get('success') is True and condition.get('workerReleased') is True and condition.get('error') is None
                     and condition.get('actualBackend') == requested and all(type(condition.get(name)) is int and condition[name] == count
                         for name, count in (('caseCount', 15), ('documentCount', 6), ('queryCount', 4))), 'Every condition requires actual backend, complete rows, success and worker release')
            def cosine(value):
                return _number(value) and threshold <= value <= 1
            _require(cosine(condition.get('minimumCosine')), 'Condition cosine must meet the precision threshold')
            case_rows, query_rows = condition.get('cases'), condition.get('queries')
            _require(_rows(case_rows, 15) == [row['id'] for row in m1['cases']] and all(cosine(row.get('cosine')) for row in case_rows), 'Every M1 case cosine is required')
            _require(_rows(query_rows, 4) == [row['id'] for row in search['queries']], 'Every fixed search query is required')
            for actual, fixed in zip(query_rows, search['queries'], strict=True):
                ranks = actual.get('ranking')
                _require(cosine(actual.get('cosine')) and isinstance(ranks, list) and len(ranks) == 6
                         and all(isinstance(rank, dict) for rank in ranks), 'Each query cosine and all six ranks are required')
                _require([row.get('document_id') for row in ranks] == [row['document_id'] for row in fixed['ranking']]
                         and all(_number(row.get('score')) and abs(row['score'] - expected_row['score']) <= tolerance
                             for row, expected_row in zip(ranks, fixed['ranking'], strict=True)), 'Search ranking or scores differ from the fixed reference')
            _require(condition['minimumCosine'] <= min(row['cosine'] for row in case_rows + query_rows) + 1e-12, 'Minimum cosine contradicts individual results')
            warm = condition.get('warmInferenceMilliseconds')
            _require(isinstance(warm, list) and len(warm) == 3 and all(_nonnegative(value) for value in warm)
                     and all(_nonnegative(condition.get(name)) for name in ('modelLoadMilliseconds', 'workerCreationMilliseconds', 'firstInferenceMilliseconds', 'releaseMilliseconds')), 'Load/worker/first/warm/release timings are required')
        memory = run.get('memory')
        _require(isinstance(memory, list) and memory and all(isinstance(row, dict) and isinstance(row.get('stage'), str) and row['stage']
                 and type(row.get('managedHeapBytes')) is int and row['managedHeapBytes'] >= 0
                 and all(row.get(name) is None or (type(row[name]) is int and row[name] >= 0) for name in ('unityAllocatedBytes', 'processWorkingSetBytes')) for row in memory), 'Nonnegative memory samples or unavailable null counters are required')
        stages = ['after_cache_audit', 'after_token_audit'] + [prefix + precision + ':' + backend
            for precision, backend in expected for prefix in ('before_', 'worker_created_', 'released_')]
        _require([row['stage'] for row in memory] == stages, 'All fourteen cache/token/worker/release memory stages are required')
        _require(all(isinstance(run.get(name), str) and run[name] for name in ('memoryScope', 'timingScope')), 'Measurement scopes must be recorded')
        ids.append(run['runId'])
        checked.append(run)
    return {'success': True, 'player_reports_consistent': True, 'android_execution_performed_by_auditor': False,
            'independent_process_exits_verified': False, 'reported_gpu_pass': True,
            'source_commit': apk['source_commit'], 'apk_sha256': apk['apk_sha256'], 'run_ids': ids, 'runs': checked,
            'scope': 'Collected reports agree with the APK audit and packaged references. This auditor does not run ADB, prove device identity/process exits, or recompute embeddings; GPU pass is reported by the pinned Player.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs=2, type=Path, required=True)
    parser.add_argument('--apk-audit', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--search-reference', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        raise ValueError('Never overwrite an existing result')
    try:
        report = audit_android_runs(args.runs, args.apk_audit, args.reference, args.search_reference)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as error:
        report = {'success': False, 'player_reports_consistent': False, 'reported_gpu_pass': False,
                  'android_execution_performed_by_auditor': False, 'independent_process_exits_verified': False, 'error': str(error)}
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(text)
    print(text, end='')
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
