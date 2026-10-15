"""Audit saved raw-chain numeric evidence without loading or executing models.

The original pinned JSON, raw bytes and each actual carry are independent of
the producer's metric summary. This does not promote an export or qualify GPU.
"""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

from .audio_native_norm_replay import PRODUCER, REVISION, REFERENCE_SHA, AUDIT_SHA
from .reference_context_failure import feature_error_metrics

INTERMEDIATE_SHAPE = (1, 280, 1024)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(actual, expected):
    result = feature_error_metrics(torch.from_numpy(actual), torch.from_numpy(expected))
    result['exact_difference_count'] = int(np.count_nonzero(actual.view('u4') != expected.view('u4')))
    return result


def _cosine(actual, expected):
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return float(np.min(np.sum(a*b, axis=-1) / denominator)) if np.all(denominator > 0) else 0.0


def audit_report(report_path, original, *, expected_variant='ordered_original_tanh'):
    if expected_variant not in ('ordered_original_tanh','ordered_original_midpoint_bmm'):
        raise ValueError('Unregistered expected raw candidate')
    original = Path(original).resolve()
    for name, expected_sha in (('audio-encoder-reference.json', REFERENCE_SHA),
                               ('audio-encoder-validation.json', AUDIT_SHA)):
        p = original / name
        if not p.is_file() or digest(p) != expected_sha:
            raise ValueError('Original reference/audit hash mismatch')
    reference = json.loads((original/'audio-encoder-reference.json').read_text(encoding='utf-8'))
    manifest = json.loads((original/'audio-encoder-validation.json').read_text(encoding='utf-8'))
    metadata = reference['metadata']
    if (metadata != manifest['metadata'] or metadata['source_commit'] != PRODUCER
            or metadata['model_revision'] != REVISION or metadata['feature_atol'] != 1e-5
            or metadata['feature_rtol'] != 1e-4 or not manifest['native_reference_complete']):
        raise ValueError('Original reference context differs')
    report_path = Path(report_path).resolve(); directory = report_path.parent
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('scope') != 'raw_original_feature_input_connected_onnx_partitions_only'
            or not all(report.get(k) is True for k in ('complete','raw_input_mode',
                'raw_original_input_executed','native_chain_control_bits_equal'))
            or any(report.get(k) is not False for k in ('whole_encoder_validated','gpu_validated','end_to_end_validated'))):
        raise ValueError('Completed explicit raw execution and independent native control required')
    provenance = report['provenance']; case_id = provenance['case_id']; profile = provenance['profile']
    rows = [c for c in reference['cases'] if c['id'] == case_id]
    if len(rows) != 1 or rows[0]['profile'] != profile:
        raise ValueError('Original case/profile differs')
    row = rows[0]; spec = reference['profiles'][profile]
    for p in (provenance, report['input_provenance'], report['target_provenance']):
        if (p['reference_source'] != PRODUCER or p['model_revision'] != REVISION
                or p['reference_sha256'] != REFERENCE_SHA or p['audit_sha256'] != AUDIT_SHA
                or p['case_id'] != case_id or p['profile'] != profile
                or p['input_sha256'] != manifest['files'][row['input_features']]['sha256']
                or p['graph_sha256'] != manifest['files'][spec['native_graph']]['sha256']):
            raise ValueError('Original raw input/graph provenance differs')
    if re.fullmatch('[0-9a-f]{40}', provenance.get('job_source','')) is None:
        raise ValueError('Immutable job source required')
    boundaries = ['raw_features','view_2','mul_77','mul_152','mul_227',
                  'slice_51' if profile == 'frames-1119-valid-839' else 'select']
    if report['boundaries'] != boundaries:
        raise ValueError('Raw chain must retain all12 original layers and final crop')

    def capture(name, sha, shape):
        p = (directory/name).resolve()
        if (not p.is_relative_to(directory) or not p.is_file()
                or p.stat().st_size != int(np.prod(shape))*4 or digest(p) != sha):
            raise ValueError('Saved capture path/size/hash mismatch')
        array = np.fromfile(p, dtype='<f4').reshape(shape)
        if not np.isfinite(array).all():
            raise ValueError('Saved capture must be finite FP32')
        return array

    output_shape = (spec['feature_count'], spec['feature_width'])
    expected = np.asarray(row['features'],dtype=np.float32)
    if tuple(expected.shape) != output_shape:
        raise ValueError('Original final shape differs')
    capture('raw-input-fp32.bin',provenance['input_sha256'],spec['feature_shape'])
    if report['input_sha256'] != provenance['input_sha256']:
        raise ValueError('Original raw input capture differs')
    captured_reference = capture('reference-output-fp32.bin',report['reference_sha256'],output_shape)
    if not np.array_equal(captured_reference.view('u4'),expected.view('u4')):
        raise ValueError('Captured final reference differs from independent original bytes')
    native = capture('native-output-fp32.bin',report['native_sha256'],output_shape)
    native_metrics = _metrics(native,expected)
    if native_metrics != report['native_whole_metrics'] or report['context_qualified'] != (native_metrics['exact_difference_count']==0):
        raise ValueError('Independent native metrics/context differ')
    cases = report['cases']
    if len(cases)!=2 or [c['variant'] for c in cases] != ['native',expected_variant]:
        raise ValueError('Both native ONNX baseline and explicitly requested candidate required')
    audited=[]
    for c in cases:
        if (len(c['parts']) != 5 or c['coefficients_unchanged'] is not True
                or c['actual_carry'] != 'previous_onnx_output_without_frozen_stage_reset'):
            raise ValueError('All five original-weight raw parts and actual carry required')
        last = report['input_sha256']
        for i, part in enumerate(c['parts']):
            if (part['index'] != i or part['from_boundary'] != boundaries[i]
                    or part['to_boundary'] != boundaries[i+1]
                    or re.fullmatch('[0-9a-f]{64}',part['graph_sha256']) is None):
                raise ValueError('Original ordered part/graph evidence differs')
            capture(part['input_capture'],part['input_sha256'],spec['feature_shape'] if i==0 else INTERMEDIATE_SHAPE)
            if part['input_sha256'] != last:
                raise ValueError('Adjacent actual ONNX carry differs')
            actual = capture(part['output_capture'],part['output_sha256'],output_shape if i==4 else INTERMEDIATE_SHAPE)
            last=part['output_sha256']
        measured = _metrics(actual,expected)
        if measured != c['metrics']:
            raise ValueError('Independent final metrics differ from recorded summary')
        audited.append(dict(variant=c['variant'],metrics=measured,minimum_cosine=_cosine(actual,expected),
            final_sha256=last,vs_current_native=_metrics(actual,native)))
    candidate=audited[1]
    return dict(case_id=case_id,profile=profile,source=provenance['job_source'],candidate_variant=expected_variant,
        report_sha256=digest(report_path),raw_input_sha256=report['input_sha256'],
        reference_sha256=report['reference_sha256'],native_context_qualified=report['context_qualified'],
        native_whole_metrics=native_metrics,candidate_metrics=candidate['metrics'],
        candidate_vs_current_native=candidate['vs_current_native'],minimum_cosine=candidate['minimum_cosine'],
        original_tolerance_passed=candidate['metrics']['mismatched_elements']==0 and candidate['minimum_cosine']>=0.999999,
        values_compared=expected.size,all_capture_and_carry_hashes_verified=True,cases=audited,
        graphs_independently_rehashed=False,gpu_validated=False,default_export_promoted=False)


def audit_reports(paths, original, *, expected_variant='ordered_original_tanh'):
    options={} if expected_variant=='ordered_original_tanh' else {'expected_variant':expected_variant}
    rows = [audit_report(path, original, **options) for path in paths]
    signals = {'silence','impulse','sine','nyquist','noise','padded_noise'}
    if len(rows)!=6 or {r['case_id'] for r in rows} != signals:
        raise ValueError('Exactly six distinct original cases required')
    if any(r['profile'] != ('frames-1119-valid-839' if r['case_id']=='padded_noise'
                           else 'frames-1119-valid-1119') for r in rows):
        raise ValueError('Original five full / one padded profile coverage required')
    if len({r['source'] for r in rows}) != 1:
        raise ValueError('All six cases must use the same immutable source')
    return dict(scope='all_six_original_raw_input_connected_onnx_numeric_capture_audit_only',
        success=all(r['original_tolerance_passed'] for r in rows),case_count=6,profile_count=2,
        source=rows[0]['source'],candidate_variant=expected_variant,values_compared=sum(r['values_compared'] for r in rows),
        candidate_tolerance_failures=sum(r['candidate_metrics']['mismatched_elements'] for r in rows),
        candidate_bit_differences=sum(r['candidate_metrics']['exact_difference_count'] for r in rows),
        native_all_six_context_qualified=all(r['native_context_qualified'] for r in rows),cases=rows,
        gpu_validated=False,default_export_promoted=False)
