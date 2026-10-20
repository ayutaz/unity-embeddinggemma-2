"""Actual FP32 image CPU reference chains with retained intermediate captures.

This is a comparison harness, not a CPU backend or fallback for the GPU SDK.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

from . import image_reference_cohort as references
from . import reference_cpu_performance as protocol
from .cpu_execution_context import cpu_execution_context
from .native_regression_distribution import _empty


NAMES = ('vision-reference.json', 'vision-validation.json', 'image-embedding-reference.json')
STAGES = ('preprocessing', 'encoder', 'backbone')


def _require(condition):
    if not condition:
        raise ValueError('Image CPU reference contract differs')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(path, expected=None):
    raw = Path(path).read_bytes()
    _require(expected is None or _sha(raw) == expected)
    return json.loads(raw)


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _metadata(root):
    cohort = references.load_image_reference_cohort(*(Path(root) / name for name in NAMES))
    encoder = _json(Path(root) / NAMES[0], cohort.encoder_reference_sha256)
    audit = _json(Path(root) / NAMES[1], references.ENCODER_AUDIT_SHA256)
    document = _json(Path(root) / NAMES[2], cohort.document_reference_sha256)
    metadata = encoder['metadata']
    _require(metadata['feature_rtol'] == 1e-4 and metadata['feature_atol'] == 1e-5
             and metadata['minimum_cosine'] == .999999)
    # Do not retain the large JSON feature lists: the cohort owns typed copies.
    return cohort, encoder['profiles'], audit['files'], document['metadata'], document['profiles']


def _validate_execution_source(source):
    root = Path(__file__).resolve().parents[2]
    _require(isinstance(source, str) and re.fullmatch('[0-9a-f]{40}', source) is not None)
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, stderr=subprocess.PIPE).decode().strip()
    _require(source == actual)
    names = ('image_cpu_reference.py', 'cpu_execution_context.py', 'image_reference_cohort.py', 'reference_cpu_performance.py',
             'model.py', 'prepare.py', 'export.py', 'vision_cpu_simd_reference.py')
    paths = ['tools/embeddinggemma_tools/' + name for name in names] + ['tools/uv.lock', 'tools/pyproject.toml']
    fingerprints = {}
    for name in paths:
        raw = (root / name).read_bytes().replace(b'\r\n', b'\n')
        expected = subprocess.check_output(['git', 'show', source + ':' + name], cwd=root, stderr=subprocess.PIPE).replace(b'\r\n', b'\n')
        _require(raw == expected)
        fingerprints[name] = _sha(raw)
    return fingerprints


def _load_pipeline(snapshot, metadata, profile, *, statistics_reference='native'):
    from .model import load_image_embedding_model
    from .prepare import load_image_processor
    from .export import MultimodalEmbeddingExport
    core = load_image_embedding_model(snapshot)

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = core.vision_tower
            self.projection = core.embed_vision

        def forward(self, value):
            return self.projection(self.vision(pixel_values=value['pixel_values'],
                pixel_position_ids=value['image_position_ids']).last_hidden_state)

    backbone = MultimodalEmbeddingExport(core.language_model, profile['sequence_length'],
        image_token_id=metadata['image_token_id'], audio_token_id=metadata['audio_token_id'],
        image_feature_count=profile['image_feature_count'], audio_feature_count=0)
    encoder=Encoder().eval()
    encoder.statistics_reference_configuration=dict(policy='native',selected_norms=0,replacements=[])
    if statistics_reference=='sixteen-lane':
        from .vision_cpu_simd_reference import apply_vision_statistics_reference
        encoder.statistics_reference_configuration=apply_vision_statistics_reference(encoder)
    elif statistics_reference!='native':raise ValueError('Unknown CPU image statistics reference')
    return load_image_processor(snapshot), encoder, backbone.eval()


def _validate_statistics_configuration(mode,configuration):
    _require(mode in ('native','sixteen-lane'))
    _require(configuration['policy']==('native' if mode=='native' else 'sixteen-lane-vision'))
    _require(configuration['selected_norms']==len(configuration['replacements']))
    if mode=='native':_require(configuration['selected_norms']==0)
    else:
        _require(configuration['selected_norms']>0)
        _require(len({row['name'] for row in configuration['replacements']})==configuration['selected_norms'])
        for row in configuration['replacements']:
            _require(row['name'].startswith('vision.') and row['width'] in (64,768))
            _require(type(row['scaled']) is bool and row['scale_borrowed'] is row['scaled'])


def _peak_rss_bytes():
    """Whole-process Linux RSS peak or Windows OS peak working-set bytes."""
    if sys.platform == 'win32':
        from .cpu_progress import own_process_memory
        observed = own_process_memory()['peak_rss_bytes']
        return observed if type(observed) is int and observed > 0 else None
    if sys.platform != 'linux':
        return None
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _tensor_bytes(value, dtype):
    return value.detach().contiguous().numpy().astype(dtype, copy=False).tobytes()


def _capture(output, name, value, dtype):
    protocol._tensor(value)
    if dtype == '<i4':
        _require(value.dtype in (torch.int32, torch.int64) and value.numel() > 0)
        _require(int(value.min()) >= -(2**31) and int(value.max()) < 2**31)
    else:
        _require(value.dtype == torch.float32)
    raw = _tensor_bytes(value, dtype)
    with (output / name).open('xb') as stream:
        stream.write(raw)
    return dict(file=name, bytes=len(raw), sha256=_sha(raw), shape=list(value.shape), encoding=dtype)


def _read_capture(output, entry, name, shape, dtype):
    _require(entry['file'] == name and entry['shape'] == list(shape) and entry['encoding'] == dtype)
    raw = (output / name).read_bytes()
    _require(len(raw) == math.prod(shape) * 4 == entry['bytes'] and _sha(raw) == entry['sha256'])
    values = np.frombuffer(raw, dtype=dtype).reshape(shape)
    _require(bool(np.isfinite(values).all()))
    return raw, values


def _encoder_result(actual, expected):
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    violations = int(np.count_nonzero(difference > 1e-5 + 1e-4 * np.abs(expected.astype(np.float64))))
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    norms = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosine = np.zeros_like(norms)
    np.divide((a * b).sum(axis=-1), norms, out=cosine, where=norms > 0)
    cosine[(np.linalg.norm(a, axis=-1) == 0) & (np.linalg.norm(b, axis=-1) == 0)] = 1
    return dict(tolerance_failures=violations, maximum_absolute_error=float(difference.max()),
                minimum_cosine=float(cosine.min()))


def _audit_rounds(output, rows, case, spec, files):
    _require(len(rows) == 4)
    result = []
    for name, row in zip(protocol.ROUNDS, rows, strict=True):
        _require(row['round'] == name)
        captures = row['captures']
        _require(set(captures) == {'pixels', 'positions', 'encoder'})
        pixel_raw, _ = _read_capture(output, captures['pixels'], name + '-pixels.bin', spec['pixel_shape'], '<f4')
        pos_raw, _ = _read_capture(output, captures['positions'], name + '-positions.bin', spec['positions_shape'], '<i4')
        _, actual = _read_capture(output, captures['encoder'], name + '-encoder.bin', case.encoder_features.shape, '<f4')
        encoder = _encoder_result(actual, case.encoder_features)
        row = dict(round=name, pixels_exact=_sha(pixel_raw) == case.pixels_sha256,
                   positions_exact=_sha(pos_raw) == files[spec['positions']]['sha256'], **encoder)
        row['success'] = (row['pixels_exact'] and row['positions_exact'] and
                          encoder['tolerance_failures'] == 0 and encoder['minimum_cosine'] >= .999999)
        result.append(row)
    return result


def measure_image_cpu_reference(snapshot, reference_directory, profile, output, *, source, fixture_root=None,statistics_reference='native'):
    """Measure four actual carries; validate and persist intermediates afterwards."""
    try:
        _require(statistics_reference in ('native','sixteen-lane'))
        output = _empty(output)
        _require(profile in references.PROFILES and not torch.cuda.is_initialized())
        fingerprints = _validate_execution_source(source)
        cohort, profiles, files, metadata, final_profiles = _metadata(reference_directory)
        case = cohort.case(profile + '-0')
        snapshot = Path(snapshot)
        for name, expected in metadata['input_sha256'].items():
            with (snapshot / name).open('rb') as stream:
                _require(hashlib.file_digest(stream, 'sha256').hexdigest() == expected)
        fixture_root = Path(fixture_root) if fixture_root is not None else Path(__file__).parent / 'fixtures/vision'
        image_path = fixture_root / case.image
        raw = image_path.read_bytes()
        _require(len(raw) == files[case.image]['bytes'] and _sha(raw) == files[case.image]['sha256'])
        with Image.open(image_path) as image:
            _require(image.mode == 'RGB')
            rgb = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).contiguous()
        ids = torch.tensor([case.input_ids], dtype=torch.int64)
        mask = torch.tensor([case.attention_mask], dtype=torch.int64)
        empty = torch.empty(0, 512)
        context_before = cpu_execution_context()
        output.mkdir(parents=True, exist_ok=True)
        admission = dict(schema=2, source=source, profile=profile, phase='loading',statistics_reference=statistics_reference,
            model_execution_complete=False, error_class=None,
            execution_context_before_load=context_before,
            source_fingerprints_normalized_LF=fingerprints,
            model_revision=references.MODEL_REVISION, snapshot_input_sha256=metadata['input_sha256'],
            encoder_reference_sha256=cohort.encoder_reference_sha256,
            document_reference_sha256=cohort.document_reference_sha256,
            CPU_comparison_qualified=False, complete_performance_accepted=False)
        source_path = output / 'image-cpu-source.json'
        _write(source_path, admission)
        start, start_cpu = time.perf_counter(), time.process_time()
        try:
            options={} if statistics_reference=='native' else dict(statistics_reference=statistics_reference)
            processor, encoder, backbone = _load_pipeline(snapshot, metadata, final_profiles[profile],**options)
        except Exception as error:
            admission.update(phase='failed_loading', error_class=type(error).__name__)
            source_path.write_text(json.dumps(admission, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            raise ValueError('Image CPU model loading failed') from None
        load_wall, load_cpu = time.perf_counter() - start, time.process_time() - start_cpu
        configuration=getattr(encoder,'statistics_reference_configuration',dict(policy='native',selected_norms=0,replacements=[]))
        _validate_statistics_configuration(statistics_reference,configuration)
        admission.update(phase='measuring',statistics_reference_configuration=configuration)
        source_path.write_text(json.dumps(admission, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        trace = dict(pixels=[], positions=[], encoder=[])

        def preprocess():
            value = processor(images=rgb, return_tensors='pt')
            return {key: value[key] for key in ('pixel_values', 'image_position_ids')}

        def encode(value):
            actual = encoder(value)
            return actual

        def finish(actual):
            return backbone(ids, mask, actual, empty)

        def observe(stage, value):
            # Snapshot after its timer stops and before the next consumer runs.
            # The actual value continues downstream; a snapshot never replaces it.
            if stage == 'preprocessing':
                trace['pixels'].append(value['pixel_values'].detach().clone())
                trace['positions'].append(value['image_position_ids'].detach().clone())
            elif stage == 'encoder':
                trace['encoder'].append(value.detach().clone())

        def signature():
            digest = hashlib.sha256(image_path.read_bytes())
            for value in (rgb, ids, mask):
                digest.update(value.contiguous().numpy().tobytes())
            return digest.hexdigest()

        measured = protocol.CpuReferenceCase(module='image-' + profile, precision='fp32',
            descriptor_sha256=protocol.DESCRIPTOR, reference_sha256=cohort.document_reference_sha256,
            input_sha256=_sha(raw), reference_case_id=case.id, source=source,
            expected=torch.from_numpy(case.embedding.copy()), input_signature=signature,
            preprocess=preprocess, encoder=encode, backbone=finish, active_modules=(encoder, backbone), after_stage=observe)
        error_class = None
        try:
            pipeline = protocol.measure_reference_pipeline(measured, output / 'pipeline.json')
        except Exception as error:
            error_class = type(error).__name__
            pipeline = _json(output / 'pipeline.json') if (output / 'pipeline.json').is_file() else dict(success=False)
        rows = []
        admission.update(phase='persisting_captures')
        source_path.write_text(json.dumps(admission, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        # All disk writes and intermediate inspections are outside measured calls.
        try:
            for index, name in enumerate(protocol.ROUNDS):
                if index >= min(len(values) for values in trace.values()):
                    break
                captures = {key: _capture(output, name + '-' + key + '.bin', values[index],
                            '<i4' if key == 'positions' else '<f4') for key, values in trace.items()}
                rows.append(dict(round=name, captures=captures))
        except Exception as error:
            admission.update(phase='failed_capture', error_class=type(error).__name__)
            source_path.write_text(json.dumps(admission, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            raise ValueError('Image CPU capture could not preserve actual values') from None
        inspections = []
        try:
            inspections = _audit_rounds(output, rows, case, profiles[profile], files)
        except ValueError:
            error_class = error_class or 'ValueError'
        peak = _peak_rss_bytes()
        context_after = cpu_execution_context()
        release_started,release_cpu_started=time.perf_counter(),time.process_time()
        processor=encoder=backbone=measured=None
        gc.collect()
        release_wall,release_cpu=time.perf_counter()-release_started,time.process_time()-release_cpu_started
        success = bool(pipeline['success'] and len(inspections) == 4 and
                       all(row['success'] for row in inspections) and not torch.cuda.is_initialized()
                       and context_before == context_after
                       and (peak is None or 0 < peak <= 16 * 1024**3))
        result = dict(schema=2, success=success, source=source, profile=profile, case=case.id,
            statistics_reference=statistics_reference,statistics_reference_configuration=configuration,
            scope='Four actual FP32 PyTorch CPU image chains against existing GPU encoder and original Document final',
            actual_backend='CPU', framework='PyTorch reference', precision='fp32',
            descriptor_sha256=protocol.DESCRIPTOR, encoder_reference_sha256=cohort.encoder_reference_sha256,
            document_reference_sha256=cohort.document_reference_sha256,
            document_encoder_generation_sha256=cohort.document_encoder_source_sha256,
            input_sha256=_sha(raw), rounds=rows, intermediate_inspections=inspections,
            final_vector_gate_passed=bool(pipeline['success']), error_class=error_class,
            load_wall_seconds=load_wall, load_process_cpu_seconds=load_cpu,
            release_wall_seconds=release_wall,release_process_cpu_seconds=release_cpu,release_completed=True,
            release_scope='Drop owning model references and collect Python cycles; not allocator zero-byte proof',
            host_peak_rss_bytes=peak, host_peak_scope='Whole process lifetime peak RSS on Linux or peak working set on Windows; unavailable on unsupported hosts',
            explicit_host_limit_bytes=16 * 1024**3, torch_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(), CPU_capability=torch.backends.cpu.get_cpu_capability(),
            runtime_versions={name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy', 'pillow')},
            source_fingerprints_normalized_LF=fingerprints, CUDA_context_initialized=torch.cuda.is_initialized(),
            execution_context_before_load=context_before, execution_context_after_execution=context_after,
            execution_context_stable=context_before == context_after,
            original_golden_changed=False, CPU_comparison_qualified=False, complete_performance_accepted=False)
        _write(output / 'image-cpu-reference.json', result)
        admission.update(phase='complete' if success else 'failed_chain',
            model_execution_complete=len(pipeline.get('rounds', [])) == 4, error_class=error_class,
            execution_context_after_execution=context_after)
        source_path.write_text(json.dumps(admission, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        _require(success)
        return result
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
        raise ValueError('Image CPU reference chain failed; preserve existing outputs') from None


def audit_image_cpu_reference(output, reference_directory, *, source):
    """Recompute numeric gates from all retained raw captures, not reported counts."""
    try:
        output = Path(output)
        cohort, profiles, files, final_metadata, _ = _metadata(reference_directory)
        report, pipeline = _json(output / 'image-cpu-reference.json'), _json(output / 'pipeline.json')
        admission = _json(output / 'image-cpu-source.json')
        _require(report['schema'] == admission['schema'] == 2)
        if 'statistics_reference' in report:
            mode=report['statistics_reference'];configuration=report['statistics_reference_configuration']
            _validate_statistics_configuration(mode,configuration)
            _require(admission['statistics_reference']==mode and admission['statistics_reference_configuration']==configuration)
            _require(report['release_completed'] is True and all(math.isfinite(report[key]) and report[key]>=0 for key in ('load_wall_seconds','load_process_cpu_seconds','release_wall_seconds','release_process_cpu_seconds')))
        _require(isinstance(source, str) and re.fullmatch('[0-9a-f]{40}', source) is not None)
        _require(report['success'] is True and pipeline['success'] is True and
                 report['source'] == pipeline['source'] == source and report['actual_backend'] == pipeline['actual_backend'] == 'CPU')
        case = cohort.case(report['case'])
        profile = report['profile']
        _require(profile in references.PROFILES and case.id == profile + '-0' and
                 report['encoder_reference_sha256'] == cohort.encoder_reference_sha256 and
                 report['document_reference_sha256'] == pipeline['reference_sha256'] == cohort.document_reference_sha256 and
                 report['document_encoder_generation_sha256'] == cohort.document_encoder_source_sha256 and
                 report['descriptor_sha256'] == pipeline['descriptor_sha256'] == protocol.DESCRIPTOR and
                 report['input_sha256'] == pipeline['input_sha256'] == files[case.image]['sha256'] and
                 report['precision'] == pipeline['precision'] == 'fp32' and
                 report['CUDA_context_initialized'] is False and report['complete_performance_accepted'] is False)
        _require(pipeline['phase'] == 'complete' and pipeline['module'] == 'image-' + profile and
                 pipeline['reference_case_id'] == case.id and
                 pipeline['input_signatures_checked_before_and_after_each_round'] is True and
                 pipeline['stage_observer_present'] is True and pipeline['stage_observer_time_excluded'] is True)
        _require(admission['source'] == source and admission['profile'] == profile and
                 admission['phase'] == 'complete' and admission['model_execution_complete'] is True and
                 admission['error_class'] is None and admission['model_revision'] == references.MODEL_REVISION and
                 admission['snapshot_input_sha256'] == final_metadata['input_sha256'] and
                 admission['source_fingerprints_normalized_LF'] == report['source_fingerprints_normalized_LF'] and
                 admission['encoder_reference_sha256'] == cohort.encoder_reference_sha256 and
                 admission['document_reference_sha256'] == cohort.document_reference_sha256)
        _require(report['execution_context_stable'] is True and
                 report['execution_context_before_load'] == report['execution_context_after_execution'] ==
                 admission['execution_context_before_load'] == admission['execution_context_after_execution'])
        rows = _audit_rounds(output, report['rounds'], case, profiles[profile], files)
        _require(all(row['success'] for row in rows) and len(pipeline['rounds']) == 4)
        finals = []
        wanted = case.embedding.astype(np.float64)
        wanted_norm = float(wanted @ wanted)
        for name, row in zip(protocol.ROUNDS, pipeline['rounds'], strict=True):
            _require(row['round'] == name and row['capture'] == 'pipeline-' + name + '.bin')
            raw = (output / row['capture']).read_bytes()
            _require(len(raw) == 768 * 4 and _sha(raw) == row['capture_sha256'])
            actual = np.frombuffer(raw, dtype='<f4').astype(np.float64)
            norm = float(actual @ actual)
            cosine = float(actual @ wanted) / math.sqrt(norm * wanted_norm) if norm > 0 else 0.
            _require(bool(np.isfinite(actual).all()) and abs(norm - 1) <= .002 and cosine >= .999)
            _require([stage['stage'] for stage in row['stages']] == list(STAGES))
            _require(all(math.isfinite(stage['wall_seconds']) and stage['wall_seconds'] > 0 and
                math.isfinite(stage['process_cpu_seconds']) and stage['process_cpu_seconds'] >= 0 and
                stage['gpu_elapsed_ns'] is None for stage in row['stages']))
            finals.append(dict(round=name, cosine=cosine, norm_squared=norm))
        return dict(schema=1, source=source, case=case.id, intermediate_inspections=rows,
            final_inspections=finals, all_four_chains_match_fixed_references=True,
            original_golden_changed=False, CPU_comparison_qualified=False, complete_performance_accepted=False)
    except (OSError, ValueError, TypeError, KeyError):
        raise ValueError('Image CPU independent capture audit failed') from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('measure', 'audit'))
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', default=os.environ.get('GITHUB_SHA'))
    parser.add_argument('--profile', choices=references.PROFILES)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--statistics-reference',choices=('native','sixteen-lane'))
    args = parser.parse_args(argv)
    try:
        if args.command == 'audit':
            _require(args.statistics_reference is None)
            result = audit_image_cpu_reference(args.output, args.references, source=args.source)
            _write(args.output / 'independent-image-CPU-audit.json', result)
        else:
            _require(args.profile is not None)
            _empty(args.output)
            _validate_execution_source(args.source)
            _metadata(args.references)
            torch.set_num_threads(4)
            torch.set_num_interop_threads(2)
            snapshot = args.snapshot
            if snapshot is None:
                from huggingface_hub import snapshot_download
                snapshot = Path(snapshot_download('google/embeddinggemma-2', revision=references.MODEL_REVISION,
                    allow_patterns=['config.json', 'processor_config.json', 'model.safetensors', 'tokenizer.json',
                                    'tokenizer_config.json', 'config_sentence_transformers.json']))
            measure_image_cpu_reference(snapshot, args.references, args.profile, args.output, source=args.source,statistics_reference=args.statistics_reference or 'native')
    except Exception as error:
        print(json.dumps(dict(success=False, error_class=type(error).__name__)))
        return 1
    print(json.dumps(dict(success=True, scope='Image CPU reference only', complete_performance_accepted=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
