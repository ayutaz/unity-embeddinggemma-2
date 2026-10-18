"""Preserve the pinned unqualified audio candidate without regenerating weights.

Source bytes remain private inputs. Only distinct metadata-clean distribution
copies and a path-free receipt are selected for the guarded preservation upload.
This does not adopt a generation or replace the original acceptance gate.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from . import audio_generation_candidate as candidate
from . import exported_program_distribution as distribution

CASES = ('silence', 'impulse', 'sine', 'nyquist', 'noise', 'padded_noise')
CONTROLS = dict(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MKL_CBWR='COMPATIBLE', MKL_DYNAMIC='FALSE')


def fetch_inputs(output, original, *, source_archive=None):
    from .artifact_zip import verified_local_artifact_reader, extract_members
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Candidate download output must be empty')
    contract = candidate.candidate_contract()
    reader, source_proof = verified_local_artifact_reader(source_archive, contract['artifact_sha256'])
    with reader:
        reference, _ = candidate.original_reference(original)
        members = ['audio-encoder-reference.json', 'audio-encoder-validation.json', *contract['graphs']]
        members += [row['input_features'] for row in reference['cases'] if row['id'] in CASES]
        extract_members(reader, output, ['audio-runtime-reference/'+name for name in members],
                        max_entry_bytes=2*1024*1024*1024)
    for name in CASES:
        candidate._candidate_input_files(output/'audio-runtime-reference', original, case_id=name)
    return source_proof


def native_outputs(paths, cases):
    """Load one profile at a time; source and distribution never coexist in RAM."""
    result = {}
    for name, path in paths.items():
        program = torch.export.load(path).module()
        if sorted(node.args[1][0] for node in candidate.checked_normalizations(program)) != [32, 128]:
            raise ValueError('Candidate original normalization widths differ')
        try:
            for row in cases:
                if row['graph'] != name: continue
                values = torch.from_numpy(np.fromfile(row['raw'], dtype='<f4').reshape(row['shape']).copy())
                with torch.inference_mode(): output = program(values)
                actual = output.detach().cpu().contiguous().numpy().copy()
                if list(actual.shape) != row['output_shape'] or actual.dtype != np.dtype('float32') or not np.isfinite(actual).all():
                    raise ValueError('Candidate native output shape, type or finite values differ')
                result[row['case_id']] = actual
                del values, output
        finally:
            del program
            gc.collect()
    if set(result) != set(CASES): raise ValueError('All six native outputs are required')
    return result


def prepare(current, original, output, *, source):
    if any(os.environ.get(key) != value for key, value in CONTROLS.items()):
        raise ValueError('Native comparison execution controls differ')
    if len(source) != 40 or any(c not in '0123456789abcdef' for c in source):
        raise ValueError('An immutable execution source is required')
    output = Path(output)
    if output.exists() and any(output.iterdir()): raise ValueError('Distribution output must be empty')
    contract = candidate.candidate_contract()
    paths, cases, profiles = {}, [], set()
    for name in CASES:
        graph, raw, spec, row, provenance = candidate._candidate_input_files(current, original, case_id=name)
        paths[spec['native_graph']] = graph; profiles.add(row['profile'])
        cases.append(dict(case_id=name, graph=spec['native_graph'], raw=raw, shape=spec['feature_shape'],
                          output_shape=[spec['feature_count'],spec['feature_width']]))
    if len(paths) != 2 or len(profiles) != 2 or set(paths) != set(contract['graphs']):
        raise ValueError('Both pinned graph profiles are required')
    graphs, cleaned = {}, {}
    for name, path in paths.items():
        cleaned[name] = output/name
        graphs[name] = distribution.sanitize(path, cleaned[name],
                           expected_source_sha256=contract['graphs'][name]['sha256'])
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        before = native_outputs(paths, cases)
        after = native_outputs(cleaned, cases)
    finally:
        torch.set_num_threads(old_threads)
    rows = []
    for name in CASES:
        if not np.array_equal(before[name].view(np.uint32), after[name].view(np.uint32)):
            raise ValueError('Source and distribution native output bits differ')
        rows.append(dict(case_id=name, shape=list(before[name].shape), bit_identical=True,
            source_sha256=hashlib.sha256(before[name].astype('<f4').tobytes()).hexdigest(),
            distribution_sha256=hashlib.sha256(after[name].astype('<f4').tobytes()).hexdigest()))
        candidate._candidate_input_files(current, original, case_id=name)
    report = dict(schema=1, scope='pinned_candidate_metadata_clean_distribution_and_native_bit_comparison_only',
        source=source, model_generation=contract['generation'], candidate_contract_sha256=candidate.CONTRACT_SHA,
        model_revision=contract['model_revision'], graph_producer=contract['producer_source'],
        original_reference_sha256=candidate.REFERENCE_SHA, original_audit_sha256=candidate.AUDIT_SHA,
        graphs=graphs, case_count=len(rows), profile_count=len(profiles), native_outputs=rows,
        native_outputs_bit_identical=True, native_execution_controls=dict(CONTROLS, torch_threads=4),
        original_numerical_goldens_unchanged=True, original_graph_bytes=False,
        generation_qualified=False, gpu_validated=False)
    with (output/'distribution.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2)+'\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--source-archive', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.fetch: fetch_inputs(args.candidate, args.original, source_archive=args.source_archive)
        current = args.candidate/'audio-runtime-reference' if args.fetch else args.candidate
        result = prepare(current, args.original, args.output, source=args.source)
        print(json.dumps(dict(case_count=result['case_count'], profile_count=result['profile_count'],
                              native_outputs_bit_identical=True, generation_qualified=False, gpu_validated=False)))
        return 0
    except Exception:
        print('Candidate distribution preparation failed; private inputs and failure details were not published.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
