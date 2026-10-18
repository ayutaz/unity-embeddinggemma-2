"""Preserve exact original numerical JSON and six raw inputs, without models.

This copies hash-pinned bytes. No reference or feature generation is performed.
"""
import argparse
import json
from pathlib import Path
import re
import shutil

from . import audio_generation_candidate as candidate

CASES = ('silence', 'impulse', 'sine', 'nyquist', 'noise', 'padded_noise')


def _cases(original):
    reference, audit = candidate.original_reference(original)
    rows = reference['cases']
    if len(rows) != 6 or {row['id'] for row in rows} != set(CASES) or len({row['profile'] for row in rows}) != 2:
        raise ValueError('All six original cases and both profiles are required')
    return reference, audit


def fetch_inputs(current, original):
    from .artifact_zip import extract_members
    current = Path(current)
    if current.exists() and any(current.iterdir()):
        raise ValueError('Source input download directory must be empty')
    candidate.fetch_original(original)
    reference, _ = _cases(original)
    reader, _ = candidate.preserved_original_reader()
    with reader:
        extract_members(reader, current/'audio-runtime-reference', [row['input_features'] for row in reference['cases']],
                        max_entry_bytes=2*1024*1024)


def preserve(current, original, output, *, source):
    if re.fullmatch('[0-9a-f]{40}', source or '') is None:
        raise ValueError('An immutable execution source is required')
    current, original, output = Path(current), Path(original), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preservation output directory must be empty')
    reference, audit = _cases(original)
    paths, specs = {}, {}
    for name, expected in [('audio-encoder-reference.json', candidate.REFERENCE_SHA),
                           ('audio-encoder-validation.json', candidate.AUDIT_SHA)]:
        spec = dict(bytes=(original/name).stat().st_size, sha256=expected)
        paths[name] = candidate.checked_file(original, name, spec); specs[name] = spec
    for row in reference['cases']:
        name = row['input_features']
        if name in paths: raise ValueError('Original input file identity is ambiguous')
        spec = audit['files'][name]
        paths[name] = candidate.checked_file(current, name, spec); specs[name] = spec
    output.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        destination = output/name; destination.parent.mkdir(parents=True, exist_ok=True)
        with path.open('rb') as left, destination.open('xb') as right:
            shutil.copyfileobj(left, right, 1024*1024)
        candidate.checked_file(output, name, specs[name])
    result = dict(schema=1, scope='exact_original_numerical_reference_and_six_raw_inputs_only',
        source=source, reference_source=candidate.PRODUCER, model_revision=candidate.REVISION,
        original_reference_sha256=candidate.REFERENCE_SHA, original_audit_sha256=candidate.AUDIT_SHA,
        case_count=6, profile_count=2, case_ids=list(CASES), files=specs,
        original_numerical_goldens_unchanged=True, raw_inputs_regenerated=False,
        models_included=False, model_executed=False, gpu_validated=False, generation_qualified=False)
    with (output/'input-preservation.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.fetch: fetch_inputs(args.current, args.original)
        current = args.current/'audio-runtime-reference' if args.fetch else args.current
        result = preserve(current, args.original, args.output, source=args.source)
        print(json.dumps(dict(case_count=result['case_count'], profile_count=result['profile_count'],
                              files_verified=len(result['files']), model_executed=False, gpu_validated=False)))
        return 0
    except Exception:
        print('Original input preservation failed; no private paths or input contents were printed.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
