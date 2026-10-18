"""Recover original artifact bytes without qualifying new numerical/GPU results."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


AUDIT_SHA256 = '68d29c96012ef0a44114f2d3edbbff8d14e116ac697abee1a36d0b212155cfa1'
ORIGINAL_PRODUCER = '4edbe17a8105233c5aaa1514faa2862b08a4f7e0'
EXECUTED_SOURCE = 'd2ad81dfdf42d057fb4b7f30dd59d557f8a6dd75'


def pinned_files(manifest=None):
    path = Path(manifest) if manifest else Path(__file__).parents[1] / 'references/original-audio-validation.json'
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != AUDIT_SHA256:
        raise ValueError('Original audit bytes do not match the immutable contract')
    files = json.loads(data)['files']
    files['audio-encoder-validation.json'] = {'bytes': len(data), 'sha256': AUDIT_SHA256}
    return files


def verify(directory, *, expected_files=None, graph_paths=None):
    root = Path(directory).resolve()
    expected_files = pinned_files() if expected_files is None else expected_files
    graph_paths = [p for p in expected_files if p.endswith('.pt2')] if graph_paths is None else graph_paths
    if not expected_files or not graph_paths or not set(graph_paths).issubset(expected_files):
        raise ValueError('A nonempty graph and archive contract is required')
    results = {}
    for name, expected in expected_files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
            raise ValueError('Original manifest path must remain relative')
        path = root / name
        if not path.resolve().is_relative_to(root) or any(p.is_symlink() for p in (path, *path.parents) if p != root and p.is_relative_to(root)):
            raise ValueError('Original artifact path escapes its directory or uses a symlink')
        record = {'match': False}
        try:
            record['bytes'] = path.stat().st_size
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            record['sha256'] = digest.hexdigest()
            record['match'] = record['bytes'] == expected['bytes'] and record['sha256'] == expected['sha256']
        except OSError as exc:
            # Exception messages can contain local paths; publish only the class.
            record['error_type'] = type(exc).__name__
        results[name] = record
    graphs_match = all(results[name]['match'] for name in graph_paths)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    unexpected_count = len(actual - set(expected_files))
    complete_match = not unexpected_count and all(record['match'] for record in results.values())
    return {
        'scope': 'full_original_archive' if complete_match else 'original_graph_bytes_only' if graphs_match else 'not_recovered',
        'graphs_byte_identical': graphs_match,
        'complete_archive_byte_identical': complete_match,
        'unexpected_file_count': unexpected_count,
        'gpu_validated': False,
        'numerical_validation_performed': False,
        'original_producer': ORIGINAL_PRODUCER,
        'executed_source': EXECUTED_SOURCE,
        'files': results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--collection-outcome', choices=['success', 'failure', 'cancelled', 'skipped'], required=True)
    args = parser.parse_args()
    try:
        result = verify(args.input, expected_files=pinned_files(args.manifest))
    except (OSError, ValueError, KeyError) as exc:
        result = {'scope': 'not_recovered', 'graphs_byte_identical': False,
                  'complete_archive_byte_identical': False, 'gpu_validated': False,
                  'error_type': type(exc).__name__}
    result['collection_outcome'] = args.collection_outcome
    result['controller_commit'] = os.environ.get('RECOVERY_CONTROLLER_SHA', '')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    if output := os.environ.get('GITHUB_OUTPUT'):
        with open(output, 'a', encoding='utf-8') as stream:
            stream.write(f"graphs_match={str(result['graphs_byte_identical']).lower()}\n")
            stream.write(f"complete_match={str(result['complete_archive_byte_identical']).lower()}\n")
    return 0 if result['graphs_byte_identical'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
