"""Preserve immutable reference bytes in Actions artifacts without running models."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

REPO = 'ayutaz/unity-embeddinggemma-2'
ORIGINAL_ID = 11814567883
PRODUCER = '4edbe17a8105233c5aaa1514faa2862b08a4f7e0'
REVISION = '914f7f89142e33e77833254d9c9b90c3cef7303b'
REFERENCE_SHA = 'ac23f39904b8149dd5ccb6485afcc93d96047f0977b5faf1d640c8bd018c13c9'
AUDIT_SHA = '68d29c96012ef0a44114f2d3edbbff8d14e116ac697abee1a36d0b212155cfa1'
BACKUP_NAME = 'pinned-audio-original-' + PRODUCER
PROFILES = ('frames-1119-valid-1119', 'frames-1119-valid-839')
CASES = ('silence', 'impulse', 'sine', 'nyquist', 'noise', 'padded_noise')


def _date(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def _live(value, now):
    try:
        run = value['workflow_run']
        return (type(value['id']) is int and value['id'] > 0 and value['expired'] is False
                and _date(value['expires_at']) > now and type(run['id']) is int and run['id'] > 0
                and type(run['repository_id']) is int and run['repository_id'] > 0
                and run['repository_id'] == run['head_repository_id'])
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def select_artifact(backups, original=None, *, now=None):
    now = now or datetime.now(timezone.utc)
    candidates = [a for a in backups if a.get('name') == BACKUP_NAME and _live(a, now)]
    if candidates:
        return max(candidates, key=lambda a: _date(a['created_at']))
    if original is not None and _live(original, now):
        return original
    raise ValueError('No unexpired pinned original reference or preserved copy is available')


def _api(endpoint):
    return json.loads(subprocess.run(['gh', 'api', endpoint], capture_output=True,
                                     text=True, check=True).stdout)


def resolve_artifact():
    backups = _api(f'repos/{REPO}/actions/artifacts?name={BACKUP_NAME}&per_page=100')['artifacts']
    try:
        return select_artifact(backups)
    except ValueError:
        original = _api(f'repos/{REPO}/actions/artifacts/{ORIGINAL_ID}')
        if original.get('id') != ORIGINAL_ID:
            raise ValueError('Original artifact identity differs')
        return select_artifact(backups, original)


def _digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_original(directory):
    root = Path(directory).resolve()
    for name, expected in [('audio-encoder-reference.json', REFERENCE_SHA),
                           ('audio-encoder-validation.json', AUDIT_SHA)]:
        if _digest(root / name) != expected:
            raise ValueError('Pinned original reference/audit SHA256 differs')
    reference = json.loads((root / 'audio-encoder-reference.json').read_text(encoding='utf-8'))
    audit = json.loads((root / 'audio-encoder-validation.json').read_text(encoding='utf-8'))
    metadata = reference['metadata']
    if (metadata != audit['metadata'] or metadata['source_commit'] != PRODUCER
            or metadata['model_revision'] != REVISION or metadata['feature_atol'] != 1e-5
            or metadata['feature_rtol'] != 1e-4 or audit['native_reference_complete'] is not True
            or set(reference['profiles']) != set(PROFILES)
            or len(reference['cases']) != 6 or {c['id'] for c in reference['cases']} != set(CASES)):
        raise ValueError('Pinned original provenance, tolerances or complete coverage differs')
    checked = {}
    for name, expected in audit['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Audited file path escapes the reference directory')
        actual = dict(bytes=path.stat().st_size, sha256=_digest(path))
        if actual != expected:
            raise ValueError(f'Original file size/SHA256 differs: {name}')
        checked[name] = actual
    if not checked:
        raise ValueError('Original audited files are missing')
    return dict(success=True, original_artifact_id=ORIGINAL_ID, original_producer=PRODUCER,
                model_revision=REVISION, reference_sha256=REFERENCE_SHA, audit_sha256=AUDIT_SHA,
                audited_file_count=len(checked), files=checked, models_executed=False, gpu_validated=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    resolve = commands.add_parser('resolve')
    resolve.add_argument('--output', type=Path, required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('--input', type=Path, required=True)
    verify.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = resolve_artifact() if args.command == 'resolve' else verify_original(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if args.command == 'resolve' and os.environ.get('GITHUB_OUTPUT'):
        with Path(os.environ['GITHUB_OUTPUT']).open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(f"artifact-id={result['id']}\nrun-id={result['workflow_run']['id']}\n")
    print(json.dumps({k: result[k] for k in (('id', 'name', 'expires_at') if args.command == 'resolve'
                                             else ('success', 'audited_file_count', 'models_executed', 'gpu_validated'))}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
