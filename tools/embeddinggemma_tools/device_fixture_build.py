"""Stage pinned local inputs for device builds without claiming runtime acceptance."""
import hashlib
import json
from pathlib import Path
import re

DIRECTORY = 'Assets/StreamingAssets/EmbeddingGemmaDeviceValidation'
MAX_FILE = 64 * 2**20
MAX_TOTAL = 512 * 2**20


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_manifest_key')
        result[key] = value
    return result


def _inventory(raw, pin):
    if (not isinstance(raw, bytes) or not 0 < len(raw) <= 256 * 1024
            or not isinstance(pin, str) or not re.fullmatch('[a-f0-9]{64}', pin)
            or hashlib.sha256(raw).hexdigest() != pin):
        raise ValueError('bounded_pinned_manifest_required')
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('invalid_fixture_manifest') from error
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'files'}
            or type(value['schema_version']) is not int or value['schema_version'] != 1
            or not isinstance(value['files'], list) or not 0 < len(value['files']) <= 512):
        raise ValueError('versioned_fixture_inventory_required')
    names, total = set(), 0
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{prefix}{i}' for prefix in ('COM', 'LPT') for i in range(1, 10)}
    for row in value['files']:
        if (not isinstance(row, dict) or set(row) != {'path', 'bytes', 'sha256'}
                or not isinstance(row['path'], str) or not isinstance(row['sha256'], str)):
            raise ValueError('exact_fixture_entry_required')
        name = row['path']
        if (not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._/-]*', name)
                or any(part in {'', '.', '..'} or part.endswith('.')
                       or part.split('.')[0].upper() in reserved for part in name.split('/'))
                or Path(name).suffix.lower() not in {'.json', '.bin', '.png'}
                or name.casefold() in names or type(row['bytes']) is not int
                or not 0 < row['bytes'] <= MAX_FILE or not re.fullmatch('[a-f0-9]{64}', row['sha256'])):
            raise ValueError('safe_unique_bounded_fixture_required')
        names.add(name.casefold())
        total += row['bytes']
    if total > MAX_TOTAL or any(any(name.startswith(other + '/') for other in names) for name in names):
        raise ValueError('fixture_inventory_size_or_path_collision')
    return value['files']


def _no_links(path):
    """Reject symlinks and Windows junctions in either source or destination ancestry."""
    path = Path(path).absolute()
    if any(item.is_symlink() or item.is_junction() for item in (path, *path.parents)):
        raise ValueError('linked_fixture_path_not_supported')
    return path


def _verify(path, row, output=None):
    path = _no_links(path)
    if not path.is_file() or path.stat().st_size != row['bytes']:
        raise ValueError('fixture_source_size_mismatch')
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while chunk := stream.read(min(1024 * 1024, row['bytes'] + 1 - size)):
            size += len(chunk)
            if size > row['bytes']:
                raise ValueError('fixture_changed_during_copy')
            digest.update(chunk)
            if output is not None:
                output.write(chunk)
    if size != row['bytes'] or digest.hexdigest() != row['sha256']:
        raise ValueError('fixture_source_hash_mismatch')


def stage_fixtures(source, project, manifest, manifest_sha256):
    """Verify every source before creating a new owned StreamingAssets directory.

    The original manifest is preserved byte for byte. The device's unchanged
    fixture loaders still supply the independent original numerical/hash gate.
    """
    rows = _inventory(manifest, manifest_sha256)
    source = _no_links(source)
    destination = _no_links(Path(project) / DIRECTORY)
    if not source.is_dir() or destination.exists():
        raise ValueError('new_fixture_destination_and_local_source_required')
    for row in rows:
        _verify(source / row['path'], row)
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / 'fixture-manifest.json').open('xb') as stream:
        stream.write(manifest)
    for row in rows:
        target = destination / 'fixtures' / row['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_name(target.name + '.part')
        created = False
        try:
            with pending.open('xb') as stream:
                created = True
                _verify(source / row['path'], row, stream)
            # Move only this invocation's new verified file, never replace a prior output.
            if target.exists():
                raise ValueError('fixture_output_collision')
            pending.rename(target)
        finally:
            if created:
                pending.unlink(missing_ok=True)
    return dict(schema=1, scope='host_streaming_fixture_build_staging', success=True,
                manifest_sha256=manifest_sha256, verified_files=len(rows),
                verified_bytes=sum(row['bytes'] for row in rows), files=rows,
                original_fixture_validated=False, gpu_verified=False,
                build_performed=False, model_execution_performed=False)
