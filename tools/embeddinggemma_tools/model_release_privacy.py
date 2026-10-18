"""Check exact reviewed model-release bytes without allowing models into Git.

The trusted policy and review receipt come from HEAD, not untracked edits.
Every upload rechecks its inventory binding, complete file hash and textual
metadata. Numerical buffers are allowed only through exact reviewed producer
provenance; they are not interpreted as arbitrary text or generally approved.
"""
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

from .privacy import HOME, SECRET, normalize, private_email_in_text

POLICY = '.privacy-reviewed-model-releases.json'
OBJECT = re.compile(r'objects/([a-f0-9]{64})\.(sentis|bin|f32|json)\Z')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('Duplicate release metadata key')
        result[key] = value
    return result


def _json(raw):
    return json.loads(raw, object_pairs_hook=_pairs)


def _versioned_json(cwd, name):
    result = subprocess.run(['git', '-C', str(cwd), 'show', 'HEAD:'+name],
                            capture_output=True, check=True)
    if len(result.stdout) > 8*1024**2: raise ValueError('Oversized review metadata')
    return _json(result.stdout)


def _canonical(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _raw(path, maximum=8*1024**2):
    if path.is_symlink() or path.stat().st_size > maximum: raise ValueError('Invalid release metadata file')
    return path.read_bytes()


def _safe_text(value):
    text = normalize(value)
    return not HOME.search(text) and not SECRET.search(text) and not private_email_in_text(text)


def _safe_description(stream, file_size):
    stream.seek(0); prefix = stream.read(4)
    if len(prefix) != 4: return False
    size = struct.unpack('<i', prefix)[0]
    if not 0 < size <= 64*1024**2 or size+4 > file_size: return False
    description = prefix+stream.read(size)
    if len(description) != size+4: return False
    found = False
    for offset in range(0, len(description)-4, 4):
        length = struct.unpack_from('<I', description, offset)[0]; end = offset+4+length
        if not 0 < length or end >= len(description) or description[end] != 0: continue
        try: text = description[offset+4:end].decode('utf-8')
        except UnicodeDecodeError: continue
        found = True
        if any(not _safe_text(value) for value in text.split('\x00')): return False
    return found


def reviewed_model_asset_safe(path, cwd):
    """Return false for every missing, changed, unreviewed or private payload."""
    try:
        path = Path(path).absolute()
        if (path.parent.name != 'objects' or path.is_symlink() or not path.is_file()
                or path.parent.is_symlink() or path.parent.parent.is_symlink()): return False
        relative = 'objects/'+path.name; match = OBJECT.fullmatch(relative)
        if match is None: return False
        root = path.parent.parent
        raw = _raw(root/'bundle.json'); descriptor_sha = hashlib.sha256(raw).hexdigest(); descriptor = _json(raw)
        policy = _versioned_json(cwd, POLICY)
        if policy.get('schema') != 1: return False
        review = policy['distributions'][descriptor_sha]
        proof_path = review['review_path']
        if re.fullmatch(r'docs/results/[a-z0-9-]+\.json', proof_path) is None: return False
        proof = _versioned_json(cwd, proof_path)
        inventory_raw = _raw(root/'release-assets.json'); inventory_sha = hashlib.sha256(inventory_raw).hexdigest()
        inventory = _json(inventory_raw)
        if (_canonical(proof) != review['review_canonical_sha256']
                or inventory_sha != review['inventory_sha256'] or proof['inventory_sha256'] != inventory_sha
                or proof['descriptor_sha256'] != descriptor_sha
                or proof['schema'] != 1 or proof['review'] != 'model-description+source-provenance'
                or proof['success'] is not True or proof['contains_private_metadata'] is not False
                or proof['images_in_distribution'] is not False
                or proof['all_source_and_destination_hashes_verified'] is not True
                or proof['all_selected_module_bindings_unchanged'] is not True): return False
        if (descriptor['schema_version'] != 1 or descriptor['backend'] != 'GPUCompute'
                or descriptor['object_layout'] != inventory['object_layout'] or descriptor['object_layout'] != 'flat'
                or descriptor['model_revision'] != '914f7f89142e33e77833254d9c9b90c3cef7303b'
                or re.fullmatch('[a-f0-9]{40}', review['producer']) is None
                or descriptor['source_commit'] != review['producer'] or inventory['producer'] != review['producer']
                or inventory['descriptor_sha256'] != descriptor_sha
                or len(descriptor['files']) != proof['object_count']
                or len(inventory['objects']) != proof['object_count']): return False
        entries = {}
        for row in inventory['objects']:
            name = row['path']; parsed = OBJECT.fullmatch(name)
            if (parsed is None or name in entries or row['asset_name'] != Path(name).name
                    or row['sha256'] != parsed[1] or type(row['bytes']) is not int or row['bytes'] <= 0): return False
            entries[name] = dict(bytes=row['bytes'], sha256=row['sha256'])
        if entries != descriptor['files'] or relative not in entries: return False
        expected = entries[relative]
        if path.stat().st_size != expected['bytes']: return False
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected['sha256']: return False
            if match[2] == 'sentis': return _safe_description(stream, expected['bytes'])
            if match[2] in ('bin', 'f32'): return expected['bytes'] % 4 == 0
            if expected['bytes'] > 64*1024**2: return False
            stream.seek(0); text = stream.read().decode('utf-8-sig')
            _json(text)
            return _safe_text(text)
    except Exception:
        return False
