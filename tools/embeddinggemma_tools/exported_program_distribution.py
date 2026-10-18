"""Create separate PT2 distribution copies without optional node stack traces.

No Torch deserialization, reference regeneration, or model qualification occurs.
All non-model members must retain their exact uncompressed bytes. Opaque tensor
payloads are hash-verified, not certified by the text metadata privacy audit.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import zipfile

from .privacy import HOME, SECRET, normalize, private_email_in_text

METADATA_LIMIT = 32*1024*1024


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _source(path, expected):
    if re.fullmatch('[0-9a-f]{64}', expected or '') is None or _digest(path) != expected:
        raise ValueError('Pinned source hash differs')


def _privacy(value):
    text = normalize(value)
    if HOME.search(text) or SECRET.search(text) or private_email_in_text(text):
        raise ValueError('Distribution metadata privacy validation failed')


def _strings(value):
    if isinstance(value, str):
        _privacy(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            _privacy(key); _strings(item)
    elif isinstance(value, list):
        for item in value: _strings(item)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('Duplicate JSON metadata key')
        result[key] = value
    return result


def _json(data):
    return json.loads(data.decode('utf-8'), object_pairs_hook=_unique)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def _without_traces(document):
    result = copy.deepcopy(document)
    try:
        nodes = result['graph_module']['graph']['nodes']
        if not isinstance(nodes, list): raise ValueError()
        count = 0
        for node in nodes:
            if not isinstance(node, dict): raise ValueError()
            metadata = node.get('metadata', {})
            if not isinstance(metadata, dict): raise ValueError()
            if 'stack_trace' in metadata:
                if not isinstance(metadata['stack_trace'], str): raise ValueError()
                del metadata['stack_trace']; count += 1
    except (KeyError, TypeError, ValueError):
        raise ValueError('Unsupported exported graph metadata structure') from None
    _strings(result)
    return result, count


def _members(archive):
    infos = archive.infolist(); names = [info.filename for info in infos]
    if len(names) != len(set(names)): raise ValueError('Ambiguous archive member names')
    _privacy(archive.comment.decode('utf-8', errors='replace'))
    models = []
    for info in infos:
        name = info.filename
        if (not name or '\\' in name or ':' in name or PurePosixPath(name).is_absolute()
                or any(part in ('', '.', '..') for part in name.split('/')) or info.is_dir()
                or info.flag_bits & 1 or info.external_attr >> 16 & 0o170000 == 0o120000):
            raise ValueError('Unsupported archive member path or type')
        for value in (name, info.comment.decode('utf-8', errors='replace'),
                      info.extra.decode('utf-8', errors='replace')):
            _privacy(value)
        if name.endswith('/models/model.json'): models.append(name)
    if len(models) != 1: raise ValueError('Exactly one exported model is required')
    return names, models[0]


def _text_members(archive, model):
    prefix = model.removesuffix('models/model.json')
    for info in archive.infolist():
        name = info.filename
        if name == model: continue
        tensor = any(name.startswith(prefix+'data/'+kind+'/') for kind in ('weights', 'constants', 'sample_inputs'))
        if tensor and not name.endswith('.json'): continue
        if info.file_size > METADATA_LIMIT: raise ValueError('Text metadata size limit exceeded')
        data = archive.read(info)
        if name.endswith('.json'): _strings(_json(data))
        else: _privacy(data.decode('utf-8'))


def _model(archive, name):
    if archive.getinfo(name).file_size > METADATA_LIMIT:
        raise ValueError('Model metadata size limit exceeded')
    return _json(archive.read(name))


def verify(source, distribution, *, expected_source_sha256):
    """Independently re-read every member, rejecting any computational change."""
    _source(source, expected_source_sha256)
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(distribution) as cleaned:
        names, model = _members(original); new_names, new_model = _members(cleaned)
        if names != new_names or model != new_model:
            raise ValueError('Distribution archive members differ')
        expected, removed = _without_traces(_model(original, model))
        actual = _model(cleaned, model); _strings(actual)
        if _canonical(actual) != _canonical(expected):
            raise ValueError('Distribution computational structure differs')
        _text_members(original, model); _text_members(cleaned, model)
        unchanged = 0
        for name in names:
            if name == model: continue
            with original.open(name) as left, cleaned.open(name) as right:
                left_sha = hashlib.file_digest(left, 'sha256').hexdigest()
                right_sha = hashlib.file_digest(right, 'sha256').hexdigest()
            if (original.getinfo(name).file_size != cleaned.getinfo(name).file_size
                    or left_sha != right_sha):
                raise ValueError('Distribution member bytes differ')
            unchanged += 1
    return dict(schema=1, scope='optional_node_debug_metadata_removal_only',
        source_sha256=expected_source_sha256, distribution_sha256=_digest(distribution),
        semantic_model_sha256=hashlib.sha256(_canonical(expected)).hexdigest(),
        removed_stack_trace_fields=removed, unchanged_members_verified=unchanged,
        computational_structure_unchanged=True, text_metadata_privacy_checked=True,
        opaque_tensor_contents_privacy_certified=False, original_graph_bytes=False,
        model_executed=False, gpu_validated=False, generation_qualified=False)


def sanitize(source, distribution, *, expected_source_sha256):
    source = Path(source); distribution = Path(distribution)
    if source.resolve() == distribution.resolve() or distribution.exists() or distribution.is_symlink():
        raise ValueError('Refusing to overwrite source or existing distribution')
    _source(source, expected_source_sha256)
    distribution.parent.mkdir(parents=True, exist_ok=True)
    descriptor, pending_name = tempfile.mkstemp(prefix='.distribution-', suffix='.pt2', dir=distribution.parent)
    os.close(descriptor); pending = Path(pending_name)
    try:
        with zipfile.ZipFile(source) as original, zipfile.ZipFile(pending, 'w', allowZip64=True) as cleaned:
            _, model = _members(original)
            expected, _ = _without_traces(_model(original, model))
            _text_members(original, model)
            cleaned.comment = original.comment
            for info in original.infolist():
                copied = copy.copy(info)
                if info.filename == model:
                    cleaned.writestr(copied, _canonical(expected))
                else:
                    with original.open(info) as left, cleaned.open(copied, 'w', force_zip64=True) as right:
                        shutil.copyfileobj(left, right, length=1024*1024)
        report = verify(source, pending, expected_source_sha256=expected_source_sha256)
        os.link(pending, distribution)  # Atomic create; never replaces an existing destination.
        return report
    finally:
        pending.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--distribution', required=True, type=Path)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args(argv)
    try:
        report = sanitize(args.source, args.distribution, expected_source_sha256=args.source_sha256)
        print(json.dumps(report))
        return 0
    except Exception:
        print('Distribution preparation failed; no private path or metadata was printed.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
