"""Acquire a pinned GPU media bundle; acquisition does not prove GPU execution.

Usage: uv run --locked python -m embeddinggemma_tools.media_setup --help
The descriptor hash must be supplied independently of the download server.
"""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import urljoin, urlsplit
from urllib.request import urlopen
import uuid

MODULES = ('text', 'image-square', 'image-landscape', 'image-portrait', 'audio-1119', 'audio-839')
REVISION = '914f7f89142e33e77833254d9c9b90c3cef7303b'
FILE_KEYS = frozenset(('model', 'tokenizer', 'tokenizer_config', 'backbone', 'preprocessor', 'weight_file'))
OBJECT = re.compile(r'objects/([a-f0-9]{64})\.(sentis|json|f32|bin)\Z')
CHUNK_BYTES = 1024 * 1024
MAX_DESCRIPTOR_BYTES = 8 * CHUNK_BYTES


def _url(value):
    parsed = urlsplit(value)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or not parsed.path.endswith('/bundle.json')):
        raise ValueError('Use an HTTPS bundle.json URL without embedded credentials or query parameters')
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate descriptor metadata key')
        result[key] = value
    return result


def _sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _read_descriptor(raw, expected):
    if (not isinstance(raw, bytes) or len(raw) > MAX_DESCRIPTOR_BYTES
            or re.fullmatch('[a-f0-9]{64}', expected or '') is None
            or hashlib.sha256(raw).hexdigest() != expected):
        raise ValueError('Pinned descriptor size or SHA256 differs')
    try:
        data = json.loads(raw, object_pairs_hook=_pairs)
        if (data['schema_version'] != 1 or type(data['schema_version']) is not int
                or data['backend'] != 'GPUCompute' or data['sentis_version'] != '2.6.1'
                or data['model_revision'] != REVISION
                or re.fullmatch('[a-f0-9]{40}', data['source_commit']) is None
                or not isinstance(data['files'], dict) or not isinstance(data['modules'], dict)
                or not data['modules'] or set(data['modules']) - set(MODULES)):
            raise ValueError('Pinned GPU media provenance or modules differ')
        for name, value in data['files'].items():
            match = OBJECT.fullmatch(name)
            if (not match or not isinstance(value, dict) or type(value.get('bytes')) is not int
                    or value['bytes'] <= 0 or value.get('sha256') != match[1]):
                raise ValueError('Invalid content-addressed object metadata')
        def reference(name, *, model=False):
            if (not isinstance(name, str) or name not in data['files']
                    or model and not name.endswith('.sentis')):
                raise ValueError('Undefined module file reference')
        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in FILE_KEYS:
                        reference(item)
                    elif key == 'models':
                        if not isinstance(item, list) or not item:
                            raise ValueError('Module models must be a nonempty list')
                        for name in item:
                            reference(name)
                    else:
                        walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
        for name, module in data['modules'].items():
            if not isinstance(module, dict) or module.get('kind') != name.split('-')[0]:
                raise ValueError('Module kind differs from its name')
            if module.get('backend', 'GPUCompute') != 'GPUCompute':
                raise ValueError('Every module must use GPUCompute')
            reference(module.get('tokenizer'))
            if module['kind'] == 'text':
                reference(module.get('model'), model=True)
            else:
                reference(module.get('tokenizer_config'))
                reference(module.get('backbone'), model=True)
                if module['kind'] == 'audio':
                    reference(module.get('preprocessor'), model=True)
                models, operations = module.get('models'), module.get('operations')
                if (not isinstance(models, list) or not models or not isinstance(operations, list)
                        or len(models) != len(operations) + 1):
                    raise ValueError('Every operation must connect two declared model segments')
                for model in models:
                    reference(model, model=True)
            walk(module)
        return data
    except (KeyError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError('Invalid GPU media descriptor') from error


def _atomic_json(path, value):
    pending = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.pending')
    try:
        pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _download(url):
    return urlopen(url, timeout=60)


def prepare_bundle(descriptor, output, *, source_url, descriptor_sha256, modules=MODULES,
                   fetch=None, cancelled=None):
    """Hash-audit streaming transfers, repair/reuse objects, publish metadata last.

    A failed attempt keeps any previously published descriptor, removes partial
    files, and records success=false. Complete objects remain reusable on retry.
    """
    source_url = _url(source_url)
    data = _read_descriptor(descriptor, descriptor_sha256)
    modules = tuple(modules)
    if not modules or len(set(modules)) != len(modules) or set(modules) - set(data['modules']):
        raise ValueError('Select available modules explicitly; required modules are missing')
    names = set()
    def collect(value):
        if isinstance(value, str) and value in data['files']:
            names.add(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
    for name in modules:
        collect(data['modules'][name])
    if not names:
        raise ValueError('Selected modules have no declared objects')
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    fetch, cancelled = fetch or _download, cancelled or (lambda: False)
    report = dict(success=False, model_revision=REVISION, producer=data['source_commit'],
                  descriptor_sha256=descriptor_sha256, source_url=source_url, modules=list(modules),
                  downloaded_files=0, reused_files=0, downloaded_bytes=0, reused_bytes=0,
                  inference_executed=False, gpu_validated=False)
    def check_cancelled():
        if cancelled():
            raise InterruptedError('Model preparation was cancelled; retry reuses complete audited objects')
    try:
        for name in sorted(names):
            check_cancelled()
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise ValueError('Output object escapes the preparation directory')
            expected = data['files'][name]
            if target.is_file() and target.stat().st_size == expected['bytes'] and _sha(target) == expected['sha256']:
                report['reused_files'] += 1
                report['reused_bytes'] += expected['bytes']
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            pending = target.with_name('.' + target.name + '.' + uuid.uuid4().hex + '.pending')
            try:
                digest, count = hashlib.sha256(), 0
                with closing(fetch(urljoin(source_url, name))) as stream, pending.open('xb') as destination:
                    while True:
                        check_cancelled()
                        chunk = stream.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        count += len(chunk)
                        if count > expected['bytes']:
                            raise ValueError('Transferred object exceeds declared size: ' + name)
                        digest.update(chunk)
                        destination.write(chunk)
                if count != expected['bytes'] or digest.hexdigest() != expected['sha256']:
                    raise ValueError('Transferred object size or SHA256 differs: ' + name)
                os.replace(pending, target)
                report['downloaded_files'] += 1
                report['downloaded_bytes'] += count
            finally:
                pending.unlink(missing_ok=True)
        check_cancelled()
        published = dict(data, files={name: data['files'][name] for name in sorted(names)},
                         modules={name: data['modules'][name] for name in modules})
        _atomic_json(root / 'bundle.json', published)
        report['success'] = True
        return report
    except BaseException as error:
        report['error_type'] = type(error).__name__
        raise
    finally:
        _atomic_json(root / 'preparation.json', report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptor-url', required=True)
    parser.add_argument('--descriptor-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modules', nargs='+', choices=MODULES, default=MODULES)
    args = parser.parse_args(argv)
    try:
        with closing(_download(_url(args.descriptor_url))) as stream:
            raw = stream.read(MAX_DESCRIPTOR_BYTES + 1)
        report = prepare_bundle(raw, args.output, source_url=args.descriptor_url,
                                descriptor_sha256=args.descriptor_sha256, modules=args.modules)
        print(json.dumps(report))
        return 0
    except (OSError, ValueError) as error:
        parser.exit(1, f'Model preparation failed ({type(error).__name__}): {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
