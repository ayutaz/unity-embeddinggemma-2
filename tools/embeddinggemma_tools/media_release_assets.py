"""Compose reviewed text/vision and audio objects for flat release hosting.

This copies hash-verified bytes. It neither converts weights nor qualifies
models, reviews binary privacy, uploads assets or publishes a release.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil

from .media_setup import MODULES, _read_descriptor


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _objects(module, files):
    found = set()
    def visit(value):
        if isinstance(value, str) and value in files:
            found.add(value)
        elif isinstance(value, dict):
            for item in value.values(): visit(item)
        elif isinstance(value, list):
            for item in value: visit(item)
    visit(module)
    return found


def _verified(root, name, expected):
    path = root/name
    if (not path.resolve().is_relative_to(root) or not path.is_file()
            or any(parent.is_symlink() for parent in (path, *path.parents) if parent.is_relative_to(root))
            or path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']):
        raise ValueError('A selected source object differs from its descriptor')
    return path


def prepare(base, audio, output, *, base_sha256, audio_sha256, producer, precision):
    if re.fullmatch('[a-f0-9]{40}', producer or '') is None or precision not in ('fp32', 'float16-weights'):
        raise ValueError('An immutable producer and explicit precision are required')
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError('A distribution destination must be new')
    roots = {'base':Path(base).resolve(), 'audio':Path(audio).resolve()}
    if any(output.resolve().is_relative_to(root) or root.is_relative_to(output.resolve()) for root in roots.values()):
        raise ValueError('Distribution output cannot overlap its sources')
    pins = {'base':base_sha256, 'audio':audio_sha256}
    data = {name:_read_descriptor((root/'bundle.json').read_bytes(), pins[name]) for name, root in roots.items()}
    if set(data['base']['modules']) != set(MODULES) or set(data['audio']['modules']) != {'audio-1119', 'audio-839'}:
        raise ValueError('All six base modules and exactly both replacement audio profiles are required')
    modules, files, sources = {}, {}, {}
    for name in MODULES:
        kind = 'audio' if name.startswith('audio-') else 'base'
        source = data[kind]; modules[name] = copy.deepcopy(source['modules'][name])
        for relative in _objects(modules[name], source['files']):
            expected = source['files'][relative]
            path = _verified(roots[kind], relative, expected)
            if relative in files and files[relative] != expected:
                raise ValueError('Shared object descriptors disagree')
            files[relative] = copy.deepcopy(expected); sources[relative] = path
    # Leave room for bundle/inventory/license/notice within the hosting limit.
    if len(files)+4 > 1000 or any(item['bytes'] >= 2*1024**3 for item in files.values()):
        raise ValueError('Release asset count or individual file size exceeds the hosting limits')
    producers = {name:source['source_commit'] for name, source in data.items()}
    descriptor = dict(data['base'], source_commit=producer, object_layout='flat',
                      modules=modules, files={name:files[name] for name in sorted(files)},
                      distribution=dict(precision=precision, source_descriptor_sha256=pins,
                                        source_producers=producers, composition_only=True))
    report = dict(schema=1, success=False, scope='Exact selected-object distribution composition only',
                  producer=producer, precision=precision, source_producers=producers,
                  source_descriptor_sha256=pins, copied_objects=0, copied_bytes=0,
                  inference_executed=False, gpu_validated=False, privacy_review_completed=False)
    output.mkdir(parents=True)
    try:
        for name in sorted(files):
            target = output/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sources[name], target)
            if target.stat().st_size != files[name]['bytes'] or sha(target) != files[name]['sha256']:
                raise ValueError('Copied distribution object differs')
            report['copied_objects'] += 1; report['copied_bytes'] += target.stat().st_size
        # Metadata becomes available only after every copy has been verified.
        descriptor_path = output/'bundle.json'
        descriptor_path.write_text(json.dumps(descriptor, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        inventory = dict(schema=1, producer=producer, descriptor_sha256=sha(descriptor_path),
                         precision=precision, object_layout='flat', privacy_review_completed=False,
                         objects=[dict(path=name, asset_name=Path(name).name, **files[name]) for name in sorted(files)])
        (output/'release-assets.json').write_text(json.dumps(inventory, indent=2)+'\n', encoding='utf-8')
        report.update(success=True, descriptor_sha256=inventory['descriptor_sha256'])
        return report
    except BaseException as error:
        (output/'bundle.json').unlink(missing_ok=True)
        report['error_type'] = type(error).__name__
        raise
    finally:
        (output/'distribution-preparation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'audio', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    for name in ('base-sha256', 'audio-sha256', 'producer'): parser.add_argument('--'+name, required=True)
    parser.add_argument('--precision', choices=('fp32', 'float16-weights'), required=True)
    args = parser.parse_args(argv)
    try:
        report = prepare(args.base, args.audio, args.output, base_sha256=args.base_sha256,
                         audio_sha256=args.audio_sha256, producer=args.producer, precision=args.precision)
        print(json.dumps(report)); return 0
    except Exception as error:
        print(json.dumps(dict(success=False, error_type=type(error).__name__,
                              message='Distribution preparation failed; private input paths were not printed.')))
        return 1


if __name__ == '__main__': raise SystemExit(main())
