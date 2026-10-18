"""A full acquisition keeps the pinned descriptor bytes on every host OS."""
import io
import json

import pytest

from embeddinggemma_tools.media_setup import MODULES, prepare_bundle
from test_media_setup import fixture, digest, BASE


@pytest.mark.parametrize('layout', ['nested', 'flat'])
@pytest.mark.parametrize('style', ['compact', 'lf', 'crlf', 'unicode'])
def test_full_acquisition_preserves_independent_descriptor_pin_byte_for_byte(tmp_path, layout, style):
    data, payloads = fixture(); data['object_layout'] = layout
    data['description'] = '日本語の固定モデル情報' if style == 'unicode' else 'Fixed model fixture'
    raw = (json.dumps(data, indent=None if style == 'compact' else 4,
                      ensure_ascii=style != 'unicode') + '\n').encode()
    if style == 'crlf': raw = raw.replace(b'\n', b'\r\n')
    def fetch(url):
        name = url.removeprefix(BASE.removesuffix('bundle.json'))
        key = next(key for key in payloads if key.rsplit('/', 1)[-1] == name) if layout == 'flat' else name
        return io.BytesIO(payloads[key])
    for modules in (MODULES, tuple(reversed(MODULES))):
        report = prepare_bundle(raw, tmp_path, source_url=BASE, descriptor_sha256=digest(raw),
                                modules=modules, fetch=fetch)
        assert (tmp_path/'bundle.json').read_bytes() == raw
        assert report['source_descriptor_bytes_preserved'] is True
        assert report['prepared_descriptor_sha256'] == report['descriptor_sha256'] == digest(raw)
        assert not report['gpu_validated'] and not report['inference_executed']


@pytest.mark.parametrize('subset', [False, True])
def test_filtered_descriptor_never_preserves_unprepared_source_object_declarations(tmp_path, subset):
    data, payloads = fixture(); unused = b'unreferenced object must stay unprepared'
    name = 'objects/'+digest(unused)+'.bin'
    data['files'][name] = dict(bytes=len(unused), sha256=digest(unused))
    raw = json.dumps(data).encode(); calls = []
    def fetch(url):
        key = url.removeprefix(BASE.removesuffix('bundle.json')); calls.append(key)
        return io.BytesIO(payloads[key])
    modules = ['text'] if subset else MODULES
    report = prepare_bundle(raw, tmp_path, source_url=BASE, descriptor_sha256=digest(raw),
                            modules=modules, fetch=fetch)
    output = (tmp_path/'bundle.json').read_bytes(); published = json.loads(output)
    assert name not in published['files'] and name not in calls
    assert set(published['modules']) == set(modules)
    assert report['source_descriptor_bytes_preserved'] is False
    assert report['prepared_descriptor_sha256'] == digest(output) != digest(raw)
