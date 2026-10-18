"""Assemble retained text/vision and qualified audio bytes without inference."""
import importlib
import json
from pathlib import Path

import pytest

from test_media_setup import fixture, digest


def module():
    return importlib.import_module('embeddinggemma_tools.media_release_assets')


@pytest.fixture
def sources(tmp_path):
    base, payloads = fixture(); audio, incoming = fixture()
    old = audio['modules']['text']['model']; body = b'new audio opaque fixture, not an executable model'
    new = 'objects/'+digest(body)+'.sentis'
    def rewrite(v):
        if isinstance(v, dict): return {k: rewrite(x) for k, x in v.items()}
        if isinstance(v, list): return [rewrite(x) for x in v]
        return new if v == old else v
    audio = rewrite(audio); audio['files'].pop(old)
    audio['files'][new] = dict(bytes=len(body), sha256=digest(body))
    incoming.pop(old); incoming[new] = body
    audio['modules'] = {name: audio['modules'][name] for name in ('audio-1119', 'audio-839')}
    audio['source_commit'] = '2'*40
    for folder, descriptor, files in ((tmp_path/'base', base, payloads), (tmp_path/'audio', audio, incoming)):
        folder.mkdir(); (folder/'bundle.json').write_text(json.dumps(descriptor))
        for name, data in files.items():
            path = folder/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    return tmp_path, base, audio, payloads, incoming


def prepare(sources, **kwargs):
    root = sources[0]
    args = dict(base_sha256=digest((root/'base/bundle.json').read_bytes()),
                audio_sha256=digest((root/'audio/bundle.json').read_bytes()),
                producer='3'*40, precision='fp32')
    args.update(kwargs)
    return module().prepare(root/'base', root/'audio', root/'output', **args)


def test_complete_distribution_replaces_only_audio_deduplicates_and_preserves_all_object_bytes(sources):
    root, base, audio, payloads, incoming = sources
    result = prepare(sources)
    data = json.loads((root/'output/bundle.json').read_text())
    assert set(data['modules']) == set(base['modules'])
    for name in base['modules']:
        assert data['modules'][name] == (audio if name.startswith('audio-') else base)['modules'][name]
    assert data['source_commit'] == '3'*40 and data['object_layout'] == 'flat'
    assert len(data['files']) == 3 and result['copied_objects'] == 3
    for name in data['files']:
        body = (incoming if name in incoming else payloads)[name]
        target = root/'output'/name
        assert target.read_bytes() == body
        assert not target.samefile(root/('audio' if name in incoming else 'base')/name)
    assert result['success'] and not result['inference_executed'] and not result['gpu_validated']
    assert result['source_producers'] == {'base':'1'*40, 'audio':'2'*40}
    inventory = json.loads((root/'output/release-assets.json').read_text())
    assert {row['asset_name'] for row in inventory['objects']} == {Path(name).name for name in data['files']}
    assert inventory['descriptor_sha256'] == digest((root/'output/bundle.json').read_bytes())
    assert inventory['privacy_review_completed'] is False


@pytest.mark.parametrize('which', ['base', 'audio'])
def test_wrong_descriptor_pin_cannot_create_distribution(sources, which):
    with pytest.raises(ValueError): prepare(sources, **{which+'_sha256':'0'*64})
    assert not (sources[0]/'output').exists()


@pytest.mark.parametrize('which', ['base', 'audio'])
def test_same_size_corrupt_selected_source_is_rejected_before_any_output(sources, which):
    root, base, audio, *_ = sources
    data = base if which == 'base' else audio
    name = next(name for name in data['files'] if name.endswith('.sentis'))
    path = root/which/name; body = path.read_bytes(); path.write_bytes(bytes([body[0]^1])+body[1:])
    with pytest.raises(ValueError): prepare(sources)
    assert not (root/'output').exists()


@pytest.mark.parametrize('problem', ['missing_base_module', 'missing_audio_module', 'foreign_audio_module', 'revision', 'precision', 'producer'])
def test_incomplete_or_mismatched_inputs_fail_before_output(sources, problem):
    root, base, audio, *_ = sources; kwargs = {}
    if problem == 'missing_base_module': base['modules'].pop('text')
    elif problem == 'missing_audio_module': audio['modules'].pop('audio-839')
    elif problem == 'foreign_audio_module': audio['modules']['image-square'] = base['modules']['image-square']
    elif problem == 'revision': audio['model_revision'] = '0'*40
    elif problem == 'precision': kwargs['precision'] = 'automatic'
    elif problem == 'producer': kwargs['producer'] = 'short'
    (root/'base/bundle.json').write_text(json.dumps(base)); (root/'audio/bundle.json').write_text(json.dumps(audio))
    with pytest.raises(ValueError): prepare(sources, **kwargs)
    assert not (root/'output').exists()


def test_foreign_output_is_preserved(sources):
    root = sources[0]; (root/'output').mkdir(); keep = root/'output/keep.txt'; keep.write_text('keep')
    with pytest.raises(FileExistsError): prepare(sources)
    assert keep.read_text() == 'keep' and not (root/'output/bundle.json').exists()
