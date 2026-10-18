"""Portable acquisition contracts. Tiny opaque files never claim model/GPU validity."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import pytest

from embeddinggemma_tools.media_setup import MODULES, prepare_bundle

REVISION = '914f7f89142e33e77833254d9c9b90c3cef7303b'
SOURCE = '1' * 40
BASE = 'https://models.example.test/fixed/bundle.json'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture():
    payloads = {}
    def object(name, extension):
        raw = name.encode()
        key = f'objects/{digest(raw)}.{extension}'
        payloads[key] = raw
        return key
    model = object('opaque model fixture, not an executable model', 'sentis')
    tokenizer = object('{}', 'json')
    module = dict(kind='text', model=model, tokenizer=tokenizer)
    modules = {'text': module}
    for name in MODULES:
        if name == 'text':
            continue
        modules[name] = dict(kind='image' if name.startswith('image-') else 'audio',
                             models=[model], backbone=model, tokenizer=tokenizer,
                             tokenizer_config=tokenizer, operations=[])
        if name.startswith('image-'):
            width, height = {'image-square': (768, 768), 'image-landscape': (1008, 576),
                             'image-portrait': (576, 1008)}[name]
            modules[name].update(resized_width=width, resized_height=height)
        else:
            modules[name].update(valid_audio_frames=int(name.split('-')[1]), preprocessor=model)
    data = dict(schema_version=1, backend='GPUCompute', sentis_version='2.6.1',
                model_revision=REVISION, source_commit=SOURCE, prepared_only=True,
                inference_executed=False, modules=modules,
                files={name: dict(bytes=len(raw), sha256=digest(raw)) for name, raw in payloads.items()})
    return data, payloads


def prepare(tmp_path, data, payloads, **kwargs):
    calls = []
    def fetch(url):
        name = url.removeprefix(BASE.removesuffix('bundle.json'))
        calls.append(name)
        return io.BytesIO(payloads[name])
    raw = json.dumps(data).encode()
    result = prepare_bundle(raw, tmp_path, source_url=BASE, descriptor_sha256=digest(raw), fetch=fetch, **kwargs)
    return result, calls


def test_fresh_directory_acquires_all_six_modules_and_publishes_audited_descriptor(tmp_path):
    data, payloads = fixture()
    receipt, calls = prepare(tmp_path, data, payloads)
    assert receipt['success'] is True
    assert set(receipt['modules']) == set(MODULES)
    assert receipt['downloaded_files'] == len(payloads) == len(calls)
    assert receipt['model_revision'] == REVISION and receipt['producer'] == SOURCE
    assert receipt['inference_executed'] is False and receipt['gpu_validated'] is False
    assert json.loads((tmp_path/'bundle.json').read_text()) == data
    for name, raw in payloads.items():
        assert (tmp_path/name).read_bytes() == raw


def test_second_preparation_rechecks_same_length_hash_and_downloads_only_corruption(tmp_path):
    data, payloads = fixture()
    prepare(tmp_path, data, payloads)
    receipt, calls = prepare(tmp_path, data, payloads)
    assert calls == [] and receipt['reused_files'] == len(payloads)
    name = next(iter(payloads))
    (tmp_path/name).write_bytes(b'!' * len(payloads[name]))
    receipt, calls = prepare(tmp_path, data, payloads)
    assert calls == [name] and receipt['downloaded_files'] == 1
    assert (tmp_path/name).read_bytes() == payloads[name]


def test_selected_module_never_advertises_unprepared_modules_or_fetches_unused_objects(tmp_path):
    data, payloads = fixture()
    unused = b'not selected'
    name = f'objects/{digest(unused)}.bin'
    payloads[name] = unused
    data['files'][name] = dict(bytes=len(unused), sha256=digest(unused))
    module = data['modules']['audio-839']
    module['models'] *= 2
    module['operations'] = [dict(kind='sequential_scale', weight_file=name, first_scale=1.0,
                                 inputs=['input'], output='output')]
    receipt, calls = prepare(tmp_path, data, payloads, modules=['text'])
    published = json.loads((tmp_path/'bundle.json').read_text())
    assert set(published['modules']) == {'text'} and name not in published['files']
    assert name not in calls and receipt['modules'] == ['text']


@pytest.mark.parametrize('change', ['revision', 'backend', 'sentis', 'producer', 'unknown_module', 'missing_module', 'undefined_file', 'traversal', 'hash_in_name', 'negative_bytes', 'boolean_bytes'])
def test_invalid_source_metadata_fails_before_any_download(tmp_path, change):
    data, payloads = fixture()
    name = next(iter(payloads))
    if change == 'revision': data['model_revision'] = '0' * 40
    if change == 'backend': data['backend'] = 'CPU'
    if change == 'sentis': data['sentis_version'] = '9.9.9'
    if change == 'producer': data['source_commit'] = 'main'
    if change == 'unknown_module': data['modules']['audio-any'] = data['modules']['audio-839']
    if change == 'missing_module': del data['modules']['audio-839']
    if change == 'undefined_file': data['modules']['text']['model'] = 'objects/' + '0'*64 + '.sentis'
    if change == 'traversal': data['files']['../outside.sentis'] = data['files'].pop(name)
    if change == 'hash_in_name': data['files'][name]['sha256'] = '0' * 64
    if change == 'negative_bytes': data['files'][name]['bytes'] = -1
    if change == 'boolean_bytes': data['files'][name]['bytes'] = True
    with pytest.raises(ValueError):
        prepare(tmp_path, data, payloads)
    assert not (tmp_path/'bundle.json').exists()
    assert not list(tmp_path.rglob('*.pending'))


def test_descriptor_hash_mismatch_rejects_before_fetch(tmp_path):
    raw = json.dumps(fixture()[0]).encode()
    with pytest.raises(ValueError, match='descriptor'):
        prepare_bundle(raw, tmp_path, source_url=BASE, descriptor_sha256='0'*64,
                       fetch=lambda _: pytest.fail('must not fetch'))


@pytest.mark.parametrize('body', [b'wrong bytes', b'', b'larger than declared size' * 100])
def test_bad_transfer_leaves_no_complete_bundle_and_retry_recovers(tmp_path, body):
    data, payloads = fixture()
    broken = deepcopy(payloads)
    broken[next(iter(payloads))] = body
    with pytest.raises(ValueError):
        prepare(tmp_path, data, broken)
    assert not (tmp_path/'bundle.json').exists()
    assert not list(tmp_path.rglob('*.pending'))
    assert json.loads((tmp_path/'preparation.json').read_text())['success'] is False
    receipt, _ = prepare(tmp_path, data, payloads)
    assert receipt['success'] is True


def test_failed_update_preserves_previously_published_descriptor(tmp_path):
    data, payloads = fixture()
    prepare(tmp_path, data, payloads)
    before = (tmp_path/'bundle.json').read_bytes()
    name = next(iter(payloads))
    (tmp_path/name).write_bytes(b'broken')
    broken = dict(payloads, **{name: b'wrong'})
    with pytest.raises(ValueError): prepare(tmp_path, data, broken)
    assert (tmp_path/'bundle.json').read_bytes() == before


def test_interrupted_transfer_cleans_pending_and_retries_without_false_success(tmp_path):
    data, payloads = fixture()
    checks = iter([False, True])
    with pytest.raises(InterruptedError):
        prepare(tmp_path, data, payloads, cancelled=lambda: next(checks, True))
    assert not (tmp_path/'bundle.json').exists() and not list(tmp_path.rglob('*.pending'))
    assert json.loads((tmp_path/'preparation.json').read_text())['success'] is False
    assert prepare(tmp_path, data, payloads)[0]['success'] is True


@pytest.mark.parametrize('url', ['http://models.example.test/bundle.json', 'file:///secret/bundle.json', 'https://user:secret@example.test/bundle.json', 'https://example.test/bundle.json?token=secret'])
def test_non_https_or_credential_bearing_source_is_rejected(tmp_path, url):
    raw = json.dumps(fixture()[0]).encode()
    with pytest.raises(ValueError):
        prepare_bundle(raw, tmp_path, source_url=url, descriptor_sha256=digest(raw), fetch=lambda _: pytest.fail('must not fetch'))


def test_duplicate_metadata_keys_are_rejected(tmp_path):
    raw = json.dumps(fixture()[0]).replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1').encode()
    with pytest.raises(ValueError):
        prepare_bundle(raw, tmp_path, source_url=BASE, descriptor_sha256=digest(raw), fetch=lambda _: pytest.fail('must not fetch'))


@pytest.mark.parametrize(('module', 'field'), [
    ('text', 'model'), ('text', 'tokenizer'),
    ('image-square', 'tokenizer'), ('image-square', 'tokenizer_config'), ('image-square', 'backbone'),
    ('audio-1119', 'tokenizer'), ('audio-1119', 'tokenizer_config'),
    ('audio-1119', 'backbone'), ('audio-1119', 'preprocessor'),
])
def test_required_recipe_file_missing_is_rejected_before_fetch(tmp_path, module, field):
    data, _ = fixture()
    del data['modules'][module][field]
    raw = json.dumps(data).encode()
    with pytest.raises(ValueError):
        prepare_bundle(raw, tmp_path, source_url=BASE, descriptor_sha256=digest(raw),
                       fetch=lambda _: pytest.fail('invalid recipe must fail before fetching any object'))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('change', ['non_model_backbone', 'non_model_preprocessor', 'non_model_segment',
                                   'missing_operations', 'non_list_operations', 'segment_count', 'module_cpu'])
def test_incomplete_media_recipe_is_rejected_before_fetch(tmp_path, change):
    data, _ = fixture()
    module = data['modules']['audio-1119']
    tokenizer = module['tokenizer']
    if change == 'non_model_backbone': module['backbone'] = tokenizer
    if change == 'non_model_preprocessor': module['preprocessor'] = tokenizer
    if change == 'non_model_segment': module['models'] = [tokenizer]
    if change == 'missing_operations': del module['operations']
    if change == 'non_list_operations': module['operations'] = {}
    if change == 'segment_count': module['models'] *= 2
    if change == 'module_cpu': module['backend'] = 'CPU'
    raw = json.dumps(data).encode()
    with pytest.raises(ValueError):
        prepare_bundle(raw, tmp_path, source_url=BASE, descriptor_sha256=digest(raw),
                       fetch=lambda _: pytest.fail('invalid recipe must fail before fetching any object'))
    assert not list(tmp_path.iterdir())


def test_retry_removes_only_owned_stale_partial_files(tmp_path):
    data, payloads = fixture()
    name = next(iter(payloads)).removeprefix('objects/')
    objects = tmp_path/'objects'; objects.mkdir()
    stale = [objects/('.'+name+'.'+'a'*32+'.pending'),
             tmp_path/('.bundle.json.'+'b'*32+'.pending'),
             tmp_path/('.preparation.json.'+'c'*32+'.pending')]
    unrelated = [objects/'notes.pending', tmp_path/'.bundle.json.not-a-uuid.pending']
    for path in stale + unrelated: path.write_bytes(b'partial owned copy or user note')
    receipt, _ = prepare(tmp_path, data, payloads)
    assert all(not path.exists() for path in stale)
    assert all(path.read_bytes() == b'partial owned copy or user note' for path in unrelated)
    assert receipt['stale_pending_files_removed'] == 3


def test_second_preparer_cannot_fetch_or_replace_owner_receipt(tmp_path):
    data, payloads = fixture(); raw = json.dumps(data).encode()
    previous = b'{"previous_receipt":"retain until owner completes"}'
    (tmp_path/'preparation.json').write_bytes(previous)
    attempted = False
    def fetch(url):
        nonlocal attempted
        if not attempted:
            attempted = True
            with pytest.raises(BlockingIOError):
                prepare_bundle(raw, tmp_path, source_url=BASE, descriptor_sha256=digest(raw),
                    fetch=lambda _: pytest.fail('The competing preparer must not fetch an object'))
            assert (tmp_path/'preparation.json').read_bytes() == previous
        return io.BytesIO(payloads[url.removeprefix(BASE.removesuffix('bundle.json'))])
    receipt = prepare_bundle(raw, tmp_path, source_url=BASE, descriptor_sha256=digest(raw), fetch=fetch)
    assert attempted and receipt['success'] and receipt['downloaded_files'] == len(payloads)


def test_cli_descriptor_interrupt_has_cancelled_exit_without_traceback(monkeypatch, tmp_path, capsys):
    from embeddinggemma_tools import media_setup
    def interrupted(_):
        raise KeyboardInterrupt
    monkeypatch.setattr(media_setup, '_download', interrupted)
    try:
        with pytest.raises(SystemExit) as exit:
            media_setup.main(['--descriptor-url', BASE, '--descriptor-sha256', '0'*64,
                              '--output', str(tmp_path/'output')])
    except KeyboardInterrupt:
        pytest.fail('CLI must turn the interrupt into a cancelled exit130, without exposing KeyboardInterrupt.')
    assert exit.value.code == 130
    assert 'cancelled' in capsys.readouterr().err.lower()
    assert not (tmp_path/'output').exists()
