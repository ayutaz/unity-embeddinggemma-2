"""Release asset URLs and local content-addressed paths are different layouts."""
import io
import json
from pathlib import Path

import pytest

from embeddinggemma_tools.media_setup import prepare_bundle
from test_media_setup import fixture, digest
from media_setup_https_harness import LocalBundleServer, create_certificate
from test_media_setup_https import cli

URL = 'https://github.com/ayutaz/unity-embeddinggemma-2/releases/download/models-preview-fixture/bundle.json'


def test_flat_release_assets_keep_content_addressed_local_paths_and_hash_checks(tmp_path):
    data, payloads = fixture(); data['object_layout'] = 'flat'
    raw = json.dumps(data).encode(); calls = []
    def fetch(url):
        calls.append(url)
        name = url.removeprefix(URL.removesuffix('bundle.json'))
        return io.BytesIO(payloads['objects/'+name])
    result = prepare_bundle(raw, tmp_path/'output', source_url=URL, descriptor_sha256=digest(raw), fetch=fetch)
    assert result['success'] and result['downloaded_files'] == len(payloads)
    assert set(calls) == {URL.removesuffix('bundle.json')+Path(name).name for name in payloads}
    for name, payload in payloads.items(): assert (tmp_path/'output'/name).read_bytes() == payload
    assert json.loads((tmp_path/'output/bundle.json').read_text()) == data
    assert not result['inference_executed'] and not result['gpu_validated']


@pytest.mark.parametrize('layout', [None, '', True, 0, 'arbitrary', '../', {'url':'https://other.example.test/'}])
def test_unknown_or_malformed_layout_cannot_fetch_or_create_output(tmp_path, layout):
    data, _ = fixture(); data['object_layout'] = layout; raw = json.dumps(data).encode(); calls = []
    with pytest.raises(ValueError):
        prepare_bundle(raw, tmp_path/'output', source_url=URL, descriptor_sha256=digest(raw), fetch=lambda url: calls.append(url))
    assert calls == [] and not (tmp_path/'output').exists()


def test_layout_change_requires_a_new_independent_descriptor_pin(tmp_path):
    data, _ = fixture(); old_pin = digest(json.dumps(data).encode()); data['object_layout'] = 'flat'
    with pytest.raises(ValueError):
        prepare_bundle(json.dumps(data).encode(), tmp_path/'output', source_url=URL, descriptor_sha256=old_pin)
    assert not (tmp_path/'output').exists()


def test_explicit_nested_layout_preserves_existing_https_object_urls(tmp_path):
    data, payloads = fixture(); data['object_layout'] = 'nested'; raw = json.dumps(data).encode(); calls = []
    def fetch(url):
        name = url.removeprefix(URL.removesuffix('bundle.json')); calls.append(name)
        return io.BytesIO(payloads[name])
    result = prepare_bundle(raw, tmp_path/'output', source_url=URL, descriptor_sha256=digest(raw), fetch=fetch)
    assert result['success'] and set(calls) == set(payloads)


@pytest.fixture
def flat_https(tmp_path):
    root = tmp_path/'flat'; root.mkdir()
    data, payloads = fixture(); data['object_layout'] = 'flat'; raw = json.dumps(data).encode()
    (root/'bundle.json').write_bytes(raw)
    for name, body in payloads.items(): (root/Path(name).name).write_bytes(body)
    certificate, private_key = create_certificate(tmp_path/'tls')
    with LocalBundleServer(root, certificate, private_key, flat_assets=True) as server:
        yield server, raw, data, payloads


def test_actual_cli_acquires_flat_https_assets_and_reuses_all_audited_objects(flat_https, tmp_path):
    server, raw, data, payloads = flat_https; output = tmp_path/'output'
    result = cli(server, output, digest(raw))
    assert result.returncode == 0, result.stderr
    assert set(server.object_requests) == {Path(name).name for name in payloads}
    assert json.loads((output/'bundle.json').read_text()) == data
    for name, body in payloads.items(): assert (output/name).read_bytes() == body
    server.reset(); result = cli(server, output, digest(raw))
    assert result.returncode == 0 and server.object_requests == []
    assert json.loads(result.stdout)['reused_files'] == len(payloads)


def test_flat_https_404_keeps_published_descriptor_then_repairs_same_size_corruption(flat_https, tmp_path):
    server, raw, _, payloads = flat_https; output = tmp_path/'output'
    assert cli(server, output, digest(raw)).returncode == 0
    previous = (output/'bundle.json').read_bytes()
    name = next(iter(payloads)); target = output/name; body = target.read_bytes()
    target.write_bytes(bytes([body[0]^1])+body[1:]); server.reset(missing_object=Path(name).name)
    failed = cli(server, output, digest(raw))
    assert failed.returncode == 1 and (output/'bundle.json').read_bytes() == previous
    assert not json.loads((output/'preparation.json').read_text())['success']
    assert not list(output.rglob('*.pending'))
    server.reset(); result = cli(server, output, digest(raw))
    assert result.returncode == 0 and target.read_bytes() == body
    assert json.loads(result.stdout)['downloaded_files'] == 1
