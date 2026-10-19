import hashlib
import importlib
import io
import json
from types import SimpleNamespace

import pytest


def implementation():
    return importlib.import_module('embeddinggemma_tools.image_reference_distribution')


class Response(io.BytesIO):
    def __init__(self, data, url='https://release-assets.githubusercontent.com/fixture', length=None):
        super().__init__(data)
        self.url = url
        self.headers = {'Content-Length': str(len(data) if length is None else length)}

    def geturl(self):
        return self.url


@pytest.fixture
def transport(tmp_path, monkeypatch):
    module = implementation()
    names = ('vision-reference.json', 'vision-validation.json', 'image-embedding-reference.json',
             'LICENSE.txt', 'NOTICE.txt', 'MODIFICATIONS.txt')
    files = {name: json.dumps(dict(metadata=dict(model_revision='a' * 40), name=name)).encode()
             if name.endswith('.json') else b'Synthetic fixture attribution' for name in names}
    rows = {name: dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) for name, raw in files.items()}
    review = dict(assets={name: rows[name] for name in names[:3]})
    review_path = tmp_path / 'review.json'
    review_path.write_text(json.dumps(review), encoding='utf-8')
    review_sha = hashlib.sha256(review_path.read_bytes()).hexdigest()
    config = dict(schema=1, repository='ayutaz/unity-embeddinggemma-2',
        tag='image-reference-preview-6644eea', review_sha256=review_sha, assets=rows)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    monkeypatch.setattr(module, 'CONFIG_PATH', config_path)
    monkeypatch.setattr(module, 'CONFIG_SHA256', hashlib.sha256(config_path.read_bytes()).hexdigest())
    monkeypatch.setattr(module, 'REVIEW_CONFIG_PATH', review_path)
    monkeypatch.setattr(module, 'REVIEW_SHA256', review_sha)
    requests, validations = [], []

    def open_asset(url):
        requests.append(url)
        assert url.startswith('https://github.com/ayutaz/unity-embeddinggemma-2/releases/download/image-reference-preview-6644eea/')
        return Response(files[url.rsplit('/', 1)[1]])

    def validate(*paths):
        validations.append(tuple(path.name for path in paths))
        return SimpleNamespace(cases=tuple(range(6)), document_encoder_source_sha256='7' * 64)

    from embeddinggemma_tools import native_regression_distribution
    monkeypatch.setattr(native_regression_distribution, '_open', open_asset)
    monkeypatch.setattr(module, 'load_image_reference_cohort', validate)
    return module, files, requests, validations, config, config_path


def test_fetches_originals_and_attribution_then_strictly_validates_cohort(transport, tmp_path):
    module, files, requests, validations, _, _ = transport
    output = tmp_path / 'output'
    result = module.fetch_image_reference_data(output)
    assert len(requests) == 6 and len(result['downloads']) == 6
    assert validations == [('vision-reference.json', 'vision-validation.json', 'image-embedding-reference.json')]
    assert all((output / name).read_bytes() == raw for name, raw in files.items())
    assert result['original_golden_changed'] is False
    assert result['model_executed'] is False and result['complete_performance_accepted'] is False
    assert json.loads((output / 'image-reference-download.json').read_text()) == result


@pytest.mark.parametrize('problem', ['corrupt', 'short', 'long', 'redirect', 'private_text', 'missing'])
def test_bad_transport_fails_without_success_receipt_or_overwriting_evidence(transport, tmp_path, monkeypatch, problem):
    module, files, _, _, config, config_path = transport
    from embeddinggemma_tools import native_regression_distribution as native
    first = sorted(files)[0]
    if problem == 'private_text':
        raw = ('C:' + '/' + 'Users' + '/' + 'fixture-user' + '/private').encode()
        files[first] = raw
        config['assets'][first] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        config_path.write_text(json.dumps(config), encoding='utf-8')
        monkeypatch.setattr(module, 'CONFIG_SHA256', hashlib.sha256(config_path.read_bytes()).hexdigest())
    else:
        def bad_open(url):
            raw = files[url.rsplit('/', 1)[1]]
            if problem == 'missing': raise OSError('unavailable local path')
            if problem == 'redirect': return Response(raw, url='http://example.test/untrusted')
            if problem == 'corrupt': raw = b'X' + raw[1:]
            if problem == 'short': raw = raw[:-1]
            if problem == 'long': raw += b'X'
            return Response(raw, length=config['assets'][url.rsplit('/', 1)[1]]['bytes'])
        monkeypatch.setattr(native, '_open', bad_open)
    output = tmp_path / 'output'
    with pytest.raises(ValueError) as error:
        module.fetch_image_reference_data(output)
    assert str(tmp_path) not in str(error.value)
    assert not (output / 'image-reference-download.json').exists()
    assert not list(output.glob('*.part'))


def test_refuses_existing_evidence_and_unpinned_configuration(transport, tmp_path):
    module, _, requests, _, _, config = transport
    output = tmp_path / 'output'
    output.mkdir()
    marker = output / 'existing.json'
    marker.write_text('preserve')
    with pytest.raises(ValueError): module.fetch_image_reference_data(output)
    assert marker.read_text() == 'preserve' and not requests
    config.write_bytes(config.read_bytes() + b' ')
    with pytest.raises(ValueError): module.fetch_image_reference_data(tmp_path / 'fresh')
    assert not requests
