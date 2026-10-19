import hashlib
import importlib
import json
from pathlib import Path
import subprocess

import pytest

from embeddinggemma_tools.privacy_hook import publication_files_safe


def implementation():
    return importlib.import_module('embeddinggemma_tools.image_reference_release_privacy')


@pytest.fixture
def reviewed(tmp_path, monkeypatch):
    module = implementation()
    root = tmp_path / 'repository'
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    metadata = dict(source_commit='a' * 40, model_revision='b' * 40)
    raw = json.dumps(dict(metadata=metadata, padding='x' * (16 * 1024**2))).encode()
    asset = root / 'vision-reference.json'
    asset.write_bytes(raw)
    config = dict(schema=1, scope='three-original-image-reference-JSON-only',
        model_revision=metadata['model_revision'],
        assets={asset.name: dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
            producer_source=metadata['source_commit'])})
    policy = root / module.REVIEW_PATH
    policy.parent.mkdir(parents=True)

    def save(*, commit=True):
        policy.write_text(json.dumps(config), encoding='utf-8')
        monkeypatch.setattr(module, 'REVIEW_SHA256', hashlib.sha256(policy.read_bytes()).hexdigest())
        if commit:
            subprocess.run(['git', '-C', str(root), 'add', module.REVIEW_PATH], check=True)
            subprocess.run(['git', '-C', str(root), '-c', 'user.name=fixture', '-c',
                'user.email=123+fixture@users.noreply.github.com', 'commit', '-q',
                '--allow-empty', '-m', 'review fixture policy'], check=True)
    save()
    return module, root, asset, raw, config, save


def test_committed_exact_review_allows_whole_large_JSON_through_actual_publication_hook(reviewed):
    module, root, asset, _, _, _ = reviewed
    assert module.reviewed_image_reference_asset_safe(asset, root)
    command = 'gh release create fixture-preview vision-reference.json --prerelease'
    assert publication_files_safe(command, root)


@pytest.mark.parametrize('problem', ['changed_bytes', 'renamed', 'uncommitted_policy',
    'modified_review', 'producer', 'revision', 'binary', 'private_text', 'oversized', 'missing'])
def test_large_asset_does_not_bypass_ordinary_privacy_or_committed_authority(reviewed, problem):
    module, root, asset, raw, config, save = reviewed
    if problem == 'changed_bytes': asset.write_bytes(raw + b' ')
    elif problem == 'renamed':
        other = root / 'other.json'
        other.write_bytes(raw)
        asset = other
    elif problem == 'uncommitted_policy':
        config['assets'][asset.name]['sha256'] = 'f' * 64
        save(commit=False)
    elif problem == 'modified_review': monkeypatch_review = 'f' * 64; module.REVIEW_SHA256 = monkeypatch_review
    elif problem in ('producer', 'revision', 'binary', 'private_text'):
        value = json.loads(raw)
        if problem == 'producer': value['metadata']['source_commit'] = 'c' * 40
        elif problem == 'revision': value['metadata']['model_revision'] = 'c' * 40
        elif problem == 'binary': value = None
        else: value['padding'] = 'C:' + '/' + 'Users' + '/' + 'fixture-user' + '/input'
        raw = b'\x00' if value is None else json.dumps(value).encode()
        asset.write_bytes(raw)
        config['assets'][asset.name].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        save()
    elif problem == 'oversized': config['assets'][asset.name]['bytes'] = 33 * 1024**2; save()
    elif problem == 'missing': asset.unlink()
    assert not module.reviewed_image_reference_asset_safe(asset, root)
    if asset.exists() and asset.stat().st_size > 16 * 1024**2:
        assert not publication_files_safe(f'gh release create fixture-preview {asset.name}', root)


def test_generic_large_json_and_large_notes_still_fail(reviewed):
    _, root, asset, raw, _, _ = reviewed
    other = root / 'unreviewed.json'
    other.write_bytes(raw)
    assert not publication_files_safe('gh release create fixture-preview unreviewed.json', root)
    assert not publication_files_safe('gh release create fixture-preview --notes-file vision-reference.json', root)


def test_production_review_pins_are_the_existing_cohort_pins():
    module = implementation()
    from embeddinggemma_tools import image_reference_cohort as cohort
    root = Path(__file__).resolve().parents[2]
    raw = (root / module.REVIEW_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == module.REVIEW_SHA256
    rows = json.loads(raw)['assets']
    assert {name: entry['sha256'] for name, entry in rows.items()} == {
        'vision-reference.json': cohort.ENCODER_REFERENCE_SHA256,
        'vision-validation.json': cohort.ENCODER_AUDIT_SHA256,
        'image-embedding-reference.json': cohort.DOCUMENT_REFERENCE_SHA256,
    }
