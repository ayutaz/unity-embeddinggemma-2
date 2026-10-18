import hashlib
import json
from embeddinggemma_tools.privacy import scan_bytes, scan_identity

def windows_home():
    return 'C:' + '\\' + 'Users' + '\\' + 'fixture-user' + '\\project'

def test_detects_real_and_json_escaped_home_paths_without_echoing_values():
    for text in (windows_home(), json.dumps({'source': windows_home()}),
                 '/home/' + 'fixture-user/project', '/Users/' + 'fixture-user/project'):
        findings = scan_bytes(text.encode(), 'receipt.json')
        assert any(f['rule'] == 'home-path' for f in findings)
        assert 'fixture-user' not in json.dumps(findings)

def test_encoded_home_path_is_also_private():
    from urllib.parse import quote
    assert scan_bytes(quote(windows_home(), safe='').encode(), 'README.md')
    assert scan_bytes(windows_home().replace('\\', '&#92;').encode(), 'README.md')

def test_personal_email_and_token_are_rejected_without_echoing_them():
    email = 'fixture-person' + '@' + 'gmail.com'
    token = 'gh' + 'p_' + 'X' * 36
    for content in (email, token):
        findings = scan_bytes(content.encode(), 'notes.md')
        assert findings
        assert content not in json.dumps(findings)

def test_public_noreply_and_relative_or_environment_paths_are_allowed():
    text = 'models/bundle.json ${USER_HOME}/models $env:USERPROFILE 123+fixture@users.noreply.github.com'
    assert scan_bytes(text.encode(), 'README.md') == []
    assert scan_identity('fixture', '123+fixture@users.noreply.github.com') == []
    assert scan_identity('GitHub', 'noreply@github.com') == []

def test_commit_metadata_must_use_public_identity():
    assert scan_identity('fixture', 'private' + '@' + 'gmail.com')
    assert scan_identity('Private Real Name', '123+fixture@users.noreply.github.com')

def test_unreviewed_raster_and_binary_cannot_silently_pass():
    image = b'\x89PNG\r\n\x1a\nnot-ocr-reviewed'
    assert scan_bytes(image, 'docs/screen.png')
    assert scan_bytes(b'opaque\x00data', 'capture.bin')

def test_image_review_is_bound_to_exact_file_hash_and_required_method():
    image = b'\x89PNG\r\n\x1a\nreviewed-image'
    path = 'docs/screen.png'
    review = {path: {'sha256': hashlib.sha256(image).hexdigest(), 'review': 'visual+ocr', 'contains_personal_data': False}}
    assert scan_bytes(image, path, review) == []
    assert scan_bytes(image + b'changed', path, review)
    review[path]['review'] = 'unchecked'
    assert scan_bytes(image, path, review)

def test_examples_and_public_license_contacts_are_not_private_email():
    assert scan_bytes(b'test@example.com https://github.com/fixture/project', 'LICENSE') == []

def test_diagnostics_never_include_the_source_line():
    findings = scan_bytes(('source=' + windows_home()).encode(), 'docs/result.json')
    assert findings and all(set(f) == {'path', 'line', 'rule'} for f in findings)

def test_reviewed_numeric_fixture_is_preserved_but_arbitrary_binary_is_not():
    data = b'\x00\x00\x80\x3f'
    path = 'tools/tests/fixtures/native_tanh/known.bin'
    review = {path: {'sha256': hashlib.sha256(data).hexdigest(), 'review': 'numeric-source-verified'}}
    assert scan_bytes(data, path, review) == []
    assert scan_bytes(data+b'changed', path, review)
    assert scan_bytes(data, 'capture.bin', {'capture.bin': review[path]})
