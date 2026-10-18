"""Release creation must review attached files just like later asset uploads."""
import pytest

from embeddinggemma_tools.privacy_hook import decide


def allowed(tmp_path, command):
    event = dict(tool_name='Bash', cwd=str(tmp_path), tool_input=dict(command=command))
    return decide(event, lambda: True)['hookSpecificOutput']['permissionDecision'] == 'allow'


@pytest.mark.parametrize('payload', ['private-email', 'private-path', 'binary', 'image', 'large'])
def test_release_creation_rejects_unreviewed_asset_bytes(tmp_path, payload):
    path = tmp_path / ('capture.png' if payload == 'image' else 'receipt.txt')
    if payload == 'private-email': raw = ('fixture'+'@'+'private-domain.dev').encode()
    elif payload == 'private-path': raw = ('C:'+'/'+'Users'+'/'+'fixture-user'+'/private').encode()
    elif payload in ('binary', 'image'): raw = b'unreviewed\x00payload'
    else: raw = b'x' * (16*1024**2 + 1)
    path.write_bytes(raw)
    assert not allowed(tmp_path, 'gh release create v1 '+path.name+' --notes public')


@pytest.mark.parametrize('name', ['missing.txt', '$asset', '-', '*.txt'])
def test_release_creation_does_not_assume_missing_or_dynamic_assets_are_safe(tmp_path, name):
    assert not allowed(tmp_path, 'gh release create v1 '+name+' --notes public')


@pytest.mark.parametrize('verb', ['create', 'upload'])
def test_release_assets_keep_labels_and_repo_options_but_check_each_file(tmp_path, verb):
    (tmp_path/'public.txt').write_text('Public receipt')
    command = f'gh release {verb} v1 "public.txt#Public receipt" --repo example/repo'
    assert allowed(tmp_path, command)
    (tmp_path/'private.txt').write_text('fixture'+'@'+'private-domain.dev')
    assert not allowed(tmp_path, command+' private.txt')


@pytest.mark.parametrize('argument', ['--unknown value', '--title', '--notes-file', '--latest=unknown'])
def test_unknown_or_incomplete_creation_flags_fail_closed(tmp_path, argument):
    assert not allowed(tmp_path, 'gh release create v1 '+argument)


def test_creation_notes_and_control_flags_are_not_mistaken_for_assets(tmp_path):
    (tmp_path/'receipt.txt').write_text('Public receipt')
    (tmp_path/'notes.md').write_text('Public notes')
    assert allowed(tmp_path, 'gh release create v1 receipt.txt --title "Public release" '
                   '--notes-file=notes.md --target main --discussion-category General '
                   '--notes-start-tag v0 --draft --prerelease --verify-tag --generate-notes '
                   '--notes-from-tag --fail-on-no-commits --latest=false')
    assert allowed(tmp_path, 'gh release create v1 -n public -t Public -d -p -R example/repo')


@pytest.mark.parametrize('verb', ['create', 'upload'])
def test_reviewed_model_assets_use_the_same_exact_byte_gate_for_both_verbs(tmp_path, monkeypatch, verb):
    from embeddinggemma_tools import model_release_privacy
    path = tmp_path/'reviewed.sentis'; path.write_bytes(b'reviewed\x00numeric')
    monkeypatch.setattr(model_release_privacy, 'reviewed_model_asset_safe',
                        lambda actual, cwd: actual == path and actual.read_bytes() == b'reviewed\x00numeric')
    assert allowed(tmp_path, f'gh release {verb} preview reviewed.sentis')
    path.write_bytes(b'changed\x00numeric')
    assert not allowed(tmp_path, f'gh release {verb} preview reviewed.sentis')


def test_model_asset_review_does_not_approve_binary_release_notes(tmp_path, monkeypatch):
    from embeddinggemma_tools import model_release_privacy
    (tmp_path/'notes.bin').write_bytes(b'numeric\x00not notes')
    monkeypatch.setattr(model_release_privacy, 'reviewed_model_asset_safe', lambda *args: True)
    assert not allowed(tmp_path, 'gh release create preview --notes-file notes.bin')


@pytest.mark.parametrize('prefix', ['gh release new', 'gh --repo example/repo release create',
                                  'gh -R example/repo release upload'])
def test_release_alias_and_inherited_repo_option_cannot_skip_actual_payload_review(tmp_path, prefix):
    path = tmp_path/'receipt.txt'; path.write_text('Public receipt')
    assert allowed(tmp_path, prefix+' v1 receipt.txt')
    path.write_text('fixture'+'@'+'private-domain.dev')
    assert not allowed(tmp_path, prefix+' v1 receipt.txt')


def test_unrelated_read_only_command_does_not_require_posix_shell_parsing(tmp_path):
    assert allowed(tmp_path, "Get-Content README.md # project's notes")
