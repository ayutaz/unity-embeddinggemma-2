"""Public provider authors differ from locally configured source identities."""
import pytest

from embeddinggemma_tools import privacy


def headers(author='yousan', email='41669061+ayutaz@users.noreply.github.com',
            committer='GitHub', committer_email='noreply@github.com', signed=True):
    result = (f'author {author} <{email}> 1 +0000\n'
              f'committer {committer} <{committer_email}> 1 +0000')
    if signed:
        result += '\ngpgsig -----BEGIN PGP SIGNATURE-----\n fixture\n -----END PGP SIGNATURE-----'
    return result


def test_reviewed_public_name_is_accepted_for_provider_author_only():
    assert privacy.scan_metadata_identities(headers(), kind='commit') == []
    # Ordinary commits and the local Git identity still require the public handle.
    assert privacy.scan_identity('yousan', '41669061+ayutaz@users.noreply.github.com')
    assert privacy.scan_metadata_identities(headers(signed=False), kind='commit')
    assert privacy.scan_metadata_identities(headers(), kind='tag')


@pytest.mark.parametrize('changes', [
    {'author': 'YOUSAN'}, {'author': 'yousan '},
    {'author': 'Synthetic Private Name'},
    {'email': '41669061+other-account@users.noreply.github.com'},
    {'email': '999+ayutaz@users.noreply.github.com'},
    {'email': 'ayutaz@users.noreply.github.com'},
    {'email': 'fixture' + '@' + 'gmail.com'},
    {'committer': 'fixture', 'committer_email': '123+fixture@users.noreply.github.com'},
    {'committer': 'GitHub', 'committer_email': 'fixture' + '@' + 'gmail.com'},
])
def test_provider_context_cannot_excuse_other_names_or_addresses(changes):
    findings = privacy.scan_metadata_identities(headers(**changes), kind='commit')
    assert findings
    assert all(row['rule'] == 'private-identity' for row in findings)
    assert 'Synthetic' not in str(findings)
    assert 'gmail' not in str(findings)


def test_public_source_author_and_provider_committer_still_pass():
    assert privacy.scan_metadata_identities(headers(author='ayutaz', signed=False), kind='commit') == []


def test_duplicate_author_does_not_acquire_provider_exception():
    with pytest.raises(ValueError, match='Malformed'):
        privacy.scan_metadata_identities(headers() + '\nauthor extra <extra@example.test> 1 +0000', kind='commit')


def test_alias_cannot_be_used_as_tagger_or_committer():
    assert privacy.scan_metadata_identities(
        'tagger yousan <41669061+ayutaz@users.noreply.github.com> 1 +0000', kind='tag')
    assert privacy.scan_metadata_identities(headers(committer='yousan',
        committer_email='41669061+ayutaz@users.noreply.github.com'), kind='commit')
