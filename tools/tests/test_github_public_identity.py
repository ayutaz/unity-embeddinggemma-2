import io
import json

from embeddinggemma_tools import privacy
from embeddinggemma_tools.privacy import github_profile_findings


def test_canonical_or_empty_public_display_name_is_allowed():
    for name in ('fixture', '', None):
        assert github_profile_findings(fetch_profile=lambda:{'login':'fixture','name':name})==[]


def test_other_display_name_is_rejected_without_printing_it():
    findings=github_profile_findings(fetch_profile=lambda:{'login':'fixture','name':'Synthetic Private Display'})
    assert findings==[{'path':'<github-profile>','line':1,'rule':'noncanonical-github-display-name'}]
    assert 'Synthetic' not in str(findings)


def test_user_selected_public_display_name_is_bound_to_its_account():
    assert github_profile_findings(fetch_profile=lambda:{'login':'ayutaz','name':'yousan'})==[]
    for login,name in [('other-account','yousan'),('ayutaz','Synthetic Private Display'),
                       ('ayutaz','yousan '),('ayutaz','YOUSAN')]:
        findings=github_profile_findings(fetch_profile=lambda:{'login':login,'name':name})
        assert findings==[{'path':'<github-profile>','line':1,'rule':'noncanonical-github-display-name'}]
        assert name not in str(findings)


def test_public_profile_alias_does_not_relax_source_commit_checks():
    assert privacy.scan_identity('yousan','41669061+ayutaz@users.noreply.github.com')


def test_profile_lookup_errors_and_malformed_responses_fail_closed():
    def unavailable():
        raise RuntimeError('private details must stay private')
    for fetch in (unavailable,lambda:[],lambda:{'login':'','name':None},lambda:{'login':'fixture','name':[]}):
        findings=github_profile_findings(fetch_profile=fetch)
        assert findings and 'private details' not in str(findings)


def test_only_github_remotes_require_profile_lookup():
    calls=[]
    def fetch():
        calls.append(True)
        return {'login':'fixture','name':'Synthetic Private Display'}
    assert github_profile_findings('local-sink.git',fetch_profile=fetch)==[]
    assert github_profile_findings('https://example.org/repo.git',fetch_profile=fetch)==[]
    assert calls==[]
    for remote in ('https://github.com/fixture/repo.git','git@github.com:fixture/repo.git','ssh://git@github.com/fixture/repo.git'):
        assert github_profile_findings(remote,fetch_profile=fetch)
    assert len(calls)==3


def test_pre_push_checks_provider_identity_even_with_clean_source_history(monkeypatch,tmp_path,capsys):
    monkeypatch.setattr(privacy,'scan_objects',lambda *args:[])
    monkeypatch.setattr(privacy,'reviews',lambda *args:{})
    called=[]
    def profile(remote):
        called.append(remote)
        return [{'path':'<github-profile>','line':1,'rule':'noncanonical-github-display-name'}]
    monkeypatch.setattr(privacy,'github_profile_findings',profile)
    monkeypatch.setattr(privacy.sys,'stdin',io.StringIO('refs/heads/topic '+'a'*40+' refs/heads/topic '+'b'*40+'\n'))
    assert privacy.main(['pre-push','--root',str(tmp_path),'--remote-url','https://github.com/fixture/repo.git'])==1
    assert called==['https://github.com/fixture/repo.git']
    assert json.loads(capsys.readouterr().out)['findings'][0]['rule']=='noncanonical-github-display-name'


def test_branch_deletion_does_not_create_provider_identity_and_needs_no_lookup(monkeypatch,tmp_path):
    def unexpected(*args):
        raise AssertionError('Deleting a ref must not require a profile lookup')
    monkeypatch.setattr(privacy,'github_profile_findings',unexpected)
    monkeypatch.setattr(privacy.sys,'stdin',io.StringIO('(delete) '+'0'*40+' refs/heads/topic '+'b'*40+'\n'))
    assert privacy.main(['pre-push','--root',str(tmp_path),'--remote-url','https://github.com/fixture/repo.git'])==0
