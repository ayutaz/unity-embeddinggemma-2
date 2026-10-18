from embeddinggemma_tools.privacy_hook import decide

def event(command, tool='Bash'):
    return {'hook_event_name':'PreToolUse','tool_name':tool,'tool_input':{'command':command}}

def denied(result):
    return result['hookSpecificOutput']['permissionDecision'] == 'deny'

def test_private_added_content_is_blocked_without_echoing_it():
    content = 'C:' + '\\' + 'Users' + '\\' + 'fixture-user'
    result = decide(event('*** Update File: README.md\n+path='+content, 'apply_patch'), lambda: True)
    assert denied(result) and 'fixture-user' not in str(result)

def test_local_absolute_patch_header_is_not_public_content():
    header = 'C:' + '\\' + 'Users' + '\\' + 'fixture-user' + '\\README.md'
    assert not denied(decide(event('*** Update File: '+header+'\n+safe relative path', 'apply_patch'), lambda: True))

def test_push_and_github_publication_require_a_successful_scan():
    for command in ('git push origin main', 'git -C repo push origin topic', 'gh pr create --body safe', 'gh api --method PATCH repos/example/repo/issues/1 -f body=hello'):
        assert denied(decide(event(command), lambda: False))
        assert not denied(decide(event(command), lambda: True))

def test_read_only_commands_are_not_blocked_by_existing_history():
    assert not denied(decide(event('git status --short'), lambda: False))
    assert not denied(decide(event('gh api repos/example/repo'), lambda: False))


def test_provider_profile_blocks_new_pr_or_merge_even_when_source_is_clean():
    for command in ('gh pr create --body safe','gh pr edit 1 --base main','gh pr merge 1 --squash --author-email 123+fixture@users.noreply.github.com'):
        assert denied(decide(event(command),lambda:True,profile_checker=lambda:False))
        assert not denied(decide(event(command),lambda:True,profile_checker=lambda:True))
    def unavailable():
        raise RuntimeError('private provider response')
    assert denied(decide(event('gh pr create --body safe'),lambda:True,profile_checker=unavailable))
    assert not denied(decide(event('gh pr view 1'),lambda:True,profile_checker=unavailable))

def test_check_bypass_is_rejected_for_publication():
    assert denied(decide(event('git push --no-verify origin main'), lambda: True))

def test_github_merge_must_explicitly_use_public_author_email():
    assert denied(decide(event('gh pr merge 1 --squash'), lambda: True))
    assert not denied(decide(event('gh pr merge 1 --squash --author-email 123+fixture@users.noreply.github.com'), lambda: True))

def test_staged_scan_errors_fail_closed():
    def broken():
        raise RuntimeError('private source details must not escape')
    result=decide(event('git push'), broken)
    assert denied(result) and 'source details' not in str(result)

def test_untracked_body_file_is_checked_before_github_publication(tmp_path):
    path=tmp_path/'body.md'
    path.write_text('C:'+'\\'+'Users'+'\\'+'fixture-user'+'\\private')
    request=event('gh pr create --body-file body.md')
    request['cwd']=str(tmp_path)
    assert denied(decide(request,lambda:True))
    path.write_text('safe body')
    assert not denied(decide(request,lambda:True))

def test_missing_or_dynamic_body_file_is_not_assumed_safe(tmp_path):
    for name in ('missing.md','$bodyFile','-'):
        request=event('gh pr create --body-file '+name)
        request['cwd']=str(tmp_path)
        assert denied(decide(request,lambda:True))


def test_write_edit_and_patch_envelopes_check_actual_added_content():
    private='C:'+'\\'+'Users'+'\\'+'fixture-user'+'\\private'
    for tool, values in (
        ('Write', {'file_path':'README.md','content':private}),
        ('Edit', {'file_path':'README.md','old_string':'safe','new_string':private}),
        ('apply_patch', {'input':'*** Update File: README.md\n+'+private}),
    ):
        request=event('',tool); request['tool_input']=values
        assert denied(decide(request,lambda:True))


def test_issue_publication_checks_payload_and_outgoing_history():
    assert denied(decide(event('gh issue create --body safe'),lambda:False))


def test_release_upload_checks_actual_file(tmp_path):
    (tmp_path/'receipt.txt').write_text('fixture'+'@'+'private-domain.dev')
    request=event('gh release upload v1 receipt.txt'); request['cwd']=str(tmp_path)
    assert denied(decide(request,lambda:True))
