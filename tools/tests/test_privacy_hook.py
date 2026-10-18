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
    for command in ('git push origin main', 'git -C repo push origin topic', 'gh pr create --body-file body.md', 'gh api --method PATCH repos/example/repo/issues/1 -f body=hello'):
        assert denied(decide(event(command), lambda: False))
        assert not denied(decide(event(command), lambda: True))

def test_read_only_commands_are_not_blocked_by_existing_history():
    assert not denied(decide(event('git status --short'), lambda: False))
    assert not denied(decide(event('gh api repos/example/repo'), lambda: False))

def test_check_bypass_is_rejected_for_publication():
    assert denied(decide(event('git push --no-verify origin main'), lambda: True))

def test_staged_scan_errors_fail_closed():
    def broken():
        raise RuntimeError('private source details must not escape')
    result=decide(event('git push'), broken)
    assert denied(result) and 'source details' not in str(result)
