"""Exercise the real hook entry point with a legacy Windows pipe encoding."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def run_hook(added):
    event = {
        'hook_event_name': 'PreToolUse',
        'tool_name': 'apply_patch',
        'cwd': str(ROOT),
        'tool_input': {'command': '*** Update File: docs/roadmap.md\n+' + added},
    }
    environment = dict(os.environ, PYTHONIOENCODING='cp932:surrogateescape')
    result = subprocess.run(
        [sys.executable, str(ROOT / 'tools/privacy_hook.py')],
        input=json.dumps(event, ensure_ascii=False).encode('utf-8'),
        capture_output=True, cwd=ROOT, env=environment, check=True,
    )
    return json.loads(result.stdout)


def test_utf8_japanese_patch_is_allowed_under_legacy_windows_pipe_encoding():
    added = '\u6b21\u306e\u8a08\u753b\u306fCI\u4fee\u6b63\u3068\u53c2\u7167\u5fa9\u65e7\u3067\u3059\u3002'
    result = run_hook(added)
    assert result['hookSpecificOutput']['permissionDecision'] == 'allow'


def test_utf8_patch_with_private_path_still_fails_without_echoing_it():
    added = '\u8cc7\u6599: ' + 'C:' + '\\' + 'Users' + '\\' + 'fixture-user' + '\\private'
    result = run_hook(added)
    assert result['hookSpecificOutput']['permissionDecision'] == 'deny'
    assert 'fixture-user' not in json.dumps(result)
