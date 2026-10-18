import json
import re
from .privacy import scan_bytes

def decide(event, checker):
    output = {'hookEventName':'PreToolUse', 'permissionDecision':'allow'}
    tool = event.get('tool_name', '')
    values = event.get('tool_input', {})
    command = values.get('command', values.get('cmd', '')) if isinstance(values, dict) else str(values)
    def deny():
        output.update(permissionDecision='deny', permissionDecisionReason='Privacy check blocked publication or private added content. Inspect safe rule/file/line diagnostics; do not bypass the guard.')
    try:
        if tool in ('apply_patch', 'Edit', 'Write'):
            added = '\n'.join(line[1:] for line in command.splitlines() if line.startswith('+') and not line.startswith('+++'))
            if scan_bytes(added.encode(), '<added-content>'):
                deny()
        else:
            git_push = re.search(r'\bgit\b[^\r\n;&|]*\bpush\b', command)
            github_write = (re.search(r'\bgh\s+(?:pr\s+(?:create|edit|merge)|release\s+(?:create|upload))\b', command)
                            or (re.search(r'\bgh\s+api\b', command) and re.search(r'(?:--method|-X)\s+(?:PUT|PATCH|POST|DELETE)\b|(?:^|\s)-(?:f|F)\s', command)))
            mcp_write = tool.startswith('mcp__') and re.search(r'(?i)create|update|edit|upload|publish|push|merge|send', tool)
            if git_push or github_write or mcp_write:
                payload = json.dumps(values) if mcp_write else command
                merge_without_public_author = (re.search(r'\bgh\s+pr\s+merge\b', command)
                    and not re.search(r'--author-email\s+\S+@users\.noreply\.github\.com\b', command))
                if '--no-verify' in command or merge_without_public_author or scan_bytes(payload.encode(), '<publication-input>') or not checker():
                    deny()
    except Exception:
        deny()
    return {'hookSpecificOutput':output}
