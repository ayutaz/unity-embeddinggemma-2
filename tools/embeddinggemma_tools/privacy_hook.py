import json
import re
import shlex
from pathlib import Path
from .privacy import scan_bytes

def publication_files_safe(command, cwd):
    tokens=shlex.split(command)
    files=[]
    for index, token in enumerate(tokens):
        if token in ('--body-file','--notes-file','--input') or (token=='-F' and ' api ' not in command):
            if index+1>=len(tokens):
                return False
            files.append(tokens[index+1])
        elif token.startswith(('--body-file=','--notes-file=','--input=')):
            files.append(token.split('=',1)[1])
        elif '=@' in token:
            files.append(token.split('=@',1)[1])
    release_upload = 'release' in tokens and 'upload' in tokens
    if release_upload:
        index=tokens.index('upload')
        for token in tokens[index+2:]:
            if token in ('--clobber',):
                continue
            if token.startswith('-'):
                return False
            files.append(token.split('#',1)[0])
    for name in files:
        if name=='-' or '$' in name or '`' in name:
            return False
        path=Path(cwd)/name
        if not path.is_file():
            return False
        if release_upload:
            from .model_release_privacy import reviewed_model_asset_safe
            if reviewed_model_asset_safe(path, cwd):
                continue
        if path.stat().st_size>16*1024**2:
            return False
        if scan_bytes(path.read_bytes(), path.name):
            return False
    return True

def decide(event, checker, profile_checker=None):
    output = {'hookEventName':'PreToolUse', 'permissionDecision':'allow'}
    tool = event.get('tool_name', '')
    values = event.get('tool_input', {})
    command = values.get('command', values.get('cmd', values.get('input', values.get('patch', '')))) if isinstance(values, dict) else str(values)
    def deny():
        output.update(permissionDecision='deny', permissionDecisionReason='Privacy check blocked publication or private added content. Inspect safe rule/file/line diagnostics; do not bypass the guard.')
    try:
        if tool in ('apply_patch', 'Edit', 'Write'):
            if tool=='Write':
                added=values.get('content',command)
            elif tool=='Edit':
                added=values.get('new_string',command)
            else:
                added = '\n'.join(line[1:] for line in command.splitlines() if line.startswith('+') and not line.startswith('+++'))
            if scan_bytes(added.encode(), '<added-content>'):
                deny()
        else:
            git_push = re.search(r'\bgit\b[^\r\n;&|]*\bpush\b', command)
            github_write = (re.search(r'\bgh\s+(?:pr\s+(?:create|edit|merge)|issue\s+(?:create|edit|comment)|release\s+(?:create|upload))\b', command)
                            or (re.search(r'\bgh\s+api\b', command) and re.search(r'(?:--method|-X)\s+(?:PUT|PATCH|POST|DELETE)\b|(?:^|\s)-(?:f|F)\s', command)))
            mcp_write = tool.startswith('mcp__') and re.search(r'(?i)create|update|edit|upload|publish|push|merge|send', tool)
            if git_push or github_write or mcp_write:
                payload = json.dumps(values) if mcp_write else command
                merge_without_public_author = (re.search(r'\bgh\s+pr\s+merge\b', command)
                    and not re.search(r'--author-email\s+\S+@users\.noreply\.github\.com\b', command))
                creates_provider_commit = re.search(r'\bgh\s+pr\s+(?:create|edit|merge)\b',command)
                if ('--no-verify' in command or merge_without_public_author
                        or scan_bytes(payload.encode(), '<publication-input>')
                        or not publication_files_safe(command, event.get('cwd', Path.cwd())) or not checker()
                        or creates_provider_commit and profile_checker is not None and not profile_checker()):
                    deny()
    except Exception:
        deny()
    return {'hookSpecificOutput':output}
