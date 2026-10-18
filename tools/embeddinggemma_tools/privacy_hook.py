import json
import re
import shlex
from pathlib import Path
from .privacy import scan_bytes

def _release_invocations(tokens):
    starts = []
    for index, token in enumerate(tokens):
        if token != 'gh':
            continue
        index += 1
        while index < len(tokens):
            if tokens[index] in ('--repo', '-R') and index+1 < len(tokens):
                index += 2
            elif tokens[index].startswith('--repo='):
                index += 1
            else:
                break
        if index+1 < len(tokens) and tokens[index] == 'release' and tokens[index+1] in ('create', 'new', 'upload'):
            starts.append((index, 'create' if tokens[index+1] == 'new' else tokens[index+1]))
    return starts


def _release_asset_arguments(tokens):
    """Parse explicit release files; unknown flag arity must never hide an asset."""
    starts = _release_invocations(tokens)
    if not starts:
        return []
    if len(starts) != 1:
        return None
    start, verb = starts[0]
    values = {'--repo', '-R'}
    switches = {'--clobber'} if verb == 'upload' else {
        '--draft', '-d', '--prerelease', '-p', '--generate-notes', '--notes-from-tag',
        '--verify-tag', '--fail-on-no-commits', '--latest'}
    if verb == 'create':
        values |= {'--title', '-t', '--notes', '-n', '--notes-file', '-F', '--target',
                   '--discussion-category', '--notes-start-tag'}
    positional = []
    index = start + 2
    literal = False
    while index < len(tokens):
        token = tokens[index]
        if token == '--' and not literal:
            literal = True
        elif token.startswith('-') and not literal:
            option, equal, value = token.partition('=')
            if option in values:
                if not equal:
                    index += 1
                    if index >= len(tokens) or tokens[index].startswith('-'):
                        return None
                    value = tokens[index]
                if not value:
                    return None
            elif option in switches:
                if equal and value not in ('true', 'false'):
                    return None
            else:
                return None
        else:
            positional.append(token)
        index += 1
    if not positional or (verb == 'upload' and len(positional) < 2):
        return None
    if any(not v or v in (';', '&&', '||', '|') for v in positional):
        return None
    return [token.split('#', 1)[0] for token in positional[1:]]


def publication_files_safe(command, cwd):
    tokens=shlex.split(command)
    assets = _release_asset_arguments(tokens)
    if assets is None:
        return False
    files=[]
    for index, token in enumerate(tokens):
        if token in ('--body-file','--notes-file','--input') or (token=='-F' and ' api ' not in command):
            if index+1>=len(tokens):
                return False
            files.append((tokens[index+1], False))
        elif token.startswith(('--body-file=','--notes-file=','--input=')):
            files.append((token.split('=',1)[1], False))
        elif '=@' in token:
            files.append((token.split('=@',1)[1], False))
    files.extend((name, True) for name in assets)
    for name, release_asset in files:
        if name=='-' or '$' in name or '`' in name:
            return False
        path=Path(cwd)/name
        if not path.is_file():
            return False
        if release_asset:
            from .model_release_privacy import reviewed_model_asset_safe
            if reviewed_model_asset_safe(path, cwd):
                continue
            from .native_regression_release_privacy import reviewed_native_asset_safe
            if reviewed_native_asset_safe(path, cwd):
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
                            or (re.search(r'\bgh\s+api\b', command) and re.search(r'(?:--method|-X)\s+(?:PUT|PATCH|POST|DELETE)\b|(?:^|\s)-(?:f|F)\s', command))
                            or (re.search(r'\bgh\b[^\r\n]*\brelease\s+(?:create|new|upload)\b', command)
                                and _release_invocations(shlex.split(command))))
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
