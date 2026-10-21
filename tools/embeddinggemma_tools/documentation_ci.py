"""Small documentation checks with full ancestry and actual payload privacy."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import unicodedata
from urllib.parse import unquote, urlsplit

from .ci_scope import event_scope
from .privacy import github_profile_findings, reviews, scan_bytes, scan_objects


def _fences(text):
    result, fence = [], None
    for line in text.splitlines(keepends=True):
        marker = re.match(r'^\s{0,3}(`{3,}|~{3,})', line)
        if marker and fence is None:
            fence = marker.group(1)
            result.append('\n' if line.endswith('\n') else '')
        elif fence is not None:
            result.append('\n' if line.endswith('\n') else '')
            if marker and marker.group(1)[0] == fence[0] and len(marker.group(1)) >= len(fence):
                fence = None
        else:
            result.append(line)
    return ''.join(result)


def _anchors(path):
    text = _fences(path.read_text(encoding='utf-8-sig'))
    ids = set(re.findall(r'\b(?:id|name)=["\x27]([^"\x27]+)', text))
    counts = Counter()
    for heading in re.findall(r'^#{1,6}\s+(.+?)\s*#*\s*$', text, re.M):
        heading = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', heading).lower()
        heading = re.sub(r'<[^>]+>', '', heading)
        slug = ''.join(c for c in heading if c in '-_ ' or unicodedata.category(c)[0] in 'LNM').replace(' ', '-')
        n = counts[slug]
        counts[slug] += 1
        ids.add(slug if not n else f'{slug}-{n}')
    return ids


def check_links(root, markdown_paths):
    root = Path(root).resolve()
    findings, anchors = [], {}
    patterns = (
        r'!?\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+["\x27][^"\x27]*["\x27])?\s*\)',
        r'^\s{0,3}\[[^\]]+\]:\s*(?:<([^>]+)>|(\S+))',
        r'\b(?:href|src)=["\x27]([^"\x27]+)["\x27]',
    )
    for relative in markdown_paths:
        file = root / relative
        if not file.resolve().is_relative_to(root):
            findings.append(dict(path=relative, line=1, rule='outside-repository-link'))
            continue
        text = _fences(file.read_text(encoding='utf-8-sig'))
        text = re.sub(r'`+[^`\n]*`+', lambda m: ' ' * len(m.group()), text)
        def ref_key(value):
            return re.sub(r'\s+', ' ', value.strip()).casefold()
        definitions = {ref_key(v) for v in re.findall(r'^\s{0,3}\[([^\]]+)\]:', text, re.M)}
        for match in re.finditer(r'!?\[([^\]]+)\]\[([^\]]*)\]', text):
            if ref_key(match.group(2) or match.group(1)) not in definitions:
                findings.append(dict(path=relative, line=text[:match.start()].count('\n') + 1,
                                     rule='missing-reference-definition'))
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.M):
                target = next(value for value in match.groups() if value is not None)
                if re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith('//'):
                    continue  # External URLs are not fetched; no token/network side effects.
                parsed = urlsplit(unquote(target))
                path = ((root / parsed.path.lstrip('/')) if parsed.path.startswith('/')
                        else (file.parent / parsed.path)).resolve() if parsed.path else file.resolve()
                rule = None
                if not path.is_relative_to(root):
                    rule = 'outside-repository-link'
                elif not path.exists():
                    rule = 'missing-local-target'
                elif parsed.fragment and path.suffix.lower() == '.md':
                    if path not in anchors:
                        anchors[path] = _anchors(path)
                    if parsed.fragment not in anchors[path]:
                        rule = 'missing-local-anchor'
                if rule:
                    findings.append(dict(path=relative, line=text[:match.start()].count('\n') + 1, rule=rule))
    return findings


def check_payload(event):
    if not isinstance(event, dict):
        return [dict(path='<event>', line=1, rule='malformed-publication-payload')]
    if 'pull_request' not in event:
        return []
    pr = event['pull_request']
    if (not isinstance(pr, dict) or not isinstance(pr.get('title'), str)
            or not pr['title'] or pr.get('body') is not None and not isinstance(pr['body'], str)):
        return [dict(path='<pr-body>', line=1, rule='malformed-publication-payload')]
    return (scan_bytes(pr['title'].encode(), '<pr-title>')
            + scan_bytes((pr.get('body') or '').encode(), '<pr-body>'))


def check_whitespace(root, base, head):
    result = subprocess.run(['git', '-C', str(root), 'diff', '--check', base, head, '--'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return [] if result.returncode == 0 else [dict(path='<diff>', line=1, rule='invalid-documentation-diff')]


def _profiles(event, reader):
    unavailable = [dict(path='<github-profile>', line=1, rule='github-profile-unavailable')]
    if (not isinstance(event, dict)
            or 'repository' in event and not isinstance(event['repository'], dict)):
        return unavailable
    candidates = [event.get('sender'), event.get('repository', {}).get('owner')]
    if isinstance(event.get('pull_request'), dict):
        candidates.append(event['pull_request'].get('user'))
    logins = {v['login'] for v in candidates if isinstance(v, dict) and isinstance(v.get('login'), str)}
    if not logins:
        return [dict(path='<github-profile>', line=1, rule='github-profile-unavailable')]
    findings = []
    for login in sorted(logins):
        if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*(?:\[bot\])?', login) is None:
            findings.append(dict(path='<github-profile>', line=1, rule='github-profile-unavailable'))
            continue
        def fetch():
            value = reader(login) if reader else json.loads(subprocess.check_output(
                ['gh', 'api', '--hostname', 'github.com', f'users/{login}'],
                stderr=subprocess.PIPE, timeout=20))
            if not isinstance(value, dict) or value.get('login') != login:
                raise ValueError('Mismatched public profile')
            return value
        findings.extend(github_profile_findings(fetch_profile=fetch))
    return findings


def audit_documentation(root, scope, event, *, profile_reader=None):
    if not isinstance(scope, dict) or scope.get('mode') != 'docs':
        return dict(success=False, model_execution=False,
                    findings=[dict(path='<scope>', line=1, rule='not-documentation-scope')])
    findings = scan_objects(root, [scope['source_sha']], reviews(root))
    findings.extend(check_payload(event))
    findings.extend(_profiles(event, profile_reader))
    if not findings:
        findings.extend(check_whitespace(root, scope['base_sha'], scope['head_sha']))
        paths = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).split(b'\0')
        markdown = [p.decode('utf-8') for p in paths if p.endswith(b'.md')]
        findings.extend(check_links(root, markdown))
    return dict(success=not findings, model_execution=False, scope=scope,
                findings=findings[:100], finding_count=len(findings))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('..'))
    args = parser.parse_args(argv)
    try:
        scope = event_scope(args.root)
        event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text(encoding='utf-8'))
        report = audit_documentation(args.root, scope, event)
        print(json.dumps(report, ensure_ascii=True))
        return 0 if report['success'] else 1
    except (OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError):
        print(json.dumps(dict(success=False, error='documentation-check-failed', model_execution=False)))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
