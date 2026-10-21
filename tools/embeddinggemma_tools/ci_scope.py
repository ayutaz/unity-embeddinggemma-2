"""Conservative, source-bound CI selection. Never print changed filenames."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess


SHA = re.compile(r'[0-9a-f]{40}\Z')
SCOPE_FIELDS = ('mode', 'reason', 'base_sha', 'head_sha', 'source_sha', 'change_digest')


def classify_paths(paths):
    if not isinstance(paths, list) or not paths:
        return 'full'
    for path in paths:
        if (not isinstance(path, str) or not path or '\\' in path
                or any(ord(c) < 32 or ord(c) == 127 for c in path)
                or any(p in ('', '.', '..') for p in path.split('/'))):
            return 'full'
        parts = PurePosixPath(path).parts
        if path == 'README.md':
            continue
        if (parts[0] != 'docs' or len(parts) < 2 or not path.endswith('.md')
                or path == 'docs/privacy.md'):
            return 'full'
    return 'docs'


def _git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE)


def _sha(value):
    return value if isinstance(value, str) and SHA.fullmatch(value) else '0' * 40


def _proof(mode, reason, base, head, source, paths=()):
    context = [mode, reason, base, head, source, sorted(paths)]
    return dict(mode=mode, reason=reason, base_sha=base, head_sha=head,
                source_sha=source, change_digest=hashlib.sha256(
                    json.dumps(context, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest())


def resolve_scope(root, event_name, event, head_sha):
    """Recompute from actual Git bytes, defaulting to full on any uncertainty."""
    head, base, source = _sha(head_sha), '0' * 40, '0' * 40
    try:
        if event_name == 'pull_request':
            base = _sha(event['pull_request']['base']['sha'])
            source = _sha(event['pull_request']['head']['sha'])
        elif event_name == 'push':
            base, source = _sha(event['before']), _sha(event['after'])
            if source != head:
                raise ValueError('Mismatched push source')
        else:
            return _proof('full', 'manual-or-unknown-event', base, head, head)
        if any(value == '0' * 40 for value in (base, head, source)):
            raise ValueError('Missing comparison source')
        if _git(root, 'rev-parse', 'HEAD').decode().strip() != head:
            raise ValueError('Checkout does not match event')
        for parent in (base, source):
            _git(root, 'merge-base', '--is-ancestor', parent, head)
        raw = _git(root, 'diff', '--name-only', '--no-renames', '-z', base, head, '--')
        paths = [p.decode('utf-8') for p in raw.split(b'\0') if p]
        mode = classify_paths(paths)
        if mode == 'docs':
            for path in paths:
                for revision in (base, head):
                    row = _git(root, 'ls-tree', '-z', revision, '--', path)
                    if row and row.split(b' ', 1)[0] != b'100644':
                        mode = 'full'  # Symlinks, submodules and executable Markdown.
        return _proof(mode, 'documentation' if mode == 'docs' else 'code-or-unknown',
                      base, head, source, paths)
    except (KeyError, TypeError, ValueError, OSError, UnicodeError, subprocess.SubprocessError):
        return _proof('full', 'unverified-context', base, head, source)


def event_scope(root, *, event_name=None, event_path=None, head_sha=None):
    path = event_path or os.environ.get('GITHUB_EVENT_PATH')
    event = json.loads(Path(path).read_text(encoding='utf-8')) if path else {}
    return resolve_scope(root, event_name or os.environ.get('GITHUB_EVENT_NAME'), event,
                         head_sha or os.environ.get('GITHUB_SHA'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('..'))
    args = parser.parse_args(argv)
    try:
        scope = event_scope(args.root)
        if os.environ.get('GITHUB_OUTPUT'):
            with Path(os.environ['GITHUB_OUTPUT']).open('a', encoding='utf-8') as output:
                output.write(''.join(f'{key}={scope[key]}\n' for key in SCOPE_FIELDS))
        print(json.dumps(scope))
        return 0
    except (OSError, ValueError, TypeError):
        print(json.dumps({'success': False, 'error': 'invalid-ci-context'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
