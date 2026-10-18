"""Fail closed before publishing private paths, identities or unchecked images."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

HOME = re.compile(r"(?i)(?:\b[a-z]:[/\\]+(?:Users|Documents and Settings)[/\\]+[^/\\\s\"'<>]+|(?<![\w])/(?:Users|home)/[^/\s\"'<>]+)")
EMAIL = re.compile(r'(?i)(?<![a-z0-9._%+\-])[a-z0-9._%+\-]+@(?:gmail|googlemail|yahoo|hotmail|outlook|live|icloud|protonmail|proton)\.[a-z.]{2,}')
SECRET = re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,}|hf_[A-Za-z0-9]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{25,})\b')
IMAGES = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tif', '.tiff'}

def normalize(text):
    for _ in range(3):
        decoded = html.unescape(unquote(text))
        if decoded == text:
            break
        text = decoded
    return text

def scan_bytes(data, path='<input>', reviewed=None):
    review = (reviewed or {}).get(path, {})
    image = Path(path).suffix.lower() in IMAGES
    if image:
        approved = (review.get('sha256') == hashlib.sha256(data).hexdigest()
                    and review.get('review') == 'visual+ocr'
                    and review.get('contains_personal_data') is False)
        return [] if approved else [dict(path=path, line=1, rule='unreviewed-image')]
    if data.startswith((b'\xff\xfe', b'\xfe\xff')):
        text = data.decode('utf-16', errors='replace')
    elif b'\x00' in data:
        if (path.startswith('tools/tests/fixtures/') and path.endswith('.bin')
                and review.get('sha256') == hashlib.sha256(data).hexdigest()
                and review.get('review') == 'numeric-source-verified'):
            return []
        return [dict(path=path, line=1, rule='unreviewed-binary')]
    else:
        text = data.decode('utf-8', errors='replace')
    findings = []
    for number, line in enumerate(normalize(text).splitlines(), 1):
        for name, pattern in (('home-path', HOME), ('personal-email', EMAIL), ('credential', SECRET)):
            if pattern.search(line):
                findings.append(dict(path=path, line=number, rule=name))
    return findings

def scan_identity(name, email):
    safe_email = email.endswith('@users.noreply.github.com') or email == 'noreply@github.com'
    safe_name = re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_\-]*(?:\[bot\])?', name) is not None
    handle = email.split('@', 1)[0].split('+')[-1]
    safe_name = safe_name and (name == handle or (name == 'GitHub' and email == 'noreply@github.com'))
    return [] if safe_email and safe_name else [dict(path='<commit-identity>', line=1, rule='private-identity')]

def git(*args, root=None):
    return subprocess.check_output(['git', *args], cwd=root)

def reviews(root, staged=False):
    try:
        data = git('show', ':.privacy-reviewed-assets.json' if staged else 'HEAD:.privacy-reviewed-assets.json', root=root)
        policy = json.loads(data)
        if policy.get('schema') != 1 or not isinstance(policy.get('assets'), dict):
            raise ValueError('Invalid image-review policy')
        return policy['assets']
    except subprocess.CalledProcessError:
        return {}

def scan_objects(root, revisions, reviewed=None):
    """Scan unique reachable objects and commit metadata, including old ancestors."""
    objects = git('rev-list', '--objects', *revisions, root=root).splitlines()
    metadata = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=root,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    findings, seen = [], set()
    try:
        for record in objects:
            oid, separator, raw_path = record.partition(b' ')
            if oid in seen:
                continue
            seen.add(oid)
            metadata.stdin.write(oid + b'\n'); metadata.stdin.flush()
            header = metadata.stdout.readline().split()
            if len(header) != 3 or header[1] not in (b'blob', b'commit', b'tree', b'tag'):
                raise ValueError('Cannot read a requested Git object')
            data = metadata.stdout.read(int(header[2]))
            if metadata.stdout.read(1) != b'\n':
                raise ValueError('Truncated Git object stream')
            if header[1] == b'blob':
                path = raw_path.decode('utf-8', errors='replace') if separator else '<blob>'
                findings.extend(scan_bytes(data, path, reviewed))
            elif header[1] in (b'commit', b'tag'):
                text = data.decode('utf-8', errors='replace')
                headers, _, message = text.partition('\n\n')
                findings.extend(scan_bytes(message.encode(), '<commit-message>'))
                for line in headers.splitlines():
                    if line.startswith(('author ', 'committer ', 'tagger ')):
                        match = re.match(r'\w+ (.*?) <([^>]*)>', line)
                        if not match:
                            raise ValueError('Malformed commit identity')
                        findings.extend(scan_identity(*match.groups()))
    finally:
        metadata.stdin.close(); metadata.stdout.close(); metadata.wait(timeout=30)
    return findings

def scan_staged(root):
    findings = []
    reviewed = reviews(root, staged=True)
    for raw_path in git('diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z', root=root).split(b'\0'):
        if raw_path:
            path = raw_path.decode('utf-8')
            findings.extend(scan_bytes(git('show', ':' + path, root=root), path, reviewed))
    name = git('config', 'user.name', root=root).decode().strip()
    email = git('config', 'user.email', root=root).decode().strip()
    findings.extend(scan_identity(name, email))
    return findings

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['staged', 'history', 'tree', 'pre-push', 'commit-message'])
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--message-file', type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.mode == 'staged':
            findings = scan_staged(root)
        elif args.mode == 'commit-message':
            findings = scan_bytes(args.message_file.read_bytes(), '<commit-message>')
        elif args.mode == 'pre-push':
            revisions = []
            for line in sys.stdin:
                fields = line.split()
                if len(fields) != 4:
                    raise ValueError('Invalid Git pre-push input')
                if set(fields[1]) != {'0'}:
                    revisions.append(fields[1])
            findings = scan_objects(root, revisions, reviews(root)) if revisions else []
        elif args.mode == 'history':
            findings = scan_objects(root, ['--all'], reviews(root))
        else:
            findings = []
            reviewed = reviews(root)
            for row in git('ls-tree', '-r', '-z', 'HEAD', root=root).split(b'\0'):
                if row:
                    entry, path = row.split(b'\t', 1)
                    findings.extend(scan_bytes(git('cat-file', 'blob', entry.split()[2].decode(), root=root), path.decode(), reviewed))
        # Only rule/file/line are emitted. Never print matching source or identities.
        print(json.dumps({'success': not findings, 'count': len(findings), 'findings': findings[:100]}))
        return 1 if findings else 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({'success': False, 'error': type(error).__name__}))
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
