"""Fail closed before publishing private paths, identities or unchecked images."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

HOME = re.compile(r"(?i)(?:\b[a-z]:[/\\]+(?:Users|Documents and Settings)[/\\]+[^/\\\s\"'<>]+|(?<![\w])/(?:Users|home)/[^/\s\"'<>]+)")
EMAIL = re.compile(r'(?i)(?<![a-z0-9._%+\-])[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}')
SECRET = re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,}|hf_[A-Za-z0-9]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{25,}|AKIA[A-Z0-9]{16})\b|-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----')
IMAGES = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tif', '.tiff'}

def public_email(value):
    domain=value.rsplit('@',1)[-1].lower()
    return (value.lower()=='noreply@github.com' or domain=='users.noreply.github.com'
            or domain in {'example.com','example.org','example.net'} or domain.endswith(('.invalid','.test','.example')))


def private_email_in_text(text):
    for match in EMAIL.finditer(text):
        if public_email(match.group()):
            continue
        # GitHub's fixed SSH service user is a URI component, not a mailbox.
        # A standalone address, another user or another host is still rejected.
        if (match.group().lower()=='git'+'@github.com'
                and re.match(r'[:/][A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',text[match.end():])):
            continue
        return True
    return False


def github_profile_findings(remote_url=None, *, fetch_profile=None):
    """Provider-generated PR commits can use the account name, bypassing Git hooks.

    This only reads the authenticated profile. It never changes account settings
    or prints names, email addresses, tokens, API bodies or remote credentials.
    """
    try:
        if remote_url is not None:
            github = (urlsplit(remote_url).hostname == 'github.com'
                      or re.match(r'(?i)^[^/@]+@github\.com:', remote_url) is not None)
            if not github:
                return []
        if fetch_profile is None:
            profile=json.loads(subprocess.check_output(
                ['gh','api','--hostname','github.com','user'],stderr=subprocess.PIPE,timeout=15))
        else:
            profile=fetch_profile()
        if (not isinstance(profile,dict) or not isinstance(profile.get('login'),str)
                or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*(?:\[bot\])?',profile['login']) is None
                or profile.get('name') is not None and not isinstance(profile['name'],str)):
            raise ValueError('Malformed account profile')
        if profile.get('name') not in (None,'',profile['login']):
            return [dict(path='<github-profile>',line=1,rule='noncanonical-github-display-name')]
        return []
    except Exception:
        return [dict(path='<github-profile>',line=1,rule='github-profile-unavailable')]

def normalize(text):
    for _ in range(3):
        decoded = html.unescape(unquote(text))
        if decoded == text:
            break
        text = decoded
    return text

def scan_bytes(data, path='<input>', reviewed=None):
    normalized_path=normalize(path)
    if (HOME.search(normalized_path) or SECRET.search(normalized_path)
            or any(not public_email(m.group()) for m in EMAIL.finditer(normalized_path))):
        return [dict(path='<private-filename>',line=1,rule='private-filename')]
    filename=Path(path).name.lower()
    if ((filename.startswith('.env') and filename not in {'.env.example','.env.sample','.env.template'})
            or filename in {'.npmrc','.pypirc','id_rsa','id_ed25519'} or Path(path).suffix.lower() in {'.pem','.key','.pfx','.p12'}):
        return [dict(path=path,line=1,rule='private-config')]
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
            if (private_email_in_text(line) if name=='personal-email' else pattern.search(line)):
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
    parser.add_argument('mode', choices=['staged', 'history', 'tree', 'pre-push', 'commit-message', 'github-profile'])
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--message-file', type=Path)
    parser.add_argument('--remote-url',help='Push destination; local and non-GitHub remotes need no GitHub profile lookup')
    parser.add_argument('--revision',help='Full source commit SHA; includes all its ancestors')
    args = parser.parse_args(argv)
    if args.revision and (args.mode!='history' or re.fullmatch(r'[0-9a-f]{40}',args.revision) is None):
        parser.error('--revision requires history mode and a full lowercase commit SHA')
    root = args.root.resolve()
    try:
        if args.mode == 'github-profile':
            findings = github_profile_findings(args.remote_url)
        elif args.mode == 'staged':
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
            if revisions:
                findings.extend(github_profile_findings(args.remote_url))
        elif args.mode == 'history':
            findings = scan_objects(root, [args.revision] if args.revision else ['--all'], reviews(root))
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
