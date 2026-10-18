"""Audit artifact selection and model metadata before upload."""
import argparse
import glob
import json
from pathlib import Path
import sys

from .privacy import IMAGES, scan_bytes
from .model_metadata_privacy import audit_model_metadata

GENERATED_NUMERIC={'.onnx','.pt2','.data','.bin','.npy','.npz','.safetensors'}


def audit(root,patterns):
    root=Path(root).resolve()
    policy_path=root/'.privacy-reviewed-assets.json'
    reviewed={}
    if policy_path.exists():
        policy=json.loads(policy_path.read_text(encoding='utf-8-sig'))
        if policy.get('schema')!=1 or not isinstance(policy.get('assets'),dict):
            raise ValueError('Invalid artifact image-review policy')
        reviewed=policy['assets']
    def checked(path):
        path=Path(path)
        resolved=path.resolve()
        if not resolved.is_relative_to(root) or path.is_symlink():
            raise ValueError('Artifact selection escapes the workspace or uses a symlink')
        return resolved
    included=set();excluded=set()
    for pattern in patterns:
        exclude=pattern.startswith('!');pattern=pattern[1:] if exclude else pattern
        checked(root/pattern.split('*',1)[0].split('?',1)[0])
        for match in glob.glob(str(root/pattern),recursive=True):
            path=checked(match)
            paths=path.rglob('*') if path.is_dir() else [path]
            for item in paths:
                item=checked(item)
                if item.is_file():(excluded if exclude else included).add(item)
    findings=[]
    for path in sorted(included-excluded):
        relative=path.relative_to(root).as_posix()
        # Diagnostics omit member names, which may themselves be private.
        findings.extend(scan_bytes(relative.encode(),'<artifact-name>'))
        suffix=path.suffix.lower()
        if suffix in IMAGES:
            findings.extend(dict(f,path='<artifact-image>') for f in scan_bytes(path.read_bytes(),relative,reviewed))
            continue
        findings.extend(dict(f,path='<artifact-name>') for f in scan_bytes(b'',relative))
        if suffix in {'.onnx','.pt2'}:
            findings.extend(audit_model_metadata(path))
            continue
        if suffix in GENERATED_NUMERIC:
            continue
        with path.open('rb') as stream:
            overlap=b''
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                findings.extend(scan_bytes(overlap+chunk,'<artifact-content>'))
                overlap=chunk[-4096:]
                if findings:break
    return findings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--paths',required=True)
    args=parser.parse_args()
    try:
        findings=audit(args.root,[line.strip() for line in args.paths.splitlines() if line.strip()])
        print(json.dumps({'success':not findings,'count':len(findings),'findings':findings,
            'scope':'Selected filenames, text, images and PT2/ONNX textual metadata. Opaque numerical tensor buffers remain separate.'}))
        return int(bool(findings))
    except Exception:
        print('Artifact privacy validation failed; no content or private path was printed.',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
