"""Reconstruct exact original golden JSON from retained FP32 output captures.

This recovers numerical reference bytes, not the missing original model graphs.
"""
import argparse
import array
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

from .pinned_reference_archive import AUDIT_SHA, REFERENCE_SHA, REPO


def capture_path(root, entry):
    case,member=entry['case'],entry['member']
    relative=PurePosixPath(member)
    if (not isinstance(case,str) or re.fullmatch(r'[a-z0-9_]+',case) is None
            or relative.is_absolute() or '..' in relative.parts or '\\' in member or ':' in member):
        raise ValueError('Unsafe original capture path')
    root=Path(root).resolve();path=root/case/member
    if not path.resolve().is_relative_to(root) or any(p.is_symlink() for p in (path,*path.parents) if p.is_relative_to(root)):
        raise ValueError('Original capture escapes its directory or uses a symlink')
    return path


def reconstruct(contract, captures, *, expected_sha=REFERENCE_SHA):
    reference=copy.deepcopy(contract['template'])
    rows=reference['cases'];entries=contract['captures']
    if (contract.get('schema')!=1 or not rows or len(rows)!=len(entries)
            or len({row['id'] for row in rows})!=len(rows)
            or [row['id'] for row in rows]!=[entry['case'] for entry in entries]):
        raise ValueError('Original golden capture coverage differs')
    for row,entry in zip(rows,entries,strict=True):
        raw=capture_path(captures,entry).read_bytes()
        count,width=entry['rows'],entry['columns']
        if (type(count) is not int or type(width) is not int or count<=0 or width<=0
                or len(raw)!=entry['bytes'] or len(raw)!=count*width*4
                or hashlib.sha256(raw).hexdigest()!=entry['sha256']):
            raise ValueError('Original golden capture bytes differ')
        values=array.array('f');values.frombytes(raw)
        if values.itemsize!=4:raise ValueError('FP32 capture format unavailable')
        if sys.byteorder!='little':values.byteswap()
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Original golden contains a nonfinite value')
        row['features']=[list(values[start:start+width]) for start in range(0,len(values),width)]
    raw=(json.dumps(reference,indent=2,allow_nan=False)+'\n').encode('utf-8')
    if hashlib.sha256(raw).hexdigest()!=expected_sha:
        raise ValueError('Reconstructed JSON is not the immutable original')
    return raw


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--download',action='store_true',help='Fetch only the six pinned original output captures')
    args=parser.parse_args(argv)
    try:
        references=Path(__file__).parents[1]/'references'
        contract=json.loads((references/'original-audio-golden.json').read_text(encoding='utf-8'))
        if args.download:
            from .artifact_zip import github_artifact_reader,extract_members
            for entry in contract['captures']:
                capture_path(args.captures,entry)
                reader,_=github_artifact_reader(REPO,entry['artifact_id'])
                with reader:
                    extract_members(reader,args.captures/entry['case'],[entry['member']],max_entry_bytes=2*1024*1024)
        raw=reconstruct(contract,args.captures)
        audit=(references/'original-audio-validation.json').read_bytes()
        if hashlib.sha256(audit).hexdigest()!=AUDIT_SHA:
            raise ValueError('Original audit bytes differ')
        args.output.mkdir(parents=True,exist_ok=True)
        payloads={'audio-encoder-reference.json':raw,'audio-encoder-validation.json':audit}
        for name,data in payloads.items():
            target=args.output/name
            if target.exists() and target.read_bytes()!=data:
                raise ValueError('Never overwrite a different preserved reference')
        for name,data in payloads.items():
            target=args.output/name
            if not target.exists():target.write_bytes(data)
        print(json.dumps({'success':True,'scope':'original_reference_json_and_audit_only',
              'reference_sha256':REFERENCE_SHA,'audit_sha256':AUDIT_SHA,
              'models_recovered':False,'gpu_validated':False}))
        return 0
    except Exception as exc:
        print(json.dumps({'success':False,'error_type':type(exc).__name__,'models_recovered':False,'gpu_validated':False}))
        return 1


if __name__=='__main__':
    raise SystemExit(main())
