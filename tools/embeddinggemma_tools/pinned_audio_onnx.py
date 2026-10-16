"""Compare every audited saved ONNX output to the immutable original audio golden."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SIGNALS = {'silence', 'impulse', 'sine', 'nyquist', 'noise', 'padded_noise'}


def compare(current, pinned):
    current, pinned = Path(current).resolve(), Path(pinned).resolve()
    def read(root):
        audit = json.loads((root/'audio-encoder-validation.json').read_text())
        path = check(root, audit, 'audio-encoder-reference.json')
        reference = json.loads(path.read_text())
        if reference['metadata'] != audit['metadata']:
            raise ValueError('reference/audit metadata mismatch')
        if len(reference['cases']) != 6 or {c['id'] for c in reference['cases']} != SIGNALS or len(reference['profiles']) != 2:
            raise ValueError('complete original six-signal/two-profile matrix required')
        return reference, audit
    def check(root, audit, name):
        path = (root/name).resolve()
        spec = audit['files'].get(name)
        if not path.is_relative_to(root) or spec is None or not path.is_file():
            raise ValueError('missing or escaped audited path')
        with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.stat().st_size != spec['bytes'] or digest != spec['sha256']:
            raise ValueError('audited file size/hash mismatch')
        return path
    new, new_audit = read(current)
    old, old_audit = read(pinned)
    for name in ('model_revision', 'input_sha256', 'versions'):
        if new['metadata'][name] != old['metadata'][name]:
            raise ValueError('original model or library provenance mismatch')
    for metadata in (new['metadata'],old['metadata']):
        if metadata['feature_atol'] != 1e-5 or metadata['feature_rtol'] != 1e-4:
            raise ValueError('original tolerance changed')
    def manifest_sha(audit, name): return audit['files'][name]['sha256']
    originals = {r['id']: r for r in old['cases']}
    rows = []
    for row in new['cases']:
        original = originals[row['id']]
        profile = row['profile']
        if profile != original['profile']:
            raise ValueError('original case profile changed')
        spec, old_spec = new['profiles'][profile], old['profiles'][profile]
        for field in ('feature_count','feature_width'):
            if spec[field] != old_spec[field]: raise ValueError('original feature shape changed')
        for new_name, old_name in ((row['input_features'], original['input_features']), (spec['frame_mask'],old_spec['frame_mask'])):
            check(current,new_audit,new_name)
            if manifest_sha(new_audit,new_name) != manifest_sha(old_audit,old_name):
                raise ValueError('original input/mask hash changed')
        check(current,new_audit,spec['graph'])
        if not row.get('onnx_features'): raise ValueError('missing saved actual ONNX output')
        path = check(current,new_audit,row['onnx_features'])
        expected = np.asarray(original['features'],dtype=np.float32)
        shape = (spec['feature_count'],spec['feature_width'])
        actual = np.fromfile(path,dtype='<f4')
        native = np.asarray(row['features'],dtype=np.float32)
        if expected.shape != shape or native.shape != shape or actual.size != expected.size:
            raise ValueError('original output shape mismatch')
        actual = actual.reshape(shape)
        if not all(np.isfinite(a).all() for a in (actual,expected,native)):
            raise ValueError('nonfinite reference or actual')
        difference = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
        allowed = 1e-5 + 1e-4*np.abs(expected.astype(np.float64))
        denominator=np.linalg.norm(actual.astype(np.float64),axis=-1)*np.linalg.norm(expected.astype(np.float64),axis=-1)
        if np.any(denominator==0): raise ValueError('zero feature vector')
        cosine=float(np.min(np.sum(actual.astype(np.float64)*expected,axis=-1)/denominator))
        mismatches=int(np.count_nonzero(difference>allowed))
        native_diff=np.abs(native.astype(np.float64)-expected.astype(np.float64))
        rows.append(dict(id=row['id'],profile=profile,success=mismatches==0 and cosine>=0.999999,
            mismatched_elements=mismatches,maximum_absolute_error=float(difference.max()),
            maximum_tolerance_ratio=float((difference/allowed).max()),minimum_cosine=cosine,
            actual_sha256=manifest_sha(new_audit,row['onnx_features']),graph_sha256=manifest_sha(new_audit,spec['graph']),
            native_exact_difference_count=int(np.count_nonzero(native!=expected)),
            native_fixed_mismatched_elements=int(np.count_nonzero(native_diff>allowed))))
    return dict(scope='saved_onnx_vs_immutable_original_audio_golden_only',success=all(r['success'] for r in rows),
        source=new['metadata']['source_commit'],reference_source=old['metadata']['source_commit'],
        case_count=6,profile_count=2,failed_case_count=sum(not r['success'] for r in rows),cases=rows,
        native_context_bit_matches_original=all(r['native_exact_difference_count']==0 for r in rows),
        native_context_fixed_matches_original=all(r['native_fixed_mismatched_elements']==0 for r in rows),
        native_context_required_for_localization=True, feature_atol=1e-5,feature_rtol=1e-4,
        unity_validated=False,end_to_end_validated=False,original_golden_changed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current',type=Path,required=True)
    parser.add_argument('--pinned',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    output=args.output.resolve()
    if any(output.is_relative_to(root.resolve()) for root in (args.current,args.pinned)):
        raise ValueError('report must not overwrite either source directory')
    report=compare(args.current,args.pinned)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(success=report['success'],case_count=report['case_count'],
        failed_case_count=report['failed_case_count'],native_context_fixed_matches_original=report['native_context_fixed_matches_original'])))
    return 0 if report['success'] else 1


if __name__ == '__main__': raise SystemExit(main())
