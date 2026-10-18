"""Diagnose actual candidate ONNX differences without qualifying model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from . import audio_generation_candidate as candidate
from .onnx_computation_identity import fingerprint

PROFILES=('frames-1119-valid-1119','frames-1119-valid-839')
BASELINE_PATH=Path(__file__).parents[2]/'docs/results/original-native-onnx-computation-components-20261018.json'
BASELINE_SHA='fbce9fcd79b2d93f519a4c4982fb8cc7896e34eea26f0f137e9a66e61d08395d'


def baseline():
    value=json.loads(BASELINE_PATH.read_text(encoding='utf-8'))
    digest=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest!=BASELINE_SHA or set(value.get('profiles',{}))!=set(PROFILES):
        raise ValueError('Original ONNX baseline hash or coverage differs')
    return value


def inspect_native(path, *, expected_sha256, profile, source):
    if re.fullmatch('[0-9a-f]{40}',source or '') is None:
        raise ValueError('Immutable diagnostic execution source required')
    if profile not in PROFILES:raise ValueError('One original profile is required')
    original=baseline()['profiles'][profile]
    actual=fingerprint(path,expected_sha256=expected_sha256)
    equality={key:actual['components'].get(key)==value for key,value in original['components'].items()}
    return dict(schema=1,scope='actual_candidate_native_onnx_computation_identity_diagnostic_only',complete=True,
        source=source,profile=profile,original_onnx_sha256=original['source_sha256'],candidate_onnx_sha256=actual['source_sha256'],
        computation_identity_matches_original=(actual['computation_sha256']==original['computation_sha256'] and all(equality.values())),
        component_identity=equality,changed_components=sorted(key for key,same in equality.items() if not same),
        original_fingerprint=original,candidate_fingerprint=actual,
        model_executed=False,gpu_validated=False,generation_qualified=False)


def fetch_native(output, *, profile,source_archive=None):
    from .artifact_zip import verified_local_artifact_reader,extract_members
    if profile not in PROFILES:raise ValueError('One original profile is required')
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Diagnostic input output must be empty')
    contract=candidate.candidate_contract()
    reader,source_proof=verified_local_artifact_reader(source_archive,contract['artifact_sha256'])
    graph_name=profile+'/audio.onnx'
    with reader:
        extract_members(reader,output,['audio-runtime-reference/audio-encoder-validation.json','audio-runtime-reference/'+graph_name],
                        max_entry_bytes=2*1024*1024*1024)
    root=output/'audio-runtime-reference';audit_path=root/'audio-encoder-validation.json'
    if candidate._digest(audit_path)!=contract['audit_sha256']:
        raise ValueError('Pinned producer audit hash differs')
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    if (audit['metadata']['source_commit']!=contract['producer_source']
            or audit['metadata']['model_revision']!=contract['model_revision'] or audit['native_reference_complete'] is not True):
        raise ValueError('Candidate native ONNX producer metadata differs')
    spec=audit['files'][graph_name]
    path=candidate.checked_file(root,graph_name,spec)
    return path,spec['sha256'],dict(candidate_artifact_id=contract['artifact_id'],candidate_producer_run=contract['producer_run'],
        candidate_producer_source=contract['producer_source'],candidate_contract_sha256=candidate.CONTRACT_SHA,
        candidate_audit_sha256=contract['audit_sha256'],source_archive_transport=source_proof['transport'],
        source_archive_sha256=source_proof['archive_sha256'],provider_metadata_revalidated=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--model-sha256')
    parser.add_argument('--profile',choices=PROFILES,required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fetch',action='store_true')
    parser.add_argument('--source-archive',type=Path)
    args=parser.parse_args(argv)
    if args.output.exists():
        print('Diagnostic output already exists; no file was overwritten.')
        return 1
    if re.fullmatch('[0-9a-f]{40}',args.source) is None:
        print('Immutable diagnostic source required.')
        return 1
    try:
        if args.fetch:path,sha,provenance=fetch_native(args.input,profile=args.profile,source_archive=args.source_archive)
        else:path,sha,provenance=args.input,args.model_sha256,{}
        report=inspect_native(path,expected_sha256=sha,profile=args.profile,source=args.source)
        report.update(provenance)
    except Exception as error:
        report=dict(schema=1,scope='candidate_native_onnx_identity_diagnostic_failed_before_complete_comparison',
            source=args.source,profile=args.profile,complete=False,error_type=type(error).__name__,
            model_executed=False,gpu_validated=False,generation_qualified=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    matched=report.get('computation_identity_matches_original') is True
    print(json.dumps(dict(complete=report['complete'],computation_identity_matches_original=matched,gpu_validated=False)))
    return 0 if matched else 1


if __name__=='__main__':
    raise SystemExit(main())
