"""Explicit, unqualified regenerated audio graphs with unchanged original goldens.

The original frozen loader and mandatory acceptance gate remain independent.
Only the reviewed generation's exact bytes may enter this opt-in diagnostic.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np
import torch

from .audio_native_norm_replay import AUDIT_SHA, PRODUCER, REFERENCE_SHA, REVISION, checked_normalizations

CONTRACT_PATH = Path(__file__).parents[1]/'references/audio-candidate-e9de158.json'
CONTRACT_SHA = 'df73e109fc6dd62e8b1fff409930042b63e6c63548865b074b593dcc70c40deb'
ORIGINAL_INPUT_ARTIFACT = 12130281114
ORIGINAL_INPUT_ARCHIVE_SHA = '3b61c62e3b3686aeb4615eb4b733f068f303dddfb3f23a084ebc5b04a854f603'
ORIGINAL_INPUT_RUN = 39110353490
ORIGINAL_INPUT_SOURCE = '2971d748fe824ea4bcc580a2ec583f2f734b7420'


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def _relative(name):
    if (not isinstance(name,str) or not name or '\\' in name or ':' in name
            or PurePosixPath(name).is_absolute() or any(p in ('','.','..') for p in name.split('/'))):
        raise ValueError('Canonical relative candidate paths are required')
    return name


def checked_file(root,name,entry):
    root=Path(root).resolve();path=root/_relative(name)
    if (not path.resolve().is_relative_to(root)
            or any(p.is_symlink() for p in (path,*path.parents) if p.is_relative_to(root))
            or not path.is_file() or type(entry.get('bytes')) is not int or entry['bytes']<=0
            or re.fullmatch('[0-9a-f]{64}',entry.get('sha256','')) is None
            or path.stat().st_size!=entry['bytes'] or _digest(path)!=entry['sha256']):
        raise ValueError('Audited candidate file path, size or hash differs')
    return path


def candidate_contract():
    if _digest(CONTRACT_PATH)!=CONTRACT_SHA:
        raise ValueError('Candidate contract hash differs')
    contract=json.loads(CONTRACT_PATH.read_text(encoding='utf-8'))
    if (contract.get('schema')!=1 or contract.get('qualified') is not False
            or re.fullmatch('audio-[a-z0-9-]+',contract.get('generation','')) is None
            or contract.get('model_revision')!=REVISION
            or any(re.fullmatch('[0-9a-f]{40}',contract.get(k,'')) is None
                   for k in ('producer_source','source_tree','dependency_lock_blob'))
            or any(type(contract.get(k)) is not int or contract[k]<=0 for k in ('producer_run','artifact_id'))
            or any(re.fullmatch('[0-9a-f]{64}',contract.get(k,'')) is None
                   for k in ('artifact_sha256','reference_sha256','audit_sha256'))
            or not isinstance(contract.get('graphs'),dict) or not contract['graphs']):
        raise ValueError('A fixed unqualified candidate generation is required')
    for name,entry in contract['graphs'].items():
        _relative(name)
        if (not name.endswith('.pt2') or not isinstance(entry,dict) or type(entry.get('bytes')) is not int
                or entry['bytes']<=0 or re.fullmatch('[0-9a-f]{64}',entry.get('sha256','')) is None):
            raise ValueError('Fixed native graph descriptors are required')
    return contract


def original_reference(directory):
    root=Path(directory).resolve()
    for name,expected in [('audio-encoder-reference.json',REFERENCE_SHA),('audio-encoder-validation.json',AUDIT_SHA)]:
        if _digest(root/name)!=expected:
            raise ValueError('Original reference/audit hash differs')
    reference=json.loads((root/'audio-encoder-reference.json').read_text(encoding='utf-8'))
    audit=json.loads((root/'audio-encoder-validation.json').read_text(encoding='utf-8'))
    metadata=reference['metadata']
    if (metadata!=audit['metadata'] or metadata['source_commit']!=PRODUCER
            or metadata['model_revision']!=REVISION or metadata['feature_atol']!=1e-5
            or metadata['feature_rtol']!=1e-4 or audit['native_reference_complete'] is not True
            or metadata['versions']['torch']!=str(torch.__version__).split('+')[0]):
        raise ValueError('Original model, tolerance or Torch context differs')
    return reference,audit


def _candidate_metadata(directory,original,*,case_id):
    contract=candidate_contract();reference,audit=original_reference(original)
    root=Path(directory).resolve()
    for name,expected in [('audio-encoder-reference.json',contract['reference_sha256']),
                          ('audio-encoder-validation.json',contract['audit_sha256'])]:
        if _digest(root/name)!=expected:
            raise ValueError('Candidate reference/audit bytes differ')
    current=json.loads((root/'audio-encoder-validation.json').read_text(encoding='utf-8'))
    fresh=json.loads((root/'audio-encoder-reference.json').read_text(encoding='utf-8'))
    metadata=current['metadata']
    if (metadata!=fresh['metadata'] or metadata['source_commit']!=contract['producer_source']
            or current['native_reference_complete'] is not True
            or any(metadata[k]!=reference['metadata'][k]
                   for k in ('model_revision','versions','input_sha256','feature_atol','feature_rtol'))):
        raise ValueError('Candidate source, official weights or numerical context differs')
    rows=[row for row in reference['cases'] if row['id']==case_id]
    fresh_rows=[row for row in fresh['cases'] if row['id']==case_id]
    if len(rows)!=1 or len(fresh_rows)!=1 or fresh_rows[0]['profile']!=rows[0]['profile']:
        raise ValueError('One fixed original case and profile is required')
    row=rows[0];spec=reference['profiles'][row['profile']];new_spec=fresh['profiles'][row['profile']]
    if any(spec[k]!=new_spec[k] for k in ('native_graph','feature_shape','feature_count','feature_width')):
        raise ValueError('Candidate graph profile differs from the original')
    graph_name=spec['native_graph'];input_name=row['input_features']
    graph=contract['graphs'].get(graph_name)
    if (graph is None or current['files'].get(graph_name)!=graph
            or current['files'].get(input_name)!=audit['files'][input_name]):
        raise ValueError('Candidate graph or original input descriptor differs')
    raw=checked_file(root,input_name,audit['files'][input_name])
    provenance=dict(reference_source=PRODUCER,model_revision=REVISION,
        case_id=case_id,profile=row['profile'],reference_sha256=REFERENCE_SHA,audit_sha256=AUDIT_SHA,
        graph_sha256=graph['sha256'],input_sha256=audit['files'][input_name]['sha256'],
        model_generation=contract['generation'],candidate_contract_sha256=CONTRACT_SHA,
        graph_producer=contract['producer_source'],original_graph_bytes=False)
    return root,raw,spec,row,graph,provenance


def _candidate_input_files(directory,original,*,case_id):
    root,raw,spec,row,graph,provenance=_candidate_metadata(directory,original,case_id=case_id)
    program_path=checked_file(root,spec['native_graph'],graph)
    return program_path,raw,spec,row,provenance


def load_candidate_input(directory,original,*,case_id):
    program_path,raw,spec,row,provenance=_candidate_input_files(directory,original,case_id=case_id)
    values=torch.from_numpy(np.fromfile(raw,dtype='<f4').reshape(spec['feature_shape']).copy())
    expected=torch.tensor(row['features'],dtype=torch.float32)
    program=torch.export.load(program_path).module()
    if sorted(node.args[1][0] for node in checked_normalizations(program))!=[32,128]:
        raise ValueError('Candidate graph requires the two original normalization widths')
    return program,values,expected,provenance


def audit_candidate_report(path,original):
    return _audit_candidate_report(path,original)


def _audit_candidate_report(path,original,*,execution_graph=None):
    from .audit_audio_raw_chain import _audit_report
    contract=candidate_contract()
    options={} if execution_graph is None else {'execution_graph':execution_graph}
    result=_audit_report(path,original,expected_variant='ordered_original_midpoint_bmm',generation=contract,**options)
    path=Path(path).resolve();report=json.loads(path.read_text(encoding='utf-8'))
    controls=report.get('native_control_context',{})
    if (controls.get('mkl_cbwr')!='COMPATIBLE' or controls.get('mkl_dynamic')!='FALSE'
            or type(controls.get('threads')) is not int or controls['threads']!=4):
        raise ValueError('Candidate native execution controls differ')
    native=path.parent/'native-output-fp32.bin';control=path.parent/'native-chain-output-fp32.bin'
    checked_file(path.parent,control.name,{'bytes':native.stat().st_size,'sha256':report['native_chain_control_sha256']})
    if not np.array_equal(np.fromfile(native,dtype='<u4'),np.fromfile(control,dtype='<u4')):
        raise ValueError('Candidate native full/cut control bytes differ')
    result.update(model_generation=contract['generation'],candidate_contract_sha256=CONTRACT_SHA,
                  original_graph_bytes=False,native_chain_control_independently_verified=True,generation_qualified=False)
    return result


def audit_candidate_reports(paths,original,*,source):
    return _aggregate_candidate_reports(paths,original,source=source,audit_one=audit_candidate_report)


def _aggregate_candidate_reports(paths,original,*,source,audit_one):
    from .audit_audio_raw_chain import _aggregate_reports
    if re.fullmatch('[0-9a-f]{40}',source or '') is None:
        raise ValueError('Immutable candidate execution source required')
    contract=candidate_contract()
    rows=[audit_one(path,original) for path in paths]
    if any(row['model_generation']!=contract['generation'] or row['candidate_contract_sha256']!=CONTRACT_SHA for row in rows):
        raise ValueError('All candidate reports must use the same fixed generation')
    result=_aggregate_reports(rows,'ordered_original_midpoint_bmm')
    if result['source']!=source or result['values_compared']!=824320:
        raise ValueError('The complete original matrix at this execution source is required')
    result.update(scope='unqualified_candidate_all_six_original_raw_input_connected_onnx_numeric_capture_audit_only',
        model_generation=contract['generation'],candidate_contract_sha256=CONTRACT_SHA,
        original_graph_bytes=False,generation_qualified=False)
    return result


def preserved_original_reader():
    from .artifact_zip import github_artifact_reader
    reader, metadata = github_artifact_reader('ayutaz/unity-embeddinggemma-2', ORIGINAL_INPUT_ARTIFACT)
    run = metadata.get('workflow_run', {})
    if (metadata.get('id') != ORIGINAL_INPUT_ARTIFACT or metadata.get('digest') != 'sha256:'+ORIGINAL_INPUT_ARCHIVE_SHA
            or run.get('id') != ORIGINAL_INPUT_RUN or run.get('head_sha') != ORIGINAL_INPUT_SOURCE
            or type(run.get('repository_id')) is not int or run['repository_id'] <= 0
            or run.get('head_repository_id') != run['repository_id']):
        reader.close()
        raise ValueError('Preserved original artifact source or digest differs')
    return reader, metadata


def fetch_original(output):
    from .artifact_zip import extract_members
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Original download output must be empty')
    reader,metadata=preserved_original_reader()
    with reader:
        report=extract_members(reader,output,['audio-encoder-reference.json','audio-encoder-validation.json'],max_entry_bytes=32*1024*1024)
    original_reference(output)
    report.update(artifact_id=metadata['id'],preservation_run=ORIGINAL_INPUT_RUN,preservation_source=ORIGINAL_INPUT_SOURCE)
    (output/'original-range-receipt.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


def fetch_candidate(output,original,*,case_id):
    from .artifact_zip import github_artifact_reader,extract_members
    if case_id not in ('silence','impulse','sine','nyquist','noise','padded_noise'):
        raise ValueError('One of the six original input cases is required')
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Candidate download output must be empty')
    contract=candidate_contract();fetch_original(original)
    profile='frames-1119-valid-'+('839' if case_id=='padded_noise' else '1119')
    members=['audio-runtime-reference/'+name for name in ('audio-encoder-reference.json','audio-encoder-validation.json',
             profile+'/audio.pt2',profile+'/'+case_id+'-features-fp32.bin')]
    reader,metadata=github_artifact_reader('ayutaz/unity-embeddinggemma-2',contract['artifact_id'])
    with reader:
        run=metadata.get('workflow_run',{})
        if (metadata.get('id')!=contract['artifact_id'] or metadata.get('digest')!='sha256:'+contract['artifact_sha256']
                or run.get('id')!=contract['producer_run'] or run.get('head_sha')!=contract['producer_source']
                or type(run.get('repository_id')) is not int or run['repository_id']<=0
                or run.get('head_repository_id')!=run['repository_id']):
            raise ValueError('Candidate artifact source, run or archive digest differs')
        report=extract_members(reader,output,members,max_entry_bytes=2*1024*1024*1024)
    _candidate_input_files(output/'audio-runtime-reference',original,case_id=case_id)
    report.update(model_generation=contract['generation'],candidate_contract_sha256=CONTRACT_SHA,
                  artifact_id=contract['artifact_id'],graph_producer=contract['producer_source'],generation_qualified=False)
    (output/'candidate-range-receipt.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    fetch=commands.add_parser('fetch');fetch.add_argument('--output',type=Path,required=True)
    fetch.add_argument('--original',type=Path,required=True)
    fetch.add_argument('--case',required=True,choices=('silence','impulse','sine','nyquist','noise','padded_noise'))
    golden=commands.add_parser('fetch-original');golden.add_argument('--output',type=Path,required=True)
    audit=commands.add_parser('audit');audit.add_argument('--reports',type=Path,required=True)
    audit.add_argument('--original',type=Path,required=True);audit.add_argument('--source',required=True)
    audit.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='fetch-original':
        fetch_original(args.output);result={'success':True,'scope':'original_golden_only','generation_qualified':False}
    elif args.command=='fetch':
        report=fetch_candidate(args.output,args.original,case_id=args.case)
        result={'success':True,'scope':'fixed_candidate_input_bytes_only','model_generation':report['model_generation'],'generation_qualified':False}
    else:
        if args.output.exists():raise ValueError('Never overwrite a candidate audit result')
        report=audit_candidate_reports(sorted(args.reports.rglob('frozen-audio-chained-onnx.json')),args.original,source=args.source)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        result={k:report[k] for k in ('success','values_compared','candidate_tolerance_failures','generation_qualified','gpu_validated')}
    print(json.dumps(result))
    return 0 if result['success'] else 1


if __name__=='__main__':raise SystemExit(main())
