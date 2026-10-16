"""Continue one pinned interrupted raw prediction, with prefix graph identity.

Only the actual noise carry from run38825780838 is accepted. Earlier graphs
are re-exported from original coefficients and SHA-compared, never re-evaluated
or replaced with stage goldens. This is not a default export or GPU gate.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch
from onnx.reference import ReferenceEvaluator

from .audio_native_norm_replay import REFERENCE_SHA, AUDIT_SHA, PRODUCER, REVISION, load_frozen_input
from .audit_audio_raw_chain import _metrics, _cosine
from .frozen_audio_chained_onnx import _replacement_counts
from .frozen_audio_prefix_onnx import PrefixExport, extract_prefix, file_digest, weight_digests
from .frozen_audio_segment_onnx import PRECISE_MIDPOINT, SegmentExport, extract_segment
from .export import export_audio_native_onnx
from .streaming_reference import run_flat_reference

PREDECESSOR_SOURCE = 'e99c44214ab7d7f526b0fede5b561862460f1d66'
PREDECESSOR_REPORT_SHA = '55c3b38843988b4f5695d75bccc79ad253d43500c00a60200b0bf5d11f358a71'
MATH_MANIFEST_SHA = '3d52d40b9786336aaed8dbc888f4f8357b408981176be4f71d6159203e7490a7'
BOUNDARIES = ['raw_features','view_2','mul_77','mul_152','mul_227','select']
INTERMEDIATE_SHAPE = (1,280,1024)


def text_digest(path):
    return hashlib.sha256(Path(path).read_text(encoding='utf-8').encode('utf-8')).hexdigest()


def verify_math_source():
    manifest = Path(__file__).with_name('raw_resume_source_e99.json')
    if text_digest(manifest) != MATH_MANIFEST_SHA:
        raise ValueError('Pinned math source manifest differs')
    data = json.loads(manifest.read_text(encoding='utf-8'))
    root = Path(__file__).resolve().parents[2]
    if data['source'] != PREDECESSOR_SOURCE:
        raise ValueError('Pinned math source differs')
    for name, wanted in data['files'].items():
        p = (root/name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or text_digest(p) != wanted:
            raise ValueError(f'Math source or dependency lock differs: {name}')
    return MATH_MANIFEST_SHA


def original_case(root):
    root = Path(root)
    for name, wanted in [('audio-encoder-reference.json',REFERENCE_SHA),('audio-encoder-validation.json',AUDIT_SHA)]:
        if file_digest(root/name) != wanted:
            raise ValueError('Pinned original reference/audit differs')
    ref = json.loads((root/'audio-encoder-reference.json').read_text(encoding='utf-8'))
    audit = json.loads((root/'audio-encoder-validation.json').read_text(encoding='utf-8'))
    m = ref['metadata']
    if (m != audit['metadata'] or m['source_commit'] != PRODUCER or m['model_revision'] != REVISION
            or m['feature_atol'] != 1e-5 or m['feature_rtol'] != 1e-4 or not audit['native_reference_complete']):
        raise ValueError('Pinned original context differs')
    rows = [r for r in ref['cases'] if r['id']=='noise']
    if len(rows)!=1 or rows[0]['profile']!='frames-1119-valid-1119':
        raise ValueError('Pinned noise/profile differs')
    row=rows[0];spec=ref['profiles'][row['profile']];f=root/row['input_features']
    raw=_array(root,row['input_features'],audit['files'][row['input_features']]['sha256'],spec['feature_shape'])
    expected=np.asarray(row['features'],dtype=np.float32)
    if expected.shape != (spec['feature_count'],spec['feature_width']):
        raise ValueError('Pinned original final shape differs')
    return raw,expected,spec


def _array(root,name,wanted,shape):
    root=Path(root).resolve();f=(root/name).resolve()
    if (not f.is_relative_to(root) or not f.is_file() or f.stat().st_size!=int(np.prod(shape))*4
            or file_digest(f)!=wanted):
        raise ValueError('Prediction capture path/size/hash differs')
    a=np.fromfile(f,dtype='<f4').reshape(shape)
    if not np.isfinite(a).all():
        raise ValueError('Prediction capture is not finite')
    return a


def load_checkpoint(directory,original):
    root=Path(directory).resolve();f=root/'frozen-audio-chained-onnx.json'
    if file_digest(f)!=PREDECESSOR_REPORT_SHA:
        raise ValueError('Pinned predecessor report differs')
    d=json.loads(f.read_text(encoding='utf-8'))
    if (d.get('scope')!='raw_original_feature_input_connected_onnx_partitions_only' or d.get('complete') is not False
            or not all(d.get(k) is True for k in ('raw_input_mode','raw_original_input_executed','native_chain_control_bits_equal'))
            or d['provenance']['job_source']!=PREDECESSOR_SOURCE or d['provenance']['case_id']!='noise'
            or d['provenance']['profile']!='frames-1119-valid-1119' or d['boundaries']!=BOUNDARIES
            or d['phase']['name']!='evaluating' or d['phase']['part']!=3 or d['phase']['variant']!=PRECISE_MIDPOINT
            or any(d.get(k) is not False for k in ('whole_encoder_validated','gpu_validated','end_to_end_validated'))):
        raise ValueError('Pinned interrupted raw execution differs')
    raw,expected,spec=original_case(original);final_shape=expected.shape
    saved_raw=_array(root,'raw-input-fp32.bin',d['input_sha256'],raw.shape)
    saved_expected=_array(root,'reference-output-fp32.bin',d['reference_sha256'],final_shape)
    native=_array(root,'native-output-fp32.bin',d['native_sha256'],final_shape)
    if (not np.array_equal(raw.view('u4'),saved_raw.view('u4'))
            or not np.array_equal(expected.view('u4'),saved_expected.view('u4'))):
        raise ValueError('Original input or reference was substituted')
    if len(d['cases'])!=2 or [r['variant'] for r in d['cases']]!=['native',PRECISE_MIDPOINT]:
        raise ValueError('Original baseline and exact candidate required')
    native_examples=[]
    for variant,row,count in zip(('native',PRECISE_MIDPOINT),d['cases'],(5,3),strict=True):
        if len(row['parts'])!=count or row['actual_carry']!='previous_onnx_output_without_frozen_stage_reset':
            raise ValueError('Exact original prefix length and carry required')
        last=d['input_sha256']
        for i,part in enumerate(row['parts']):
            if (part['index']!=i or part['from_boundary']!=BOUNDARIES[i] or part['to_boundary']!=BOUNDARIES[i+1]
                    or part['input_sha256']!=last or re.fullmatch('[0-9a-f]{64}',part['graph_sha256']) is None):
                raise ValueError('Original prefix boundary/carry/graph differs')
            inp=_array(root,part['input_capture'],part['input_sha256'],raw.shape if i==0 else INTERMEDIATE_SHAPE)
            actual=_array(root,part['output_capture'],part['output_sha256'],final_shape if i==4 else INTERMEDIATE_SHAPE)
            if variant=='native': native_examples.append((inp,part['native_input_strides']))
            last=part['output_sha256']
    return dict(root=root,report=d,expected=expected,native=native,spec=spec,
        native_examples=native_examples,carry=actual,report_sha256=PREDECESSOR_REPORT_SHA)


def wrap_piece(piece,index):
    return PrefixExport(piece,'ordered_original_stem') if index==0 else SegmentExport(piece,PRECISE_MIDPOINT)


def validate_suffix_counts(counts,index):
    wanted=dict(linear_replacements=32 if index==4 else 30,attention_bmm_replacements=9,
        parameter_bmm_replacements=3,relative_position_replacements=3,tanh_replacements=3,
        midpoint_attention_bmm_replacements=3,rms_mean_replacements=28 if index==4 else 27,
        rms_inverse_replacements=28 if index==4 else 27,silu_replacements=9,
        softmax_replacements=3,sigmoid_replacements=3,depthwise_replacements=3)
    if any(counts[k]!=v for k,v in wanted.items()):
        raise ValueError('Remaining original three-layer coverage differs')


def _example(array,strides):
    if (len(strides)!=array.ndim or any(type(s) is not int or s<0 for s in strides)
            or 1+sum((d-1)*s for d,s in zip(array.shape,strides))!=array.size):
        raise ValueError('Original dense native layout differs')
    x=torch.empty_strided(tuple(array.shape),tuple(strides),dtype=torch.float32)
    return x.copy_(torch.from_numpy(array))


def resume(program,checkpoint,output,*,job_source):
    if re.fullmatch('[0-9a-f]{40}',job_source) is None or job_source==PREDECESSOR_SOURCE:
        raise ValueError('Immutable resume job source required')
    math_sha=verify_math_source();out=Path(output);receipt=out/'audio-raw-resume.json'
    if receipt.exists() or any(out.glob('*.bin')) or any(out.glob('*.onnx')):
        raise ValueError('Refusing to overwrite continuation evidence')
    before=weight_digests(program);d=checkpoint['report']
    if before!=d['original_weights']:
        raise ValueError('Original learned coefficients differ')
    cuts=d['boundaries'][1:]
    pieces=[extract_prefix(program,cuts[0]),*[extract_segment(program,a,b) for a,b in zip(cuts,cuts[1:])]]
    out.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    r=dict(scope='pinned_noise_raw_actual_onnx_carry_continuation_only',complete=False,
        job_source=job_source,predecessor_source=PREDECESSOR_SOURCE,predecessor_report_sha256=checkpoint['report_sha256'],
        math_source_manifest_sha256=math_sha,verified_prefix_graphs=[],parts=[],
        original_tolerance_passed=False,coefficients_unchanged=False,gpu_validated=False,default_export_promoted=False)
    def save():receipt.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    def phase(name,index):
        r['phase']=dict(name=name,part=index,elapsed_seconds=time.perf_counter()-started);save()
        print(json.dumps(r['phase']),flush=True)
    def capture(name,array):
        array.astype('<f4').tofile(out/name);return file_digest(out/name)
    actual=checkpoint['carry'];save()
    try:
        for i,piece in enumerate(pieces):
            wrapper=wrap_piece(piece,i);counts=_replacement_counts(wrapper,is_stem=i==0)
            phase('verifying_prefix_graph' if i<3 else 'exporting',i)
            graph=out/f'resumed-part{i}.onnx'
            with torch.no_grad():export_audio_native_onnx(wrapper,_example(*checkpoint['native_examples'][i]),graph)
            graph_sha=file_digest(graph)
            if i<3:
                previous=d['cases'][1]['parts'][i]
                if graph_sha!=previous['graph_sha256'] or any(previous[k]!=v for k,v in counts.items()):
                    raise ValueError('Regenerated original prefix graph differs')
                r['verified_prefix_graphs'].append(dict(index=i,sha256=graph_sha,execution_source=PREDECESSOR_SOURCE))
                save()
            else:
                validate_suffix_counts(counts,i)
                input_name=f'part{i}-input-fp32.bin';output_name=f'part{i}-output-fp32.bin'
                inp_sha=capture(input_name,actual);phase('evaluating',i)
                evaluator=ReferenceEvaluator(str(graph));actual=run_flat_reference(evaluator,{'input':actual})[0]
                wanted=checkpoint['expected'].shape if i==4 else INTERMEDIATE_SHAPE
                if actual.dtype!=np.float32 or actual.shape!=wanted or not np.isfinite(actual).all():
                    raise ValueError('Continuation output shape/dtype/finite contract differs')
                r['parts'].append(dict(index=i,from_boundary=d['boundaries'][i],to_boundary=d['boundaries'][i+1],
                    graph_sha256=graph_sha,input_capture=input_name,output_capture=output_name,input_sha256=inp_sha,
                    output_sha256=capture(output_name,actual),execution_source=job_source,
                    onnx_operator_counts=dict(Counter(n.op_type for n in evaluator.proto_.graph.node)),**counts))
                save();del evaluator
            del wrapper;gc.collect()
        if weight_digests(program)!=before:
            raise ValueError('Original coefficients changed while regenerating/continuing')
        r['coefficients_unchanged']=True;r['metrics']=_metrics(actual,checkpoint['expected'])
        r['minimum_cosine']=_cosine(actual,checkpoint['expected'])
        r['vs_predecessor_native']=_metrics(actual,checkpoint['native'])
        r['original_tolerance_passed']=r['metrics']['mismatched_elements']==0 and r['minimum_cosine']>=0.999999
        r['complete']=True;phase('complete',4)
    except Exception as error:
        r['diagnostic_error']=dict(type=type(error).__name__,message=str(error));save();raise
    return r


def audit_resume(report_path,predecessor,original):
    checkpoint=load_checkpoint(predecessor,original);previous=checkpoint['report']
    report_path=Path(report_path).resolve();root=report_path.parent
    r=json.loads(report_path.read_text(encoding='utf-8'))
    if (r.get('scope')!='pinned_noise_raw_actual_onnx_carry_continuation_only'
            or not all(r.get(k) is True for k in ('complete','coefficients_unchanged'))
            or any(r.get(k) is not False for k in ('gpu_validated','default_export_promoted'))
            or r['predecessor_source']!=PREDECESSOR_SOURCE
            or r['predecessor_report_sha256']!=PREDECESSOR_REPORT_SHA
            or r['math_source_manifest_sha256']!=verify_math_source()
            or re.fullmatch('[0-9a-f]{40}',r['job_source']) is None
            or r['job_source']==PREDECESSOR_SOURCE
            or len(r['verified_prefix_graphs'])!=3 or len(r['parts'])!=2):
        raise ValueError('Completed pinned continuation provenance differs')
    for i,p in enumerate(r['verified_prefix_graphs']):
        if (p['index']!=i or p['sha256']!=previous['cases'][1]['parts'][i]['graph_sha256']
                or p['execution_source']!=PREDECESSOR_SOURCE):
            raise ValueError('Regenerated prefix graph evidence differs')
    last=previous['cases'][1]['parts'][-1]['output_sha256']
    for i,p in zip((3,4),r['parts'],strict=True):
        if (p['index']!=i or p['from_boundary']!=previous['boundaries'][i]
                or p['to_boundary']!=previous['boundaries'][i+1] or p['input_sha256']!=last
                or p['execution_source']!=r['job_source']
                or re.fullmatch('[0-9a-f]{64}',p['graph_sha256']) is None):
            raise ValueError('Continued actual prediction carry/source differs')
        validate_suffix_counts(p,i)
        _array(root,p['input_capture'],p['input_sha256'],INTERMEDIATE_SHAPE)
        actual=_array(root,p['output_capture'],p['output_sha256'],checkpoint['expected'].shape if i==4 else INTERMEDIATE_SHAPE)
        last=p['output_sha256']
    measured=_metrics(actual,checkpoint['expected']);cosine=_cosine(actual,checkpoint['expected'])
    native_metrics=_metrics(actual,checkpoint['native'])
    passed=measured['mismatched_elements']==0 and cosine>=0.999999
    if (r['metrics']!=measured or r['minimum_cosine']!=cosine
            or r['vs_predecessor_native']!=native_metrics or r['original_tolerance_passed']!=passed):
        raise ValueError('Independent resumed final metrics differ')
    return dict(scope='independently_audited_raw_noise_prediction_resume_only',case_id='noise',
        profile='frames-1119-valid-1119',original_tolerance_passed=passed,values_compared=actual.size,
        predecessor_source=PREDECESSOR_SOURCE,job_source=r['job_source'],math_source_manifest_sha256=verify_math_source(),
        execution_sources=sorted({PREDECESSOR_SOURCE,r['job_source']}),report_sha256=file_digest(report_path),
        predecessor_report_sha256=PREDECESSOR_REPORT_SHA,final_capture_sha256=last,
        candidate_metrics=measured,minimum_cosine=cosine,candidate_vs_predecessor_native=native_metrics,
        predecessor_native_context_qualified=bool(previous.get('context_qualified',False)),
        all_capture_and_carry_hashes_verified=True,prefix_graphs_reexported_and_bit_identity_verified=True,
        graphs_independently_rehashed=False,gpu_validated=False,default_export_promoted=False)


def audit_six_with_resume(completed_paths,resume_report,predecessor,original):
    from .audit_audio_raw_chain import audit_report
    rows=[audit_report(p,original,expected_variant=PRECISE_MIDPOINT) for p in completed_paths]
    if (len(rows)!=5 or {r['case_id'] for r in rows}!={'silence','impulse','sine','nyquist','padded_noise'}
            or any(r['source']!=PREDECESSOR_SOURCE for r in rows)):
        raise ValueError('Exactly five completed original cases at predecessor source required')
    for row in rows:
        padded=row['case_id']=='padded_noise'
        if (row['profile']!=('frames-1119-valid-839' if padded else 'frames-1119-valid-1119')
                or row['values_compared']!=(107520 if padded else 143360)):
            raise ValueError('Original case/profile/value coverage differs')
    last=audit_resume(resume_report,predecessor,original)
    if (last['case_id']!='noise' or last['profile']!='frames-1119-valid-1119'
            or last['predecessor_source']!=PREDECESSOR_SOURCE or last['values_compared']!=143360):
        raise ValueError('Original resumed noise/source/value coverage differs')
    native_qualified=all(r['native_context_qualified'] for r in rows) and last['predecessor_native_context_qualified']
    rows.append(last)
    return dict(scope='all_six_original_raw_input_candidate_with_audited_actual_prediction_resume',
        success=all(r['original_tolerance_passed'] for r in rows),case_count=6,profile_count=2,
        math_source=PREDECESSOR_SOURCE,math_source_manifest_sha256=verify_math_source(),
        execution_sources=last['execution_sources'],all_six_at_one_job_source=False,
        values_compared=824320,candidate_tolerance_failures=sum(r['candidate_metrics']['mismatched_elements'] for r in rows),
        candidate_bit_differences=sum(r['candidate_metrics']['exact_difference_count'] for r in rows),cases=rows,
        native_all_six_context_qualified=native_qualified,original_reference_unchanged=True,
        intermediate_stage_reference_used=False,gpu_validated=False,default_export_promoted=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-source',action='store_true');p.add_argument('--input',type=Path)
    p.add_argument('--predecessor',type=Path);p.add_argument('--output',type=Path);a=p.parse_args(argv)
    verify_math_source()
    if a.verify_source:return 0
    if not all((a.input,a.predecessor,a.output)):p.error('--input, --predecessor and --output are required')
    c=load_checkpoint(a.predecessor,a.input)
    program,_,_,_=load_frozen_input(a.input,case_id='noise')
    r=resume(program,c,a.output,job_source=os.environ.get('GITHUB_SHA',''))
    print(json.dumps(dict(complete=r['complete'],original_tolerance_passed=r['original_tolerance_passed'],metrics=r.get('metrics'))))
    return 0 if r['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
