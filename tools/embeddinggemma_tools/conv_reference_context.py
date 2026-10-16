"""CPU diagnosis of a frozen convolution, never an accepted replacement golden."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import numpy as np
import torch
import torch.nn.functional as F


def _runtime():
    return dict(torch_version=torch.__version__,torch_threads=torch.get_num_threads(),
        cpu_capability=torch.backends.cpu.get_cpu_capability(),torch_config=torch.__config__.show(),platform=platform.platform(),
        env={name:os.environ.get(name) for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','ATEN_CPU_CAPABILITY','MKL_CBWR','DNNL_MAX_CPU_ISA','ONEDNN_VERBOSE']})


def replay_frozen_conv_context(*,encoder,subsample,encoder_reference_sha256,encoder_audit_sha256,
                              subsample_reference_sha256,weight_sha256):
    encoder,subsample=Path(encoder).resolve(),Path(subsample).resolve()
    def digest(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    def checked(path,wanted):
        if re.fullmatch('[a-f0-9]{64}',wanted or '') is None or digest(path)!=wanted:
            raise ValueError('Frozen convolution provenance hash mismatch')
        return path
    reference_path=checked(encoder/'audio-encoder-reference.json',encoder_reference_sha256)
    audit_path=checked(encoder/'audio-encoder-validation.json',encoder_audit_sha256)
    replay_path=checked(subsample/'audio-layer-replay-report.json',subsample_reference_sha256)
    reference=json.loads(reference_path.read_text());audit=json.loads(audit_path.read_text());replay=json.loads(replay_path.read_text())
    producer='4edbe17a8105233c5aaa1514faa2862b08a4f7e0'
    if (reference.get('metadata')!=audit.get('metadata') or reference['metadata'].get('source_commit')!=producer
            or not audit.get('native_reference_complete') or replay.get('reference_source')!=producer
            or not replay.get('success') or not replay.get('subsample_native_final_exact_match')):
        raise ValueError('Frozen native convolution reference is not complete or has another producer')
    def binary(name):
        part=replay['parts']['subsample.'+name];shape=part['shape']
        path=(subsample/'audio-layer-replay'/part['file']).resolve()
        if (not path.is_relative_to(subsample) or len(shape)!=4 or any(type(d) is not int or d<=0 for d in shape)
                or math.prod(shape)*4!=part['bytes'] or part['bytes']>32*1024*1024 or path.stat().st_size!=part['bytes']):
            raise ValueError('Frozen convolution tensor shape, path or size differs')
        checked(path,part['sha256']);value=torch.from_numpy(np.fromfile(path,'<f4').reshape(shape).copy())
        if not torch.isfinite(value).all():raise ValueError('Frozen convolution tensors must be finite')
        return value
    values=binary('layer0.act');expected=binary('layer1.conv')
    if values.shape[1]!=128 or tuple(expected.shape)!=(values.shape[0],32,(values.shape[2]+1)//2,(values.shape[3]+1)//2):
        raise ValueError('Frozen convolution geometry differs')
    graph_name=reference['profiles'][replay['profile']]['native_graph'];graph_path=(encoder/graph_name).resolve()
    spec=audit['files'][graph_name]
    if not graph_path.is_relative_to(encoder) or graph_path.stat().st_size!=spec['bytes']:
        raise ValueError('Saved native graph path or size differs')
    checked(graph_path,spec['sha256'])
    if re.fullmatch('[a-f0-9]{64}',weight_sha256 or '') is None:raise ValueError('Pin the original coefficient digest')
    program=torch.export.load(graph_path)
    matches=[]
    for name,value in program.state_dict.items():
        if (isinstance(value,torch.Tensor) and value.dtype==torch.float32 and value.device.type=='cpu'
                and tuple(value.shape)==(32,128,3,3)):
            if hashlib.sha256(value.detach().numpy().astype('<f4').tobytes()).hexdigest()==weight_sha256:
                matches.append((name,value.detach().clone()))
    if len(matches)!=1:raise ValueError('Original saved convolution coefficient is missing or ambiguous')
    weight_name,weight=matches[0];del program
    with torch.backends.mkldnn.verbose(torch.backends.mkldnn.VERBOSE_ON),torch.no_grad():
        actual=F.conv2d(values,weight,stride=2,padding=1)
    if not torch.isfinite(actual).all():raise ValueError('Native convolution produced nonfinite values')
    differences=(actual.double()-expected.double()).abs();allowed=1e-5+1e-4*expected.double().abs()
    exact=int(torch.count_nonzero(actual.view(torch.int32)!=expected.view(torch.int32)))
    return dict(success=exact==0,context_validated=exact==0,scope='frozen_original_conv2_cpu_context_only',
        expected_unchanged=True,reference_source=producer,model_revision=reference['metadata']['model_revision'],
        native_graph_sha256=spec['sha256'],weight_name=weight_name,weight_sha256=weight_sha256,
        input_shape=list(values.shape),output_shape=list(expected.shape),exact_differences=exact,
        mismatched_elements=int(torch.count_nonzero(differences>allowed)),maximum_absolute_error=float(differences.max()),
        maximum_tolerance_ratio=float((differences/allowed).max()),gpu_validated=False,
        full_encoder_validated=False,end_to_end_validated=False,
        runtime=_runtime())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['encoder','subsample','output']:parser.add_argument('--'+name,type=Path,required=True)
    for name in ['encoder_reference_sha256','encoder_audit_sha256','subsample_reference_sha256','weight_sha256']:
        parser.add_argument('--'+name.replace('_','-'),required=True)
    args=vars(parser.parse_args(argv));output=args.pop('output')
    if output.exists():raise ValueError('Refusing to overwrite a convolution context report')
    try:report=replay_frozen_conv_context(**args)
    except Exception as error:report=dict(success=False,context_validated=False,scope='frozen_original_conv2_cpu_context_only',
        error_type=type(error).__name__,error=str(error),gpu_validated=False,full_encoder_validated=False,end_to_end_validated=False)
    report['job_source']=os.environ.get('GITHUB_SHA')
    report['requested_reference']={name:str(value) for name,value in args.items()}
    report.setdefault('runtime',_runtime());output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(report))
    return 0 if report['success'] else 1


if __name__=='__main__':raise SystemExit(main())
