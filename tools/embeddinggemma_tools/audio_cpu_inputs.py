"""Materialize byte-identical procedural audio benchmark inputs, not models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import torch

from . import audio_cpu_cohort as refs
from . import native_regression_distribution as transport


REFERENCE_DIR=Path(__file__).parents[1]/'references/audio-cpu-original'


def _require(condition):
    if not condition:
        raise ValueError('Original procedural audio input contract differs')


def _validate_execution_source(source):
    root=Path(__file__).resolve().parents[2]
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,stderr=subprocess.PIPE).decode().strip()
    _require(actual==source)
    paths=['tools/embeddinggemma_tools/'+name for name in ('audio_cpu_inputs.py','audio_cpu_cohort.py','native_regression_distribution.py')]
    paths+=['tools/references/audio-cpu-original/'+name for name in ('audio-reference.json','audio-validation.json','audio-embedding-reference.json')]
    paths+=['tools/uv.lock','tools/pyproject.toml']
    result={}
    for name in paths:
        raw=(root/name).read_bytes()
        expected=subprocess.check_output(['git','show',source+':'+name],cwd=root,stderr=subprocess.PIPE)
        if name.endswith('.py') or name in ('tools/uv.lock','tools/pyproject.toml'):
            raw,expected=raw.replace(b'\r\n',b'\n'),expected.replace(b'\r\n',b'\n')
        _require(raw==expected)
        result[name]=hashlib.sha256(raw).hexdigest()
    return result


def _procedural_inputs():
    # Preserve the original operation order and signed zero bits in padding.
    noise=torch.rand((1,179200),generator=torch.Generator(device='cpu').manual_seed(17),
        dtype=torch.float32,device='cpu')*1.6-.8
    result={}
    for case_id in refs.CPU_CASES:
        valid,frames=(134400,839) if case_id=='padded_noise' else (179200,1119)
        sample_mask=(torch.arange(179200,device='cpu')[None]<valid).long()
        waveform=(noise.clone() if case_id=='padded_noise' else torch.zeros((1,179200),dtype=torch.float32,device='cpu'))*sample_mask
        frame_mask=(torch.arange(1119,device='cpu')[None]<frames).int()
        result[case_id]=dict(waveform=waveform.numpy().astype('<f4',copy=False).tobytes(),
            sample_mask=sample_mask.numpy().astype('<i4',copy=False).tobytes(),
            frame_mask=frame_mask.numpy().astype('<i4',copy=False).tobytes())
    return result


def _complete_original_inputs(original,encoder,audit,*,fetch):
    rows=refs._rows(encoder)
    missing={}
    for case_id in refs.CPU_CASES:
        name=rows[case_id]['input_features']
        refs._relative(name)
        target=original/name
        if target.exists():
            refs._file(original,name,audit['files'])
        else:
            missing['input-'+case_id+'.bin']=name
    if not missing:return 0
    _require(fetch is True)
    config,inventory=transport.configuration()
    for asset,name in missing.items():
        spec=audit['files'][name]
        _require(inventory['files'][asset]['original_member']==name)
        _require(all(config['assets'][asset][key]==spec[key] for key in ('bytes','sha256')))
    transport._fetch(original,missing,config,inventory)
    for name in missing.values():refs._file(original,name,audit['files'])
    return len(missing)


def materialize_audio_cpu_inputs(encoder_reference,output,*,source,fetch_missing_inputs=False):
    """Replay only two fixed input cases and verify full original byte hashes."""
    try:
        _require(isinstance(source,str) and re.fullmatch('[0-9a-f]{40}',source) is not None)
        output=transport._empty(output)
        fingerprints=_validate_execution_source(source)
        original=Path(encoder_reference)
        pre,pre_raw=refs._json(REFERENCE_DIR/'audio-reference.json',refs.PREPROCESS_REFERENCE_SHA256)
        audit,audit_raw=refs._json(REFERENCE_DIR/'audio-validation.json',refs.PREPROCESS_AUDIT_SHA256)
        document,doc_raw=refs._json(REFERENCE_DIR/'audio-embedding-reference.json',refs.DOCUMENT_REFERENCE_SHA256)
        encoder,_=refs._json(original/'audio-encoder-reference.json',refs.ENCODER_REFERENCE_SHA256)
        enc_audit,_=refs._json(original/'audio-encoder-validation.json',refs.ENCODER_AUDIT_SHA256)
        _require(audit['success'] is True and audit['metadata']==pre['metadata'])
        _require(enc_audit['native_reference_complete'] is True and enc_audit['metadata']==encoder['metadata'])
        _require(pre['metadata']['model_revision']==encoder['metadata']['model_revision']==document['metadata']['model_revision']==refs.MODEL_REVISION)
        _require(pre['metadata']['fixture_provenance']=='procedurally_generated_pcm_no_user_audio')
        _require(pre['metadata']['feature_atol']==.004 and pre['metadata']['feature_rtol']==1e-4)
        _require(str(torch.__version__).split('+')[0]==pre['metadata']['versions']['torch'])
        _require(encoder['metadata']['preprocessing_reference_sha256']==refs.PREPROCESS_REFERENCE_SHA256)
        _require(document['metadata']['encoder_reference_sha256']==refs.ENCODER_REFERENCE_SHA256)
        pre_rows,enc_rows=refs._rows(pre),refs._rows(encoder)
        _require(all(row['role']=='document' for row in refs._rows(document).values()))
        fetched=_complete_original_inputs(original,encoder,enc_audit,fetch=fetch_missing_inputs)
        generated=_procedural_inputs()
        payloads={}
        origins={}
        for case_id in refs.CPU_CASES:
            row=pre_rows[case_id]
            for field,raw in generated[case_id].items():
                _require(field in ('waveform','sample_mask','frame_mask'))
                name=refs._relative(row[field])
                spec=audit['files'][name]
                _require(len(raw)==spec['bytes'] and hashlib.sha256(raw).hexdigest()==spec['sha256'])
                _require(name not in payloads)
                payloads[name]=raw
                origins[name]='procedural-input-replay-byte-identical-to-original'
            name=refs._relative(row['features'])
            raw=refs._file(original,enc_rows[case_id]['input_features'],enc_audit['files'])
            spec=audit['files'][name]
            _require(name not in payloads and len(raw)==spec['bytes'] and hashlib.sha256(raw).hexdigest()==spec['sha256'])
            payloads[name]=raw
            origins[name]='verbatim-original-reference-features-not-inference-input'
        for name,raw in [('audio-reference.json',pre_raw),('audio-validation.json',audit_raw),('audio-embedding-reference.json',doc_raw)]:
            _require(name not in payloads)
            payloads[name]=raw
            origins[name]='verbatim-original-reference-JSON'
        output.mkdir(parents=True,exist_ok=True)
        for name,raw in payloads.items():
            target=output/name
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(raw)
        cohort=refs.load_audio_cpu_reference_cohort(output,original,output/'audio-embedding-reference.json')
        _require(len(cohort.cases)==2)
        result=dict(schema=1,success=True,source=source,
            scope='Byte-identical procedural input reconstruction and unchanged reference buffers only',
            torch_runtime=str(torch.__version__),source_fingerprints=fingerprints,
            model_revision=refs.MODEL_REVISION,case_ids=list(refs.CPU_CASES),
            procedural_buffers_reconstructed=6,reference_feature_buffers_copied_verbatim=2,
            original_reference_inputs_fetched=fetched,
            files={name:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),origin=origins[name]) for name,raw in payloads.items()},
            original_reference_or_model_regenerated=False,model_executed=False,
            CPU_comparison_qualified=False,complete_performance_accepted=False,GPU_validated=False)
        with (output/'audio-cpu-inputs.json').open('x',encoding='utf-8') as stream:
            json.dump(result,stream,indent=2,allow_nan=False)
            stream.write('\n')
        return result
    except (OSError,ValueError,TypeError,KeyError,IndexError,subprocess.SubprocessError):
        raise ValueError('Original audio CPU input materialization failed; preserve existing files') from None


def main(argv=None):
    parser=argparse.ArgumentParser(prog='audio_cpu_inputs',description=__doc__,exit_on_error=False)
    parser.add_argument('--encoder-reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',default=os.environ.get('GITHUB_SHA'))
    parser.add_argument('--fetch-missing-inputs',action='store_true')
    try:
        args=parser.parse_args(argv)
        torch.set_num_threads(4)
        materialize_audio_cpu_inputs(args.encoder_reference,args.output,source=args.source,
            fetch_missing_inputs=args.fetch_missing_inputs)
    except Exception as error:
        print(json.dumps(dict(success=False,error_class=type(error).__name__,model_executed=False)))
        return 1
    print(json.dumps(dict(success=True,model_executed=False,CPU_comparison_qualified=False)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
