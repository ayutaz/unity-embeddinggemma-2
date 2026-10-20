"""Actual PCM-to-vector CPU comparison, separate from the Sentis GPU SDK."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import torch

from . import audio_cpu_cohort as references
from . import reference_cpu_performance as protocol
from .audio_cpu_encoder import AuditedAudioCpuEncoder, _qualify_retained_graph_producer
from .cpu_execution_context import cpu_execution_context
from .image_cpu_reference import (_capture, _encoder_result, _json, _peak_rss_bytes,
                                  _read_capture, _write)
from .native_regression_distribution import _empty


PREPROCESSING_IMPLEMENTATION='official_Gemma4AudioFeatureExtractor_from_actual_PCM'


def _require(condition):
    if not condition:
        raise ValueError('Audio CPU reference contract differs')


class OfficialAudioCpuPreprocessor(torch.nn.Module):
    """Recompute the pinned official CPU reference directly from actual PCM."""

    def __init__(self,extractor):
        super().__init__()
        from transformers import Gemma4AudioFeatureExtractor
        from .audio import AudioPreprocessorExport
        _require(isinstance(extractor,Gemma4AudioFeatureExtractor))
        self._input_guard=AudioPreprocessorExport(179200,extractor).eval()
        self._extractor=extractor

    def forward(self,waveform,sample_mask):
        _require(isinstance(waveform,torch.Tensor) and isinstance(sample_mask,torch.Tensor)
                 and waveform.device.type==sample_mask.device.type=='cpu')
        self._input_guard.validate_inputs(waveform,sample_mask)
        raw,frames=self._extractor._extract_spectrogram(
            waveform.detach().numpy()[0],sample_mask.detach().numpy()[0])
        _require(isinstance(raw,np.ndarray) and isinstance(frames,np.ndarray)
                 and raw.shape==(1119,128) and frames.shape==(1119,))
        valid_frames=(int(sample_mask.sum())-161)//160+1
        _require(bool(np.isfinite(raw).all()) and np.array_equal(frames,np.arange(1119)<valid_frames))
        # Preserve the official producer's FP32 cast, mask multiplication and signed zero bits.
        features=(raw.astype(np.float32)*frames[:,None])[None]
        return (torch.from_numpy(np.array(features,dtype=np.float32,copy=True,order='C')),
                torch.from_numpy(np.array(frames[None],dtype=np.int32,copy=True,order='C')))


def _validate_execution_source(source):
    root=Path(__file__).resolve().parents[2]
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,stderr=subprocess.PIPE).decode().strip()
    _require(source==actual)
    names=('audio_cpu_reference.py','audio_cpu_cohort.py','audio_cpu_encoder.py','onnx_cpu_comparison.py',
        'cpu_progress.py','cpu_execution_context.py','reference_cpu_performance.py','image_cpu_reference.py',
        'audio.py','model.py','export.py','text.py','native_regression_distribution.py',
        'audio_generation_candidate.py','audio_native_norm_replay.py','audit_audio_raw_chain.py')
    paths=['tools/embeddinggemma_tools/'+name for name in names]+['tools/uv.lock','tools/pyproject.toml']
    result={}
    for name in paths:
        raw=(root/name).read_bytes().replace(b'\r\n',b'\n')
        expected=subprocess.check_output(['git','show',source+':'+name],cwd=root,stderr=subprocess.PIPE).replace(b'\r\n',b'\n')
        _require(raw==expected)
        result[name]=hashlib.sha256(raw).hexdigest()
    return result


def _validate_snapshot(snapshot,cohort):
    for name,expected in cohort.snapshot_input_sha256:
        with (Path(snapshot)/name).open('rb') as stream:
            _require(hashlib.file_digest(stream,'sha256').hexdigest()==expected)


def _admit_chain(chain_report,encoder_directory,source,case,*,retained_producer_source=None):
    from .native_regression_distribution import audit_distribution_report
    audited=audit_distribution_report(chain_report,encoder_directory)
    producer=source if retained_producer_source is None else retained_producer_source
    _require(audited['original_tolerance_passed'] is True and audited['source']==producer
        and audited['case_id']==case.id and audited['profile']==case.profile)
    if retained_producer_source is not None:
        _qualify_retained_graph_producer(retained_producer_source,source)


def _download_snapshot(cohort):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download('google/embeddinggemma-2',revision=references.MODEL_REVISION,
        allow_patterns=[name for name,_ in cohort.snapshot_input_sha256]))


def _load_modules(snapshot,encoder_root,chain_report,source,case,final_metadata,final_profile,*,retained_producer_source=None):
    from transformers import Gemma4AudioFeatureExtractor
    from .export import MultimodalEmbeddingExport
    from .model import load_text_model
    settings=_json(Path(snapshot)/'processor_config.json')['feature_extractor']
    extractor=Gemma4AudioFeatureExtractor(**{name:value for name,value in settings.items()
        if name not in ('feature_extractor_type','frame_length','hop_length','fft_length')})
    preprocessor=OfficialAudioCpuPreprocessor(extractor).eval()
    options={} if retained_producer_source is None else dict(retained_producer_source=retained_producer_source)
    encoder=AuditedAudioCpuEncoder(chain_report,encoder_root,source=source,**options)
    try:
        _require(encoder.configuration['case_id']==case.id and encoder.configuration['profile']==case.profile)
        backbone=MultimodalEmbeddingExport(load_text_model(Path(snapshot)),final_profile['sequence_length'],
            image_token_id=final_metadata['image_token_id'],audio_token_id=final_metadata['audio_token_id'],
            image_feature_count=0,audio_feature_count=final_profile['audio_feature_count'])
        return preprocessor,encoder,backbone
    except Exception:
        encoder.close()
        raise


def _inspections(output,rows,case):
    results=[]
    for name,row in zip(protocol.ROUNDS,rows,strict=True):
        _require(row['round']==name and set(row['captures'])=={'features','frame_mask','encoder'})
        captures=row['captures']
        _,features=_read_capture(output,captures['features'],name+'-features.bin',(1,1119,128),'<f4')
        _,frame_mask=_read_capture(output,captures['frame_mask'],name+'-frame_mask.bin',(1,1119),'<i4')
        _,encoded=_read_capture(output,captures['encoder'],name+'-encoder.bin',case.encoder_features.shape,'<f4')
        delta=np.abs(features.astype(np.float64)-case.preprocessing_features.astype(np.float64))
        failures=int(np.count_nonzero(delta>.004+1e-4*np.abs(case.preprocessing_features.astype(np.float64))))
        encoder=_encoder_result(encoded,case.encoder_features)
        mask_exact=bool(np.array_equal(frame_mask,case.frame_mask))
        results.append(dict(round=name,preprocessing_tolerance_failures=failures,
            maximum_preprocessing_absolute_error=float(delta.max()),frame_mask_exact=mask_exact,
            encoder_tolerance_failures=encoder['tolerance_failures'],
            maximum_encoder_absolute_error=encoder['maximum_absolute_error'],
            minimum_encoder_cosine=encoder['minimum_cosine'],
            success=failures==encoder['tolerance_failures']==0 and mask_exact and encoder['minimum_cosine']>=.999999))
    return results


def measure_audio_cpu_reference(snapshot,preprocessing,encoder_directory,document_reference,
                                chain_report,output,*,source,case_id,retained_producer_source=None):
    """Measure actual stages; observers snapshot outputs before each consumer."""
    owner=None
    try:
        _require(isinstance(source,str) and re.fullmatch('[0-9a-f]{40}',source) is not None)
        _require(case_id in references.CPU_CASES and not torch.cuda.is_initialized())
        output=_empty(output)
        fingerprints=_validate_execution_source(source)
        cohort=references.load_audio_cpu_reference_cohort(preprocessing,encoder_directory,document_reference)
        case=cohort.case(case_id)
        options={} if retained_producer_source is None else dict(retained_producer_source=retained_producer_source)
        _admit_chain(chain_report,encoder_directory,source,case,**options)
        _validate_snapshot(snapshot,cohort)
        document=_json(document_reference,cohort.document_reference_sha256)
        before=cpu_execution_context()
        output.mkdir(parents=True,exist_ok=True)
        journal=dict(schema=1,source=source,case=case_id,profile=case.profile,phase='loading',
            preprocessing_implementation=PREPROCESSING_IMPLEMENTATION,
            model_execution_complete=False,error_class=None,model_revision=references.MODEL_REVISION,
            source_fingerprints_normalized_LF=fingerprints,
            snapshot_input_sha256=dict(cohort.snapshot_input_sha256),
            execution_context_before_load=before,CPU_comparison_qualified=False,complete_performance_accepted=False)
        if retained_producer_source is not None:
            journal['retained_chain_producer_source']=retained_producer_source
        source_path=output/'audio-cpu-source.json'
        _write(source_path,journal)
        def progress(**values):
            journal.update(values)
            source_path.write_text(json.dumps(journal,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        start,start_cpu=time.perf_counter(),time.process_time()
        try:
            preprocessor,owner,backbone=_load_modules(snapshot,encoder_directory,chain_report,source,
                case,document['metadata'],document['profiles'][case.profile],**options)
        except Exception as error:
            progress(phase='failed_loading',error_class=type(error).__name__)
            raise ValueError('Audio CPU model loading failed') from None
        load_wall,load_cpu=time.perf_counter()-start,time.process_time()-start_cpu
        configuration=owner.configuration
        _require(configuration['source']==source and configuration['case_id']==case.id and configuration['profile']==case.profile)
        wave=torch.from_numpy(case.waveform.copy())
        samples=torch.from_numpy(case.sample_mask.copy()).long()
        ids=torch.tensor([case.input_ids],dtype=torch.int64)
        mask=torch.tensor([case.attention_mask],dtype=torch.int64)
        empty=torch.empty(0,512)
        wanted_mask=torch.from_numpy(case.frame_mask.copy())
        rows=[]
        completed_stages=0
        partial=dict(schema=1,source=source,case=case_id,profile=case.profile,
            last_completed_stage=None,rounds=rows,model_execution_complete=False,
            CPU_comparison_qualified=False,complete_performance_accepted=False,
            scope='Completed stage snapshots only; not a complete round, numerical pass or process liveness record')
        partial_path=output/'partial-audio-CPU-captures.json'
        _write(partial_path,partial)
        def preprocess():
            return preprocessor(wave,samples)
        def encode(value):
            features,actual_mask=value
            _require(torch.equal(actual_mask,wanted_mask))
            return owner(features)
        def finish(value):
            return backbone(ids,mask,empty,value)
        def observe(stage,value):
            nonlocal completed_stages
            _require(stage==('preprocessing','encoder','backbone')[completed_stages%3])
            index=completed_stages//3
            _require(index<len(protocol.ROUNDS))
            name=protocol.ROUNDS[index]
            if stage=='preprocessing':
                captures={key:_capture(output,name+'-'+key+'.bin',tensor,dtype)
                    for key,tensor,dtype in (('features',value[0],'<f4'),('frame_mask',value[1],'<i4'))}
                rows.append(dict(round=name,captures=captures))
            elif stage=='encoder':
                rows[index]['captures']['encoder']=_capture(output,name+'-encoder.bin',value,'<f4')
            completed_stages+=1
            partial.update(last_completed_stage=stage,last_completed_round=name)
            temporary=partial_path.with_suffix('.tmp')
            _write(temporary,partial)
            temporary.replace(partial_path)
        def signature():
            digest=hashlib.sha256()
            for tensor in (wave,samples,ids,mask):digest.update(tensor.contiguous().numpy().tobytes())
            return digest.hexdigest()
        measured=protocol.CpuReferenceCase(module='audio-839' if case_id=='padded_noise' else 'audio-1119',
            precision='fp32',descriptor_sha256=protocol.DESCRIPTOR,reference_sha256=cohort.document_reference_sha256,
            input_sha256=case.input_sha256,reference_case_id=case.id,source=source,
            expected=torch.from_numpy(case.embedding.copy()),input_signature=signature,
            preprocess=preprocess,encoder=encode,backbone=finish,active_modules=(preprocessor,owner,backbone),after_stage=observe)
        progress(phase='measuring')
        error_class=None
        try:
            pipeline=protocol.measure_reference_pipeline(measured,output/'pipeline.json')
        except Exception as error:
            error_class=type(error).__name__
            pipeline=_json(output/'pipeline.json') if (output/'pipeline.json').is_file() else dict(success=False,rounds=[])
        progress(phase='inspecting_completed_stage_captures')
        inspections=[]
        if len(rows)==4 and all(len(v['captures'])==3 for v in rows):
            try:inspections=_inspections(output,rows,case)
            except ValueError:error_class=error_class or 'ValueError'
        after=cpu_execution_context()
        peak=_peak_rss_bytes()
        complete=len(pipeline.get('rounds',[]))==4
        success=bool(pipeline['success'] and complete and len(inspections)==4 and all(v['success'] for v in inspections)
            and before==after and not torch.cuda.is_initialized() and (peak is None or 0<peak<=16*1024**3))
        result=dict(schema=1,success=success,phase='complete' if success else 'failed_chain',
            preprocessing_implementation=PREPROCESSING_IMPLEMENTATION,
            scope='Four actual FP32 PCM preprocessing, audited ONNX encoder and PyTorch backbone CPU carries only',
            source=source,case=case_id,profile=case.profile,actual_backend='CPU',precision='fp32',
            model_revision=references.MODEL_REVISION,model_execution_complete=complete,error_class=error_class,
            descriptor_sha256=protocol.DESCRIPTOR,input_sha256=case.input_sha256,
            preprocessing_reference_sha256=cohort.preprocessing_reference_sha256,
            encoder_reference_sha256=cohort.encoder_reference_sha256,document_reference_sha256=cohort.document_reference_sha256,
            source_fingerprints_normalized_LF=fingerprints,snapshot_input_sha256=dict(cohort.snapshot_input_sha256),
            encoder_configuration=configuration,execution_context_before_load=before,execution_context_after_execution=after,
            execution_context_stable=before==after,load_wall_seconds=load_wall,load_process_cpu_seconds=load_cpu,
            host_peak_rss_bytes=peak,explicit_host_limit_bytes=16*1024**3,rounds=rows,intermediate_inspections=inspections,
            final_vector_gate_passed=bool(pipeline['success']),CUDA_context_initialized=torch.cuda.is_initialized(),
            runtime_versions={name:importlib.metadata.version(name) for name in ('torch','transformers','numpy','onnxruntime')},
            original_golden_changed=False,CPU_comparison_qualified=False,complete_performance_accepted=False)
        if retained_producer_source is not None:
            result['retained_chain_producer_source']=retained_producer_source
        _write(output/'audio-cpu-reference.json',result)
        progress(phase=result['phase'],model_execution_complete=complete,error_class=error_class,
            execution_context_after_execution=after)
        _require(success)
        return result
    except (OSError,ValueError,TypeError,KeyError,subprocess.SubprocessError):
        raise ValueError('Audio CPU reference chain failed; preserve existing outputs') from None
    finally:
        if owner is not None:owner.close()


def audit_audio_cpu_reference(output,preprocessing,encoder_directory,document_reference,*,source):
    """Recompute saved CPU intermediate/final gates; no model graph is loaded."""
    try:
        _require(isinstance(source,str) and re.fullmatch('[0-9a-f]{40}',source) is not None)
        output=Path(output)
        cohort=references.load_audio_cpu_reference_cohort(preprocessing,encoder_directory,document_reference)
        report=_json(output/'audio-cpu-reference.json')
        pipeline=_json(output/'pipeline.json')
        journal=_json(output/'audio-cpu-source.json')
        case=cohort.case(report['case'])
        _require(report['schema']==journal['schema']==1 and report['success'] is True and pipeline['success'] is True)
        _require(report['phase']==pipeline['phase']==journal['phase']=='complete')
        _require(report['source']==pipeline['source']==journal['source']==source)
        retained=report.get('retained_chain_producer_source')
        _require(journal.get('retained_chain_producer_source')==retained)
        qualification=report['encoder_configuration'].get('retained_graph_producer_qualification')
        if retained is not None:
            _require(qualification==_qualify_retained_graph_producer(retained,source))
        else:
            _require(qualification is None)
        _require(report['profile']==journal['profile']==case.profile and journal['case']==case.id)
        _require(report['model_revision']==journal['model_revision']==references.MODEL_REVISION)
        _require(report['model_execution_complete'] is journal['model_execution_complete'] is True)
        _require(report['error_class'] is journal['error_class'] is None)
        _require(report['actual_backend']==pipeline['actual_backend']=='CPU' and report['precision']==pipeline['precision']=='fp32')
        _require(report['preprocessing_implementation']==journal['preprocessing_implementation']==PREPROCESSING_IMPLEMENTATION)
        _require(report['CUDA_context_initialized'] is False and report['original_golden_changed'] is False)
        _require(report['CPU_comparison_qualified'] is report['complete_performance_accepted'] is False)
        _require(report['descriptor_sha256']==pipeline['descriptor_sha256']==protocol.DESCRIPTOR)
        _require(report['input_sha256']==pipeline['input_sha256']==case.input_sha256)
        _require(report['preprocessing_reference_sha256']==cohort.preprocessing_reference_sha256)
        _require(report['encoder_reference_sha256']==cohort.encoder_reference_sha256)
        _require(report['document_reference_sha256']==pipeline['reference_sha256']==cohort.document_reference_sha256)
        _require(report['snapshot_input_sha256']==journal['snapshot_input_sha256']==dict(cohort.snapshot_input_sha256))
        _require(report['source_fingerprints_normalized_LF']==journal['source_fingerprints_normalized_LF']==_validate_execution_source(source))
        _require(report['execution_context_stable'] is True and report['execution_context_before_load']==
            report['execution_context_after_execution']==journal['execution_context_before_load']==journal['execution_context_after_execution'])
        config=report['encoder_configuration']
        _require(config['source']==source and config['case_id']==case.id and config['profile']==case.profile)
        _require(config['original_reference_audit_passed'] is config['complete_graph_hashes_checked_at_load'] is True)
        _require(config['references_used_as_inference_inputs'] is config['CPU_reference_comparison_qualified'] is config['GPU_validated'] is config['complete_performance_accepted'] is False)
        _require(config['execution_schedule']=='chain-major' and config['maximum_owned_sessions']==len(config['sessions'])==5)
        _require(re.fullmatch('[0-9a-f]{64}',config['chain_report_sha256']) is not None)
        for session in config['sessions']:
            _require(session['providers']==['CPUExecutionProvider'] and session['graph_optimizations']=='disabled')
            _require(session['node_execution_order']=='PRIORITY_BASED' and session['execution_mode']=='sequential')
            _require(session['intra_op_threads']==4 and session['inter_op_threads']==1 and session['fallback_enabled'] is False)
            _require(all(session[k]=='disabled' for k in ('cpu_memory_arena','memory_pattern','constant_prepacking')))
            _require(session['onnxruntime_version']==report['runtime_versions']['onnxruntime'])
            _require(re.fullmatch('[0-9a-f]{64}',session['graph_sha256']) is not None)
        _require(pipeline['module']==('audio-839' if case.id=='padded_noise' else 'audio-1119') and pipeline['reference_case_id']==case.id)
        _require(pipeline['stage_observer_present'] is pipeline['stage_observer_time_excluded'] is True)
        _require(pipeline['input_signatures_checked_before_and_after_each_round'] is True)
        _require(len(report['rounds'])==len(pipeline['rounds'])==4)
        inspections=_inspections(output,report['rounds'],case)
        _require(all(row['success'] for row in inspections) and inspections==report['intermediate_inspections'])
        finals=[]
        wanted=case.embedding.astype(np.float64)
        for name,row in zip(protocol.ROUNDS,pipeline['rounds'],strict=True):
            _require(row['round']==name and row['capture']=='pipeline-'+name+'.bin')
            raw=(output/row['capture']).read_bytes()
            _require(len(raw)==3072 and hashlib.sha256(raw).hexdigest()==row['capture_sha256'])
            actual=np.frombuffer(raw,dtype='<f4').astype(np.float64)
            norm=float(actual@actual)
            cosine=float(actual@wanted)/float(np.sqrt(norm*(wanted@wanted))) if norm>0 else 0.
            _require(bool(np.isfinite(actual).all()) and abs(norm-1)<=.002 and cosine>=.999)
            _require([v['stage'] for v in row['stages']]==['preprocessing','encoder','backbone'])
            _require(all(np.isfinite(v['wall_seconds']) and v['wall_seconds']>0 and
                np.isfinite(v['process_cpu_seconds']) and v['process_cpu_seconds']>=0 and v['gpu_elapsed_ns'] is None for v in row['stages']))
            _require(row['input_state_sha256']==pipeline['input_state_sha256'])
            finals.append(dict(round=name,cosine=cosine,norm_squared=norm))
        peak=report['host_peak_rss_bytes']
        _require(report['explicit_host_limit_bytes']==16*1024**3 and (peak is None or type(peak) is int and 0<peak<=16*1024**3))
        return dict(schema=1,source=source,case=case.id,intermediate_inspections=inspections,final_inspections=finals,
            all_four_chains_match_originals=True,original_golden_changed=False,
            graph_binaries_independently_rehashed_by_capture_auditor=False,
            CPU_comparison_qualified=False,complete_performance_accepted=False,GPU_validated=False)
    except (OSError,ValueError,TypeError,KeyError,IndexError,subprocess.SubprocessError):
        raise ValueError('Audio CPU saved-capture audit failed') from None


def main(argv=None):
    parser=argparse.ArgumentParser(prog='audio_cpu_reference',description=__doc__,exit_on_error=False)
    parser.add_argument('command',choices=('measure','audit'))
    parser.add_argument('--preprocessing',type=Path,required=True)
    parser.add_argument('--encoder-reference',type=Path,required=True)
    parser.add_argument('--document-reference',type=Path,required=True)
    parser.add_argument('--chain-report',type=Path)
    parser.add_argument('--case',choices=references.CPU_CASES)
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',default=os.environ.get('GITHUB_SHA'))
    parser.add_argument('--retained-chain-source',help='Explicit retained export producer; require identical generator inputs')
    try:
        args=parser.parse_args(argv)
        if args.command=='audit':
            _require(args.retained_chain_source is None)
            result=audit_audio_cpu_reference(args.output,args.preprocessing,args.encoder_reference,
                args.document_reference,source=args.source)
            _write(args.output/'independent-audio-CPU-audit.json',result)
        else:
            _require(args.case in references.CPU_CASES and args.chain_report is not None)
            _require(isinstance(args.source,str) and re.fullmatch('[0-9a-f]{40}',args.source) is not None)
            _empty(args.output)
            _validate_execution_source(args.source)
            cohort=references.load_audio_cpu_reference_cohort(args.preprocessing,args.encoder_reference,args.document_reference)
            options={} if args.retained_chain_source is None else dict(retained_producer_source=args.retained_chain_source)
            _admit_chain(args.chain_report,args.encoder_reference,args.source,cohort.case(args.case),**options)
            torch.set_num_threads(4)
            torch.set_num_interop_threads(2)
            snapshot=args.snapshot if args.snapshot is not None else _download_snapshot(cohort)
            measure_audio_cpu_reference(snapshot,args.preprocessing,args.encoder_reference,args.document_reference,
                args.chain_report,args.output,source=args.source,case_id=args.case,**options)
    except Exception as error:
        print(json.dumps(dict(success=False,error_class=type(error).__name__)))
        return 1
    print(json.dumps(dict(success=True,scope='Audio CPU reference only',complete_performance_accepted=False)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
