"""Four real chained CPU reference calls; not a Sentis CPU support claim."""
from dataclasses import dataclass
import hashlib,json,math,re,time
from pathlib import Path
from typing import Callable

import torch

DESCRIPTOR='40aefa5de06f575d31b7582c24314c1ab035da374116b3ce3036de0e919cd647'
IMAGE_REFERENCE='90692ad38c6d7d844dc6b6e79c7d5cfa719ea59866946dbf94a40bca10ac9eac'
AUDIO_REFERENCE='281d12573b431667e8841033db1c6a10dce4382ef98c9a5345c18044a1194001'
CASES={'image-square':'square-0','image-landscape':'landscape-0','image-portrait':'portrait-0','audio-1119':'silence','audio-839':'padded_noise'}
ROUNDS=('first','warm','repeat-1','repeat-2')


@dataclass(frozen=True)
class CpuReferenceCase:
    module:str
    precision:str
    descriptor_sha256:str
    reference_sha256:str
    input_sha256:str
    reference_case_id:str
    source:str
    expected:torch.Tensor
    input_signature:Callable
    preprocess:Callable
    encoder:Callable
    backbone:Callable
    active_modules:tuple


def _tensor(value):
    if not isinstance(value,torch.Tensor) or value.device.type!='cpu':
        raise ValueError('Reference tensors must stay on CPU')
    if value.is_floating_point() and value.dtype!=torch.float32:
        raise ValueError('Reference floating tensors must use the matched FP32 representation')


def _tree(value):
    if isinstance(value,torch.Tensor):_tensor(value)
    elif isinstance(value,(tuple,list)) and value:
        for item in value:_tree(item)
    elif isinstance(value,dict) and value:
        for item in value.values():_tree(item)
    else:raise ValueError('A measured stage must return actual CPU tensors')


def _vector(value):
    _tensor(value)
    if value.dtype!=torch.float32 or tuple(value.shape) not in ((768,),(1,768)):
        raise ValueError('Reference vectors must have exactly 768 FP32 values')
    numbers=value.detach().reshape(768).tolist()
    if not all(math.isfinite(v) for v in numbers):raise ValueError('Reference vector is not finite')
    norm=math.fsum(v*v for v in numbers)
    if abs(norm-1)>.002:raise ValueError('Reference vector is not already unit length')
    return numbers,norm


def _validate(case):
    if case.module not in CASES or case.reference_case_id!=CASES[case.module]:
        raise ValueError('Choose the same fixed input case as the reviewed GPU measurement')
    expected=IMAGE_REFERENCE if case.module.startswith('image-') else AUDIO_REFERENCE
    if case.precision!='fp32' or case.descriptor_sha256!=DESCRIPTOR or case.reference_sha256!=expected:
        raise ValueError('A matched FP32 descriptor and original reference are required')
    if re.fullmatch('[a-f0-9]{40}',case.source or '') is None or re.fullmatch('[a-f0-9]{64}',case.input_sha256 or '') is None:
        raise ValueError('Immutable source and input hashes are required')
    if not case.active_modules or not all(isinstance(m,torch.nn.Module) for m in case.active_modules):
        raise ValueError('Actual active CPU modules are required')
    for module in case.active_modules:
        for value in list(module.parameters())+list(module.buffers()):_tensor(value)
    return _vector(case.expected)


def measure_reference_pipeline(case,output):
    """Keep timings separate from validation, and never replace an intermediate."""
    expected,expected_norm=_validate(case);output=Path(output)
    captures={name:output.with_name(output.stem+'-'+name+'.bin') for name in ROUNDS}
    if output.exists() or any(path.exists() for path in captures.values()):raise FileExistsError('Preserve every existing report and capture')
    signature=case.input_signature()
    if re.fullmatch('[a-f0-9]{64}',signature or '') is None:raise ValueError('Input state requires a SHA256 signature')
    output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(schema=1,success=False,phase='starting',scope='Reference CPU chained numerical repeats and process timings only',framework='PyTorch reference',actual_backend='CPU',module=case.module,precision=case.precision,descriptor_sha256=case.descriptor_sha256,reference_sha256=case.reference_sha256,input_sha256=case.input_sha256,reference_case_id=case.reference_case_id,source=case.source,input_state_sha256=signature,rounds=[],CPU_scope='Whole reference process CPU threads; not Sentis/Editor CPU time',GPU_execution_claimed=False,complete_performance_accepted=False,model_generation_qualified=False,stage_scope='Actual callback only; protocol validation/report writes are excluded',case_scope='Three actual stages plus protocol observers; input signatures and vector checks outside the interval')
    with output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    def save():output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    try:
        with torch.no_grad():
            for round_name in ROUNDS:
                if case.input_signature()!=signature:raise ValueError('The repeated input state changed')
                report['phase']=round_name;save();stages=[];wall=time.perf_counter();cpu=time.process_time()
                def execute(name,function,*args):
                    begin=time.perf_counter();begin_cpu=time.process_time();value=function(*args)
                    elapsed=time.perf_counter()-begin;used=time.process_time()-begin_cpu
                    _tree(value)
                    stages.append(dict(stage=name,wall_seconds=elapsed,process_cpu_seconds=used,gpu_elapsed_ns=None))
                    return value
                preprocessed=execute('preprocessing',case.preprocess)
                encoded=execute('encoder',case.encoder,preprocessed)
                actual=execute('backbone',case.backbone,encoded)
                elapsed=time.perf_counter()-wall;used=time.process_time()-cpu
                if case.input_signature()!=signature:raise ValueError('A stage mutated the repeated input state')
                values,norm=_vector(actual)
                cosine=math.fsum(x*y for x,y in zip(values,expected,strict=True))/math.sqrt(norm*expected_norm)
                if not math.isfinite(cosine) or cosine<.999:raise ValueError('Actual CPU output differs from the original reference')
                capture=captures[round_name];raw=actual.detach().reshape(768).contiguous().numpy().astype('<f4',copy=False).tobytes()
                with capture.open('xb') as f:f.write(raw)
                report['rounds'].append(dict(round=round_name,wall_seconds=elapsed,process_cpu_seconds=used,stages=stages,cosine=cosine,norm_squared=norm,input_state_sha256=signature,capture=capture.name,capture_sha256=hashlib.sha256(raw).hexdigest()));save()
        report.update(success=True,phase='complete',input_signatures_checked_before_and_after_each_round=True);save();return report
    except BaseException as error:
        report.update(success=False,phase='failed',error_class=type(error).__name__);save();raise
