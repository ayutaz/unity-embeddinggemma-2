import base64,hashlib,json,zlib
from pathlib import Path
import numpy as np
import pytest
import torch
from embeddinggemma_tools import reference_context_failure as context


def function():
    result=getattr(context,'ordered_vision_softmax16',None)
    assert callable(result),'CPU-only ordered vision softmax diagnostic is missing'
    return result


def fixture():
    value=json.loads((Path(__file__).parent/'fixtures/native_softmax16_rows.json').read_text(encoding='utf-8'))
    result=[]
    for key in ('input','expected'):
        raw=zlib.decompress(base64.b64decode(value[key]['data']));assert hashlib.sha256(raw).hexdigest()==value[key]['sha256']
        result.append(torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy().reshape(value['shape'])))
    return result


def test_ordered_cpu_softmax_matches_frozen_actual_avx512_prefix_bits_and_preserves_borrowed_input():
    values,expected=fixture();before=values.clone();actual=function()(values)
    assert torch.equal(actual.view(torch.int32),expected.view(torch.int32))
    assert actual.device.type=='cpu' and actual.dtype==torch.float32 and actual.data_ptr()!=values.data_ptr()
    assert torch.equal(values,before)


def test_ordered_cpu_softmax_is_stable_for_large_finite_logits_without_overflow():
    values=torch.zeros((2,2520),dtype=torch.float32);values[0]=10000;values[1]=-10000
    actual=function()(values)
    expected=np.float32(1/2520)
    assert torch.equal(actual,torch.full_like(actual,float(expected)))
    assert torch.isfinite(actual).all()


def mode_type():
    value=getattr(context,'VisionSoftmax16Diagnostic',None)
    assert value is not None,'Scoped ordered vision softmax dispatch is missing'
    return value


def test_diagnostic_dispatch_is_scoped_and_does_not_change_unrelated_native_operations():
    values,expected=fixture();original=torch.ops.aten._softmax.default(values,-1,False)
    mode=mode_type()()
    with mode:
        actual=torch.ops.aten._softmax.default(values,-1,False)
        unrelated=torch.add(values,7)
    assert mode.replacements==1 and torch.equal(actual.view(torch.int32),expected.view(torch.int32))
    assert torch.equal(unrelated,values+7)
    assert torch.equal(torch.ops.aten._softmax.default(values,-1,False).view(torch.int32),original.view(torch.int32))


def test_diagnostic_dispatch_rejects_wrong_softmax_axis_without_hiding_missing_execution():
    values,_=fixture();mode=mode_type()()
    with mode:
        with pytest.raises(ValueError):torch.ops.aten._softmax.default(values,0,False)
    assert mode.replacements==0


@pytest.mark.parametrize('problem',['double','int','width','rank','empty','nan','noncontiguous'])
def test_ordered_cpu_softmax_rejects_unqualified_inputs(problem):
    values=torch.zeros((2,2520),dtype=torch.float32)
    if problem=='double':values=values.double()
    elif problem=='int':values=values.int()
    elif problem=='width':values=values[:,:2519].contiguous()
    elif problem=='rank':values=values[0]
    elif problem=='empty':values=values[:0]
    elif problem=='nan':values[0,0]=float('nan')
    else:values=torch.zeros((2520,2)).t()
    with pytest.raises((TypeError,ValueError)):function()(values)
