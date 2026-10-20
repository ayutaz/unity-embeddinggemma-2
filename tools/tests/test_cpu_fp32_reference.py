"""Differential checks against scalar C99 fmaf, not a GPU or model pass."""
import ctypes
import ctypes.util
import sys
import numpy as np
import pytest
from embeddinggemma_tools.cpu_fp32_reference import NativeFp32Reference

@pytest.fixture(scope='module')
def native(tmp_path_factory):return NativeFp32Reference.build(tmp_path_factory.mktemp('cpu-fp32-reference'))

@pytest.fixture(scope='module')
def scalar():
    library=ctypes.CDLL('ucrtbase' if sys.platform=='win32' else ctypes.util.find_library('m'))
    function=library.fmaf;function.argtypes=[ctypes.c_float]*3;function.restype=ctypes.c_float
    return function

def oracle(a,b,bias,fma):
    a=a[None] if a.ndim==2 else a;b=b[None] if b.ndim==2 else b
    batch,rows,width=a.shape;columns=b.shape[-1];out=np.empty((batch,rows,columns),np.float32)
    for q in range(batch):
        for i in range(rows):
            for j in range(columns):
                total=None
                for begin in range(0,width,192):
                    value=np.float32(0)
                    for k in range(begin,min(begin+192,width)):value=np.float32(fma(float(a[q,i,k]),float(b[q,k,j]),float(value)))
                    if begin+192>width:value=np.float32(fma(0.,0.,float(value)))
                    if total is None:total=np.float32(value+(bias[j] if bias is not None else np.float32(0))) if bias is not None else value
                    else:total=np.float32(total+value)
                out[q,i,j]=total
    return out

@pytest.mark.parametrize('batch,rows,width,columns',[(1,1,1,1),(1,2,7,5),(2,2,191,10),(1,3,192,16),(2,2,193,19),(1,2,385,8)])
def test_block192_bias_order_tail_and_batches_match_independent_scalar_oracle(native,scalar,batch,rows,width,columns):
    rng=np.random.default_rng(width+columns)
    a=rng.normal(size=(batch,rows,width)).astype(np.float32)
    b=rng.normal(size=(batch,width,columns)).astype(np.float32)
    bias=rng.normal(size=columns).astype(np.float32)
    original=(a.tobytes(),b.tobytes(),bias.tobytes())
    expected=oracle(a,b,bias,scalar)
    actual=native.matmul(a,b,bias=bias)
    np.testing.assert_array_equal(actual.view('u4'),expected.view('u4'))
    assert (a.tobytes(),b.tobytes(),bias.tobytes())==original
    assert native.configuration['GPU_execution_validated'] is False

def test_first_block_bias_is_folded_before_later_blocks(native):
    a=np.zeros((1,193),np.float32);b=np.ones((193,7),np.float32)
    a[0,0]=524288;a[0,192]=-524288;b[0,:]=32;b[192,:]=32
    actual=native.matmul(a,b,bias=np.ones(7,np.float32))
    np.testing.assert_array_equal(actual,np.zeros((1,7),np.float32))

def test_explicit_fma_preserves_a_residual_lost_by_separate_product_rounding(native):
    a=np.array([1+2.**-23],np.float32);b=np.array([1-2.**-23],np.float32);c=np.array([-1],np.float32)
    actual=native.fma(a,b,c)
    np.testing.assert_array_equal(actual,np.array([-2.**-46],np.float32))
    assert np.float32(a[0]*b[0]+c[0])==0

def test_elementwise_fma_and_broadcast_match_scalar_c99(native,scalar):
    rng=np.random.default_rng(23)
    a=(rng.normal(size=(7,1))*2.**rng.integers(-10,10,size=(7,1))).astype(np.float32)
    b=rng.normal(size=(1,13)).astype(np.float32);c=rng.normal(size=(7,13)).astype(np.float32)
    aa,bb,cc=np.broadcast_arrays(a,b,c)
    expected=np.array([scalar(float(x),float(y),float(z)) for x,y,z in zip(aa.flat,bb.flat,cc.flat)],np.float32).reshape(7,13)
    np.testing.assert_array_equal(native.fma(a,b,c).view('u4'),expected.view('u4'))

def test_noncontiguous_inputs_are_copied_without_mutation(native,scalar):
    a=np.arange(30,dtype=np.float32).reshape(3,10)[:,::-2]
    b=np.arange(35,dtype=np.float32).reshape(7,5).T
    expected=oracle(a,b,None,scalar)[0]
    np.testing.assert_array_equal(native.matmul(a,b),expected)

@pytest.mark.parametrize('fault',['dtype','nan','shape','batch','empty','bias'])
def test_bad_inputs_are_rejected_before_native_execution(native,fault):
    a=np.ones((2,5),np.float32);b=np.ones((5,3),np.float32);bias=None
    if fault=='dtype':a=a.astype(np.float64)
    elif fault=='nan':a[0,0]=np.nan
    elif fault=='shape':b=np.ones((4,3),np.float32)
    elif fault=='batch':a=np.ones((2,2,5),np.float32);b=np.ones((3,5,3),np.float32)
    elif fault=='empty':a=np.ones((0,5),np.float32)
    else:bias=np.ones((2,3),np.float32)
    with pytest.raises(ValueError):native.matmul(a,b,bias=bias)

def test_library_and_source_bytes_are_bound_and_changed_cache_is_rejected(native):
    assert len(native.configuration['source_sha256'])==len(native.configuration['native_library_sha256'])==64
    assert native.configuration['model_weights_changed'] is False
    assert native.configuration['AVX2_FMA_and_OS_vector_state_verified'] is True


def test_changed_cached_build_is_rejected_without_overwriting_evidence(native,tmp_path):
    import json,shutil
    from pathlib import Path
    directory=tmp_path/'changed-cache';directory.mkdir()
    original=Path(native.library._name)
    shutil.copyfile(original,directory/original.name)
    data={k:v for k,v in native.configuration.items() if k in ('source_sha256','compiler_binary_sha256','compiler_kind','compiler_flags','native_library_sha256')}
    data['native_library_sha256']='0'*64
    manifest=directory/'build.json';manifest.write_text(json.dumps(data),encoding='utf-8')
    before=manifest.read_bytes()
    with pytest.raises(ValueError,match='differs'):NativeFp32Reference.build(directory)
    assert manifest.read_bytes()==before


@pytest.mark.parametrize('shape',[(193,),(2,193),(2,3,193)])
def test_torch_linear_adapter_preserves_rank_bias_and_original_weights(native,scalar,shape):
    import torch
    from embeddinggemma_tools.cpu_fp32_reference import TorchFp32Reference
    rng=np.random.default_rng(19);a=rng.normal(size=shape).astype(np.float32)
    weight=rng.normal(size=(9,193)).astype(np.float32);bias=rng.normal(size=9).astype(np.float32)
    wanted=oracle(a.reshape(-1,193),weight.T,bias,scalar)[0].reshape(*shape[:-1],9)
    values,weights,offset=map(torch.from_numpy,(a,weight,bias))
    copies=[v.clone() for v in (values,weights,offset)]
    engine=TorchFp32Reference(native)
    with torch.no_grad():actual=engine.linear(values,weights,offset)
    np.testing.assert_array_equal(actual.numpy().view('u4'),wanted.view('u4'))
    assert all(torch.equal(x,y) for x,y in zip((values,weights,offset),copies))
    assert engine.calls=={'linear':1,'matmul':0,'fma':0}


def test_linear_scope_runs_actual_fma_path_and_restores_after_failure(native,monkeypatch):
    import torch
    from embeddinggemma_tools.cpu_fp32_reference import TorchFp32Reference
    original=torch.nn.functional.linear
    engine=TorchFp32Reference(native)
    values=torch.tensor([[1.,2.]],dtype=torch.float32)
    weight=torch.tensor([[3.,4.]],dtype=torch.float32)
    with pytest.raises(RuntimeError,match='downstream'):
        with torch.no_grad(),engine.linear_scope():
            assert torch.nn.functional.linear(values,weight).item()==11
            assert engine.calls['linear']==1
            with pytest.raises(ValueError,match='active'):
                with TorchFp32Reference(native).linear_scope():pass
            raise RuntimeError('downstream')
    assert torch.nn.functional.linear is original


@pytest.mark.parametrize('fault',['grad','float64','rank','device'])
def test_torch_reference_rejects_unsupported_inputs_instead_of_native_fallback(native,fault):
    import torch
    from embeddinggemma_tools.cpu_fp32_reference import TorchFp32Reference
    engine=TorchFp32Reference(native);value=torch.ones(2,3);weight=torch.ones(4,3)
    if fault=='grad':
        with torch.enable_grad(),pytest.raises(ValueError):engine.linear(value,weight)
    else:
        if fault=='float64':value=value.double()
        elif fault=='rank':weight=weight.unsqueeze(0)
        else:value=torch.empty(2,3,device='meta')
        with torch.no_grad(),pytest.raises(ValueError):engine.linear(value,weight)
    assert engine.calls=={'linear':0,'matmul':0,'fma':0}


def test_torch_matmul_and_broadcast_fma_call_explicit_native_operations(native,scalar):
    import torch
    from embeddinggemma_tools.cpu_fp32_reference import TorchFp32Reference
    engine=TorchFp32Reference(native)
    a=torch.arange(12,dtype=torch.float32).reshape(2,2,3);b=torch.ones(2,3,5)
    with torch.no_grad():
        actual=engine.matmul(a,b)
        residue=engine.fma(torch.tensor([1+2.**-23]),torch.tensor([1-2.**-23]),torch.tensor([-1.]))
    np.testing.assert_array_equal(actual.numpy(),oracle(a.numpy(),b.numpy(),None,scalar))
    assert residue.item()==-2.**-46
    assert engine.calls=={'linear':0,'matmul':1,'fma':1}


@pytest.mark.parametrize('channels',[1,16,32])
def test_stem_convolution_follows_kernel_then_channel_then_separate_block_order(native,scalar,channels):
    import torch
    from embeddinggemma_tools.cpu_fp32_reference import TorchFp32Reference
    rng=np.random.default_rng(channels)
    values=rng.normal(size=(1,channels,3,5)).astype(np.float32)
    weight=rng.normal(size=(3,channels,3,3)).astype(np.float32)
    wanted=np.empty((1,3,2,3),np.float32);chunk=1 if channels==1 else 16
    for out in range(3):
        for row in range(2):
            for col in range(3):
                total=None
                for start in range(0,channels,chunk):
                    partial=np.float32(0)
                    for kr in range(3):
                        for kc in range(3):
                            y,x=row*2+kr-1,col*2+kc-1
                            for c in range(start,start+chunk):
                                value=float(values[0,c,y,x]) if 0<=y<3 and 0<=x<5 else 0.
                                partial=np.float32(scalar(value,float(weight[out,c,kr,kc]),float(partial)))
                    total=partial if total is None else np.float32(total+partial)
                wanted[0,out,row,col]=total
    engine=TorchFp32Reference(native)
    with torch.no_grad():actual=engine.stem_convolution(torch.from_numpy(values),torch.from_numpy(weight))
    np.testing.assert_array_equal(actual.numpy().view('u4'),wanted.view('u4'))
    assert engine.calls['stem_convolution']==1


@pytest.mark.parametrize('fault',['channels','kernel','rank'])
def test_stem_reference_rejects_unaudited_convolution_contracts(native,fault):
    import torch
    from embeddinggemma_tools.cpu_fp32_reference import TorchFp32Reference
    values=torch.ones(1,16,3,5);weight=torch.ones(3,16,3,3)
    if fault=='channels':values=values[:,:3];weight=weight[:,:3]
    elif fault=='kernel':weight=weight[:,:,:2,:2]
    else:values=values[0]
    with torch.no_grad(),pytest.raises(ValueError):TorchFp32Reference(native).stem_convolution(values,weight)


def test_four_dimensional_attention_heads_match_independent_FMA_oracle(native,scalar):
    import torch
    from embeddinggemma_tools.cpu_fp32_reference import TorchFp32Reference
    rng=np.random.default_rng(44)
    a=rng.normal(size=(1,2,3,7)).astype(np.float32);b=rng.normal(size=(1,2,7,5)).astype(np.float32)
    wanted=oracle(a.reshape(2,3,7),b.reshape(2,7,5),None,scalar).reshape(1,2,3,5)
    engine=TorchFp32Reference(native)
    with torch.no_grad():actual=engine.head_matmul(torch.from_numpy(a),torch.from_numpy(b))
    np.testing.assert_array_equal(actual.numpy().view('u4'),wanted.view('u4'))
    assert engine.calls['head_matmul']==1


def test_head_matmul_scope_selects_only_rank4_and_restores_original(native):
    import torch
    from embeddinggemma_tools.cpu_fp32_reference import TorchFp32Reference
    original=torch.matmul;engine=TorchFp32Reference(native)
    with pytest.raises(RuntimeError,match='downstream'):
        with torch.no_grad(),engine.head_matmul_scope():
            assert torch.matmul(torch.ones(1,2,3,4),torch.ones(1,2,4,5)).shape==(1,2,3,5)
            assert torch.matmul(torch.ones(2,3),torch.ones(3,4)).shape==(2,4)
            assert engine.calls['head_matmul']==1
            with pytest.raises(ValueError):
                with engine.head_matmul_scope():pass
            raise RuntimeError('downstream')
    assert torch.matmul is original
