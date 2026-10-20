"""Small contracts for explicit CPU execution; real-model results are separate."""
import importlib
import pytest
import torch
from test_cpu_fp32_reference import native

SOURCE='a'*40

@pytest.fixture
def prepared(native,monkeypatch,tmp_path):
    module=importlib.import_module('embeddinggemma_tools.native_audio_cpu_encoder')
    program=torch.nn.Linear(1,1,bias=False)
    calls=[]
    def load(directory,original,*,case_id):
        calls.append(case_id)
        return program,torch.tensor([999.]),torch.tensor([888.]),dict(case_id=case_id,
            profile='frames-1119-valid-'+('839' if case_id=='padded_noise' else '1119'),
            executed_graph_sha256='b'*64,graph_producer='c'*40)
    monkeypatch.setattr(module,'load_distribution_input',load)
    class Part(torch.nn.Module):
        def __init__(self,index,padded):super().__init__();self.index=index;self.padded=padded
        def forward(self,value):
            if self.index<4:return (value.mean()+1).expand(1,280,1024).contiguous()
            return value.mean().expand(210 if self.padded else 280,512).contiguous()
    monkeypatch.setattr(module,'_make_wrappers',lambda program,operations,case_id:
        [Part(index,case_id=='padded_noise') for index in range(5)])
    def make(case_id='silence',source=SOURCE):
        return module.NativeAudioCpuEncoder(tmp_path,tmp_path,source=source,case_id=case_id,native=native)
    return module,make,program,calls

@pytest.mark.parametrize('case_id',['silence','padded_noise'])
def test_actual_values_carry_through_five_parts_and_release_is_explicit(prepared,case_id):
    _,make,_,calls=prepared
    encoder=make(case_id)
    first=encoder(torch.ones(1,1119,128));second=encoder(torch.full((1,1119,128),2.))
    assert tuple(first.shape)==(210 if case_id=='padded_noise' else 280,512)
    assert torch.equal(first,torch.full_like(first,5.)) and torch.equal(second,first+1)
    assert calls==[case_id] and encoder.verify_coefficients() is True
    info=encoder.configuration
    assert info['executor']=='native-fp32-reference' and info['source']==SOURCE
    assert info['references_used_as_inference_inputs'] is info['original_ONNX_bytes_executed'] is False
    assert info['original_coefficients_verified_unchanged'] is True
    encoder.close();encoder.close()
    assert not list(encoder.parameters()) and encoder.configuration['closed'] is True
    with pytest.raises(ValueError):encoder(torch.ones(1,1119,128))

@pytest.mark.parametrize('source',[None,'main','A'*40])
def test_invalid_source_rejected_before_loading(prepared,source):
    _,make,_,calls=prepared
    with pytest.raises(ValueError):make(source=source)
    assert not calls

@pytest.mark.parametrize('fault',['shape','dtype','nan'])
def test_bad_inference_input_never_reaches_parts(prepared,fault):
    _,make,_,_=prepared;encoder=make();value=torch.ones(1,1119,128)
    if fault=='shape':value=value[:,:100]
    elif fault=='dtype':value=value.double()
    else:value[0,0,0]=float('nan')
    with pytest.raises(ValueError):encoder(value)
    encoder.close()

def test_modified_original_coefficient_is_detected(prepared):
    _,make,program,_=prepared;encoder=make()
    with torch.no_grad():program.weight.add_(1)
    with pytest.raises(ValueError):encoder.verify_coefficients()
    assert encoder.configuration['original_coefficients_verified_unchanged'] is False
    encoder.close()

def test_shared_fma_functions_restored_after_downstream_failure_and_nested_scope_rejected(prepared):
    module,make,_,_=prepared;encoder=make()
    export=importlib.import_module('embeddinggemma_tools.export');original=export.fp32_fma
    class Failure(torch.nn.Module):
        def forward(self,value):
            assert export.fp32_fma is not original
            with pytest.raises(ValueError):
                with module.native_audio_fma_scope(encoder.operations.fma):pass
            raise RuntimeError('synthetic downstream failure')
    encoder.parts[2]=Failure()
    with pytest.raises(ValueError):encoder(torch.ones(1,1119,128))
    assert export.fp32_fma is original
    encoder.close()

def test_partial_loading_failure_never_changes_shared_math_functions(prepared,monkeypatch):
    module,make,_,_=prepared
    export=importlib.import_module('embeddinggemma_tools.export');original=export.fp32_fma
    def fail(*args):raise MemoryError('private diagnostic details')
    monkeypatch.setattr(module,'_make_wrappers',fail)
    with pytest.raises(ValueError,match='loading failed'):make()
    assert export.fp32_fma is original

def test_native_primitive_binding_uses_real_operations_in_all_parts(native,monkeypatch):
    module=importlib.import_module('embeddinggemma_tools.native_audio_cpu_encoder')
    from embeddinggemma_tools.cpu_fp32_reference import TorchFp32Reference
    program=torch.nn.Identity();operations=TorchFp32Reference(native)
    selected=[]
    class Wrapper(torch.nn.Module):
        def __init__(self,*args):super().__init__();selected.append(self)
    monkeypatch.setattr(module,'PrefixExport',Wrapper)
    monkeypatch.setattr(module,'SegmentExport',Wrapper)
    monkeypatch.setattr(module,'extract_prefix',lambda *args:program)
    monkeypatch.setattr(module,'extract_segment',lambda *args:program)
    parts=module._make_wrappers(program,operations,'silence')
    assert len(parts)==len(selected)==5
    a=torch.tensor([[1.,2.]]);b=torch.tensor([[3.],[4.]])
    with torch.no_grad():
        assert parts[0].linear_candidate(a,b).item()==11
        assert parts[0].biased_linear_candidate(torch.tensor([3.]),a,b).item()==14
        assert parts[1].bmm_candidate(a[None],b[None]).item()==11
        actual=parts[0].convolution_candidate(torch.arange(1,5,dtype=torch.float32).reshape(1,1,2,2),torch.ones(1,1,3,3))
    assert actual.item()==10 and operations.calls['matmul']==3 and operations.calls['stem_convolution']==1
