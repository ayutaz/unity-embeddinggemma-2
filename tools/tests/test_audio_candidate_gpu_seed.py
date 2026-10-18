import importlib
import hashlib
import json

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest
import torch

from embeddinggemma_tools.export import TorchWelfordLayerNorm, export_audio_native_onnx


def module():
    return importlib.import_module('embeddinggemma_tools.audio_candidate_gpu_seed')


class Encoder(torch.nn.Module):
    def __init__(self, statistics=False):
        super().__init__()
        self.wide = torch.nn.LayerNorm(128, eps=1e-6, bias=False)
        self.narrow = torch.nn.LayerNorm(32, eps=1e-6, bias=False)
        self.linear = torch.nn.Linear(32, 7, bias=False)
        self.statistics = statistics
        with torch.no_grad():
            self.wide.weight.copy_(torch.linspace(.7, 1.3, 128))
            self.narrow.weight.copy_(torch.linspace(1.2, .8, 32))

    def forward(self, x):
        a = torch.ops.aten.native_layer_norm.default(x, [128], self.wide.weight, None, 1e-6)
        b = torch.ops.aten.native_layer_norm.default(a[0][..., :32], [32], self.narrow.weight, None, 1e-6)
        out = self.linear(torch.nn.functional.silu(b[0])) + x[..., :7]
        return (out, a[1]) if self.statistics else out


class EncoderWithCopy(Encoder):
    def forward(self,x):
        return super().forward(torch.ops.aten._to_copy.default(x,dtype=torch.float32))


@pytest.fixture
def saved():
    torch.manual_seed(71)
    encoder = Encoder().eval()
    values = torch.randn(1, 3, 128)
    return encoder, values, torch.export.export(encoder, (values,)).module()


def test_rewrite_preserves_parameter_objects_source_graph_and_unrelated_operations(saved):
    encoder, values, source = saved
    graph_before = str(source.graph)
    state = {k: v.clone() for k, v in source.state_dict().items()}
    derived, report = module().rewrite_welford_seed(source)
    expected = TorchWelfordLayerNorm(encoder.wide)(values)
    expected = TorchWelfordLayerNorm(encoder.narrow)(expected[..., :32])
    expected = encoder.linear(torch.nn.functional.silu(expected)) + values[..., :7]
    torch.testing.assert_close(derived(values), expected, rtol=0, atol=0)
    assert str(source.graph) == graph_before
    assert not any(n.target == torch.ops.aten.native_layer_norm.default for n in derived.graph.nodes)
    for name, value in source.named_parameters():
        assert derived.get_parameter(name) is value
        assert torch.equal(value, state[name])
    assert report['normalization_widths'] == [32, 128]
    assert report['replaced_normalizations'] == 2
    assert report['source_parameter_objects_preserved'] is True
    assert report['new_learned_parameters'] == 0
    assert report['gpu_validated'] is False and report['generation_qualified'] is False


@pytest.mark.parametrize('problem', ['epsilon', 'bias', 'widths', 'computed_scale'])
def test_unqualified_native_graph_is_rejected_before_rewriting(saved, problem):
    _, _, source = saved
    nodes = [n for n in source.graph.nodes if n.target == torch.ops.aten.native_layer_norm.default]
    node = nodes[0]; args = list(node.args)
    if problem == 'epsilon': args[4] = 1e-5
    if problem == 'bias': args[3] = args[2]
    if problem == 'widths': args[1] = [32]
    if problem == 'computed_scale': args[2] = args[0]
    node.args = tuple(args)
    before = str(source.graph)
    with pytest.raises(ValueError): module().rewrite_welford_seed(source)
    assert str(source.graph) == before


def test_live_native_statistics_cannot_be_discarded():
    encoder = Encoder(statistics=True).eval(); values = torch.randn(1, 2, 128)
    source = torch.export.export(encoder, (values,)).module()
    with pytest.raises(ValueError, match='statistic'): module().rewrite_welford_seed(source)


def test_saved_graph_can_export_standard_onnx_without_reloading_official_weights(saved, tmp_path):
    _, values, source = saved
    derived, _ = module().rewrite_welford_seed(source)
    path = tmp_path/'seed.onnx'
    export_audio_native_onnx(derived, values, path)
    model = onnx.load(path); onnx.checker.check_model(model)
    assert not any(n.op_type == 'LayerNormalization' or n.domain not in ('', 'ai.onnx') for n in model.graph.node)
    actual = ReferenceEvaluator(model).run(None, {'input': values.numpy()})[0]
    with torch.no_grad(): expected = derived(values).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_saved_aten_copy_exports_as_equivalent_typed_copy_without_changing_source(tmp_path):
    values=torch.randn(1,3,128);encoder=EncoderWithCopy().eval()
    source=torch.export.export(encoder,(values,)).module();before=str(source.graph)
    derived,report=module().rewrite_welford_seed(source)
    path=tmp_path/'copy.onnx';export_audio_native_onnx(derived,values,path)
    actual=ReferenceEvaluator(onnx.load(path)).run(None,{'input':values.numpy()})[0]
    with torch.no_grad():expected=derived(values).numpy()
    np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-5)
    assert str(source.graph)==before
    assert report['export_equivalent_copy_count']==1


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64,torch.int64,torch.bool])
def test_legacy_copy_bridge_keeps_cast_values_and_distinct_storage(dtype):
    values=torch.tensor([0,1,0,1],dtype=dtype)
    before=values.clone()
    actual=module()._typed_copy_for_export(values,dtype=torch.float32)
    expected=torch.ops.aten._to_copy.default(values,dtype=torch.float32)
    assert torch.equal(actual,expected) and actual.dtype==torch.float32
    assert actual.data_ptr()!=values.data_ptr() and torch.equal(values,before)


def test_privacy_audit_removes_only_optional_metadata_and_rejects_private_node_names():
    m = module()
    model = onnx.helper.make_model(onnx.helper.make_graph([], 'empty', [], []))
    model.doc_string = 'optional debug text'
    model.metadata_props.add(key='debug', value='optional source context')
    sanitized, report = m.audit_seed_metadata(model)
    assert sanitized.doc_string == '' and not sanitized.metadata_props
    assert report['optional_metadata_fields_removed'] == 2
    assert report['computation_unchanged'] is True
    assert model.doc_string == 'optional debug text'
    private = onnx.helper.make_node('Identity', ['x'], ['y'], name='/'.join(('C:', 'Users', 'private-person', 'source.py')))
    model.graph.node.append(private)
    with pytest.raises(ValueError, match='metadata'): m.audit_seed_metadata(model)


def test_preparation_receipt_binds_actual_export_versions_and_keeps_acceptance_false(saved, tmp_path, monkeypatch):
    _, values, source = saved
    from embeddinggemma_tools import audio_generation_candidate as candidate
    monkeypatch.setattr(candidate, 'load_candidate_input', lambda *a, **k: (source,values,None,{'graph_sha256':'b'*64}))
    inputs=tmp_path/'input';original=tmp_path/'original';output=tmp_path/'output'
    inputs.mkdir();original.mkdir()
    report=module().prepare_seed(inputs,original,output,case_id='silence',source='a'*40)
    assert report['success'] is True
    assert report['versions']=={'torch':str(torch.__version__),'onnx':onnx.__version__}
    assert report['onnx_sha256']==hashlib.sha256((output/'audio.onnx').read_bytes()).hexdigest()
    assert report['source_tensor_hashes_preserved_after_export'] is True
    assert report['gpu_validated'] is False and report['numeric_acceptance'] is False
    assert report['generation_qualified'] is False
    assert json.loads((output/'audio-gpu-seed.json').read_text())==report


def test_fetch_rejects_occupied_output_before_network_or_deserialization(tmp_path, monkeypatch):
    from embeddinggemma_tools import audio_generation_candidate as candidate
    output=tmp_path/'occupied';output.mkdir();marker=output/'existing.txt';marker.write_text('keep')
    calls=[]
    monkeypatch.setattr(candidate,'fetch_candidate',lambda *a,**k:calls.append('network'))
    monkeypatch.setattr(candidate,'load_candidate_input',lambda *a,**k:pytest.fail('model was deserialized'))
    result=module().main(['--input',str(tmp_path/'input'),'--original',str(tmp_path/'original'),
        '--output',str(output),'--source','a'*40,'--case','silence','--fetch'])
    assert result==1 and not calls and marker.read_text()=='keep'


def test_failed_tracing_retains_a_failed_receipt_without_raw_error_text(saved, tmp_path, monkeypatch):
    _,values,source=saved
    from embeddinggemma_tools import audio_generation_candidate as candidate
    m=module();monkeypatch.setattr(candidate,'load_candidate_input',lambda *a,**k:(source,values,None,{}))
    def fail(*a,**k): raise RuntimeError('synthetic tracing failure')
    monkeypatch.setattr(m,'export_audio_native_onnx',fail)
    with pytest.raises(RuntimeError):m.prepare_seed(tmp_path/'input',tmp_path/'original',tmp_path/'out',case_id='silence',source='a'*40)
    report=json.loads((tmp_path/'out/audio-gpu-seed.json').read_text())
    assert report['success'] is False and report['error_type']=='RuntimeError'
    assert 'synthetic tracing failure' not in json.dumps(report)
