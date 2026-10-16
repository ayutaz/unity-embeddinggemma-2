import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest
import torch

from embeddinggemma_tools import export


def tail():
    model = getattr(export, 'VisionTailRmsExport', None)
    assert callable(model), 'Unscaled final vision RMS export is missing'
    return model()


def values():
    return torch.stack((torch.arange(768, dtype=torch.float32)*2.5,
                        torch.full((768,), -1200.), torch.zeros(768)))


def expected(data):
    return data * torch.pow(data.square().mean(-1, keepdim=True)+1e-6, -.5)


def test_final_unscaled_rms_matches_native_fp32_bits_above_existing_statistics_domain():
    data = values()
    before = data.clone()
    actual = tail()(data)
    np.testing.assert_array_equal(actual.numpy().view(np.int32), expected(data).numpy().view(np.int32))
    assert torch.equal(data, before) and actual.dtype == torch.float32


def test_tail_export_uses_only_standard_fp32_ops_and_keeps_exact_native_bits(tmp_path):
    data = values()
    path = tmp_path/'tail-rms.onnx'
    export.export_audio_native_onnx(tail(), data, path)
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    assert all(node.domain in ('', 'ai.onnx') for node in graph.graph.node)
    assert all(value.data_type != onnx.TensorProto.DOUBLE for value in graph.graph.initializer)
    actual = ReferenceEvaluator(graph).run(None, {'input': data.numpy()})[0]
    np.testing.assert_array_equal(actual.view(np.int32), expected(data).numpy().view(np.int32))


@pytest.mark.parametrize('data', [torch.zeros(1,64), torch.full((1,768),2000.),
                                torch.full((1,768),float('nan'))])
def test_tail_export_rejects_unproven_shape_finite_and_statistic_domain(data):
    with pytest.raises(ValueError):
        tail()(data)


def test_extended_inverse_domain_is_explicit_and_keeps_existing_default_limit():
    data = torch.tensor([151862.328125, 1576229.625, 2e6])
    with pytest.raises(ValueError, match='bounded'):
        export.fp32_refined_rstd(data)
    actual = export.fp32_refined_rstd(data, maximum_statistic=2e6)
    roots = np.sqrt(data.numpy().astype(np.float64)).astype(np.float32)
    target = (1/roots.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(actual.numpy().view(np.int32), target.view(np.int32))
    with pytest.raises(ValueError, match='bounded'):
        export.fp32_refined_rstd(torch.tensor([2e6+1]), maximum_statistic=2e6)
