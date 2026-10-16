import hashlib

import numpy as np
import pytest
import torch

from embeddinggemma_tools import reference_context_failure as context


class TwoStages(torch.nn.Module):
    def forward(self, values):
        first = values + 1
        return first * 2


def capture(*args, **kwargs):
    method = getattr(context, 'capture_saved_program_nodes', None)
    assert callable(method), 'Original saved-program node capture is missing'
    return method(*args, **kwargs)


def program():
    values = torch.tensor([[1., -2.]])
    return torch.export.export(TwoStages(), (values,)).module(), values


def test_captures_requested_saved_nodes_without_changing_graph_inputs_or_final_bits(tmp_path):
    module, values = program()
    code, original = module.code, values.clone()
    expected = module(values)
    report = capture(module, (values,), ('add', 'mul'), tmp_path, expected)
    assert module.code == code and torch.equal(values, original)
    assert report['observed_final_matches_normal_bits'] is True
    assert report['original_golden_metrics']['mismatched_elements'] == 0
    assert report['full_encoder_validated'] is False
    assert report['end_to_end_validated'] is False
    for row, target in zip(report['nodes'], (values+1, expected), strict=True):
        path = tmp_path / row['file']
        assert row['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert np.array_equal(np.fromfile(path, '<f4').reshape(row['shape']), target.numpy())


def test_capture_reports_failed_original_golden_without_replacing_it(tmp_path):
    module, values = program()
    golden = module(values) + 1
    original = golden.clone()
    report = capture(module, (values,), ('add',), tmp_path, golden)
    assert report['observed_final_matches_normal_bits'] is True
    assert report['original_golden_metrics']['mismatched_elements'] == 2
    assert torch.equal(golden, original)


def test_capture_row_prefix_limits_only_saved_observation_and_keeps_full_final_bits(tmp_path):
    values=torch.arange(6,dtype=torch.float32).reshape(3,2)
    module=torch.export.export(TwoStages(),(values,)).module()
    report=capture(module,(values,),('add',),tmp_path,module(values),row_prefixes={'add':1})
    row,=report['nodes']
    assert row['source_shape']==[3,2] and row['shape']==[1,2] and row['prefix_rows']==1
    assert row['bytes']==8
    assert np.array_equal(np.fromfile(tmp_path/row['file'],'<f4'),[1.,2.])
    assert report['observed_final_matches_normal_bits'] is True
    assert report['original_golden_metrics']['mismatched_elements']==0


@pytest.mark.parametrize('prefixes',[{'missing':1},{'add':0},{'add':True},{'add':2}])
def test_capture_rejects_invalid_row_prefixes_without_writing_files(tmp_path,prefixes):
    module,values=program()
    with pytest.raises(ValueError,match='prefix'):
        capture(module,(values,),('add',),tmp_path,module(values),row_prefixes=prefixes)
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('names', [('missing',), ('add', 'add'), ('../add',), ()])
def test_capture_rejects_invalid_names_before_execution(tmp_path, names):
    module, values = program()
    with pytest.raises(ValueError, match='node'):
        capture(module, (values,), names, tmp_path, module(values))
    assert list(tmp_path.iterdir()) == []
