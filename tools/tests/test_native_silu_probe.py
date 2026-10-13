import json

import onnx
import pytest
import torch

from embeddinggemma_tools import __main__ as cli


def preparation():
    function=getattr(cli,"prepare_silu_probe",None)
    assert callable(function),"Native SiLU and standard Swish probe preparation is missing."
    return function


@pytest.mark.parametrize("case_index",[0,1])
def test_native_saved_graph_keeps_silu_and_matches_changed_vector_inputs_exactly(tmp_path,case_index):
    prepare=preparation()
    result=prepare(tmp_path,source_commit="a"*40)
    assert result["success"] is True and result["unity_validated"] is False
    reference=json.loads((tmp_path/"silu-reference.json").read_text(encoding="utf-8"))
    case=reference["cases"][case_index]
    program=torch.export.load(tmp_path/case["native_graph"])
    targets={str(node.target) for node in program.graph.nodes if node.op=='call_function'}
    assert 'aten.silu.default' in targets
    assert not any('sigmoid' in target or target.startswith('aten.exp.') for target in targets)
    values=torch.tensor(case["input"],dtype=torch.float32).reshape(case["shape"])*0.7
    assert torch.equal(program.module()(values),torch.nn.functional.silu(values))
    model=onnx.load(tmp_path/case["onnx_graph"])
    onnx.checker.check_model(model)
    assert len(model.graph.node)==1 and model.graph.node[0].op_type=='Swish'
    assert model.opset_import[0].version==24
    assert case["native_saved_exact"] is True
    original = torch.tensor(case["input"], dtype=torch.float32).reshape(case["shape"])
    native = torch.nn.functional.silu(original)
    div_exp = original / (1 + torch.exp(-original))
    # Diagnostic counts depend on the platform math kernels. Equality for one
    # fixture is valid and does not weaken the saved-native exactness above.
    assert case["div_exp_exact_mismatches"] == int((div_exp != native).sum())
    assert case["div_exp_maximum_absolute_error"] == (div_exp - native).abs().max().item()


@pytest.mark.parametrize("value",["local","main"])
def test_mutable_source_cannot_leave_stale_probe_success(tmp_path,value):
    prepare=preparation()
    receipt=tmp_path/"silu-validation.json";receipt.write_text('{"success":true}')
    with pytest.raises(ValueError,match="immutable"):
        prepare(tmp_path,source_commit=value)
    assert not receipt.exists()


def test_cli_probe_never_downloads_weights_or_tokenizers(tmp_path,monkeypatch,capsys):
    preparation()
    def download(*args,**kwargs):pytest.fail("A scalar activation probe must not download model assets")
    def prepare(*args,**kwargs):return dict(success=True,stage="native_silu_swish_probe_only",case_count=2,unity_validated=False)
    monkeypatch.setattr(cli,"snapshot_download",download)
    monkeypatch.setattr(cli,"prepare_silu_probe",prepare)
    monkeypatch.setenv("GITHUB_SHA","b"*40)
    cli.main(["prepare-silu-probe","--output",str(tmp_path)])
    assert json.loads(capsys.readouterr().out)["unity_validated"] is False
