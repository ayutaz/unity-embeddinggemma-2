from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from embeddinggemma_tools import frozen_audio_segment_onnx as segment
from embeddinggemma_tools import frozen_audio_chained_onnx as chain
from test_ordered_audio_tanh import fixture

VARIANT='ordered_original_midpoint_bmm'


def test_explicit_midpoint_candidate_changes_only_audited_query_key_bmm_and_keeps_previous_mode():
    source,_,_=fixture();old=segment.SegmentExport(source,segment.PRECISE_TANH)
    new=segment.SegmentExport(source,VARIANT)
    for family in ('tanh_nodes','position_nodes','linear_nodes','bmm_nodes','rms_means','rms_inverses',
                   'silu_nodes','softmax_nodes','sigmoid_nodes','convolution_nodes'):
        assert getattr(new,family)==getattr(old,family)
    assert len(new.midpoint_bmm_nodes)==1 and not getattr(old,'midpoint_bmm_nodes',set())
    qk=next(iter(new.midpoint_bmm_nodes))
    assert tuple(qk.args[0].meta['val'].shape)==(192,12,128)
    assert tuple(qk.args[1].meta['val'].shape)==(192,128,24)
    counts=chain._replacement_counts(new,is_stem=False)
    assert counts['midpoint_attention_bmm_replacements']==1
    assert old.bmm_candidate.__name__=='static_block192_bmm_export'


def test_midpoint_variant_rejects_missing_query_key_contract():
    source,_,_=fixture()
    qk=next(n for n in source.graph.nodes if n.target==torch.ops.aten.bmm.default
            and tuple(n.args[0].meta['val'].shape)==(192,12,128))
    qk.args[0].meta['val']=torch.zeros(192,12,127)
    with pytest.raises(ValueError,match='attention|BMM|bmm|contract'):
        segment.SegmentExport(source,VARIANT)


def test_new_midpoint_cli_still_requires_source_before_loading_model(tmp_path,monkeypatch):
    monkeypatch.delenv('GITHUB_SHA',raising=False)
    with pytest.raises(ValueError,match='provenance'):
        chain.main(['--input',str(tmp_path),'--output',str(tmp_path/'out'),
                    '--frozen-stage-source',str(tmp_path),'--variant',VARIANT])


def test_workflow_midpoint_candidate_requires_explicit_connected_mode_before_download():
    text=(Path(__file__).resolve().parents[2]/'.github/workflows/frozen-audio-prefix-onnx.yml').read_text()
    assert VARIANT in text and 'Midpoint candidate requires connected diagnosis' in text


def test_midpoint_variant_executes_actual_connected_onnx_with_retained_original_target(tmp_path):
    source,values,_=fixture();wrapper=segment.SegmentExport(source,VARIANT)
    path=tmp_path/'midpoint-connected.onnx'
    segment.export_audio_native_onnx(wrapper,values,path)
    graph=onnx.load(path)
    assert any(n.op_type=='Not' for n in graph.graph.node), 'Midpoint sticky condition must be exported'
    actual=segment.run_flat_reference(segment.ReferenceEvaluator(graph),{'input':values.numpy()})[0]
    np.testing.assert_allclose(actual,segment.tensor(source(values)).numpy(),atol=1e-5,rtol=1e-4)
