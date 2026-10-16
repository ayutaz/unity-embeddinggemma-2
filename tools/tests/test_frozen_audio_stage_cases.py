import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from embeddinggemma_tools import frozen_audio_prefix_onnx as probe
from test_frozen_audio_prefix_onnx import frozen_fixture
from test_frozen_audio_case_loading import CASES, FULL, PADDED


def fixture(tmp_path, monkeypatch, *, stage='projected_features', invalid_case=None, duplicate=False):
    directory, _ = frozen_fixture(tmp_path, monkeypatch, stage=stage)
    ref = directory / 'audio-stage-reference.json'; audit_path = directory / 'audio-stage-validation.json'
    r, a = json.loads(ref.read_text()), json.loads(audit_path.read_text())
    template = r['cases'][0]; r['cases'] = []
    for index, name in enumerate(CASES):
        row = copy.deepcopy(template); profile = PADDED if name == 'padded_noise' else FULL
        row.update(id=name, profile=profile, input_features=f'{profile}/{name}-features-fp32.bin')
        shape = ([210 if profile == PADDED else 280, 512] if stage == 'projected_features' else [1, 280, 1024])
        path = directory / profile / f'{name}-stage-fp32.bin'; path.parent.mkdir(exist_ok=True)
        np.full(shape, index + .125, dtype='<f4').tofile(path)
        relative = path.relative_to(directory).as_posix()
        row['stages'][0].update(shape=shape, expected=relative)
        if name == invalid_case: row['native_final_matches_original'] = False
        r['cases'].append(row)
        a['files'][relative] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if duplicate: r['cases'].append(copy.deepcopy(r['cases'][1]))
    ref.write_text(json.dumps(r), encoding='utf-8'); audit_path.write_text(json.dumps(a), encoding='utf-8')
    monkeypatch.setattr(probe, 'STAGE_REFERENCE_SHA', probe.file_digest(ref))
    monkeypatch.setattr(probe, 'STAGE_AUDIT_SHA', probe.file_digest(audit_path))
    return directory


@pytest.mark.parametrize('case_id', CASES)
def test_all_original_final_stage_cases_use_their_own_audited_values_and_padded_count(tmp_path, monkeypatch, case_id):
    root = fixture(tmp_path, monkeypatch)
    result = probe.load_frozen_stage_reference(root, 'projected_features', case_id=case_id)
    assert result.expected.shape == (210 if case_id == 'padded_noise' else 280, 512)
    assert torch.equal(result.expected, torch.full_like(result.expected, CASES.index(case_id) + .125))
    assert result.provenance['case_id'] == case_id
    assert result.provenance['profile'] == (PADDED if case_id == 'padded_noise' else FULL)


def test_padded_intermediate_stage_still_has280rows_before_final_crop(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch, stage='layer_0')
    result = probe.load_frozen_stage_reference(root, 'layer_0', case_id='padded_noise')
    assert result.expected.shape == (1, 280, 1024)


def test_selected_case_context_failure_is_not_replaced_by_qualified_silence(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch, invalid_case='noise')
    with pytest.raises(ValueError, match='context'):
        probe.load_frozen_stage_reference(root, 'projected_features', case_id='noise')


@pytest.mark.parametrize('duplicate,name', [(False, 'unknown'), (True, 'impulse')])
def test_unknown_or_duplicate_stage_case_fails_without_fallback(tmp_path, monkeypatch, duplicate, name):
    root = fixture(tmp_path, monkeypatch, duplicate=duplicate)
    with pytest.raises(ValueError, match='context|case'):
        probe.load_frozen_stage_reference(root, 'projected_features', case_id=name)


def test_selected_non_silence_stage_hash_is_checked(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch)
    path = root / FULL / 'impulse-stage-fp32.bin'; raw = bytearray(path.read_bytes()); raw[4] ^= 1; path.write_bytes(raw)
    with pytest.raises(ValueError, match='hash'):
        probe.load_frozen_stage_reference(root, 'projected_features', case_id='impulse')


def graph(*, crop=False, end=210, return_crop=True):
    g = torch.fx.Graph(); x = g.placeholder('x')
    selected = g.call_function(torch.ops.aten.select.int, (x, 0, 0), name='select')
    final = g.call_function(torch.ops.aten.slice.Tensor, (selected, 0, 0, end), name='slice_51') if crop else selected
    g.output((final if return_crop else selected,))
    return torch.fx.GraphModule(torch.nn.Module(), g)


def test_final_boundary_respects_pinned_full_and_padded_original_graph_outputs():
    assert probe.boundary_for_stage(graph(), 'projected_features') == 'select'
    assert probe.boundary_for_stage(graph(crop=True), 'projected_features', profile=PADDED) == 'slice_51'


@pytest.mark.parametrize('kwargs', [dict(crop=False), dict(crop=True, end=209), dict(crop=True, return_crop=False)])
def test_padded_boundary_rejects_missing_wrong_or_bypassed_original_crop(kwargs):
    with pytest.raises(ValueError, match='boundary|crop'):
        probe.boundary_for_stage(graph(**kwargs), 'projected_features', profile=PADDED)
