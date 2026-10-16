import hashlib
import json
import operator
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embeddinggemma_tools import audio_native_norm_replay as source

CASES = ('silence', 'impulse', 'sine', 'nyquist', 'noise', 'padded_noise')
FULL, PADDED = 'frames-1119-valid-1119', 'frames-1119-valid-839'


def fixture(root, monkeypatch, *, duplicate=False):
    owner = torch.nn.Module()
    owner.register_buffer('scale128', torch.ones(128)); owner.register_buffer('scale32', torch.ones(32))
    graph = torch.fx.Graph(); x = graph.placeholder('x')
    norm = graph.call_function(torch.ops.aten.native_layer_norm.default,
        (x, [128], graph.get_attr('scale128'), None, 1e-6))
    first = graph.call_function(operator.getitem, (norm, 0))
    sliced = graph.call_function(operator.getitem, (first, (..., slice(None, 32))))
    norm = graph.call_function(torch.ops.aten.native_layer_norm.default,
        (sliced, [32], graph.get_attr('scale32'), None, 1e-6))
    result = graph.call_function(operator.getitem, (norm, 0)); graph.output(result)
    program = torch.fx.GraphModule(owner, graph)
    metadata = dict(source_commit=source.PRODUCER, model_revision=source.REVISION,
        feature_atol=1e-5, feature_rtol=1e-4, versions={'torch': str(torch.__version__).split('+')[0]})
    reference = dict(metadata=metadata, profiles={}, cases=[])
    audit = dict(metadata=metadata, native_reference_complete=True, files={})
    def entry(name, content):
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
        audit['files'][name] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    for profile, count in ((FULL, 2), (PADDED, 1)):
        entry(f'{profile}/audio.pt2', f'synthetic-{profile}-not-a-real-program'.encode())
        reference['profiles'][profile] = dict(feature_shape=[1, 3, 128], feature_count=count,
            feature_width=512, native_graph=f'{profile}/audio.pt2')
    for i, name in enumerate(CASES):
        profile = PADDED if name == 'padded_noise' else FULL
        entry(f'{profile}/{name}-features-fp32.bin', np.full((1, 3, 128), i + 1, dtype='<f4').tobytes())
        count = reference['profiles'][profile]['feature_count']
        reference['cases'].append(dict(id=name, profile=profile, input_features=f'{profile}/{name}-features-fp32.bin',
            features=np.full((count, 512), i + .5, dtype=np.float32).tolist()))
    if duplicate: reference['cases'].append(dict(reference['cases'][1]))
    for filename, data, constant in (('audio-encoder-reference.json', reference, 'REFERENCE_SHA'),
                                      ('audio-encoder-validation.json', audit, 'AUDIT_SHA')):
        path = root / filename; path.write_text(json.dumps(data), encoding='utf-8')
        monkeypatch.setattr(source, constant, source.digest(path))
    loads = []
    def load(path):
        loads.append(path)
        return SimpleNamespace(module=lambda: program)
    monkeypatch.setattr(torch.export, 'load', load)
    return reference, audit, loads


@pytest.mark.parametrize('case_id', CASES)
def test_each_original_case_selects_own_pinned_input_graph_profile_and_expected(tmp_path, monkeypatch, case_id):
    reference, audit, loads = fixture(tmp_path, monkeypatch)
    _, values, expected, provenance = source.load_frozen_input(tmp_path, case_id=case_id)
    i = CASES.index(case_id); row = reference['cases'][i]
    assert torch.equal(values, torch.full((1, 3, 128), float(i + 1)))
    assert expected.shape == (1 if case_id == 'padded_noise' else 2, 512)
    assert torch.equal(expected, torch.full_like(expected, i + .5))
    assert provenance['case_id'] == case_id and provenance['profile'] == row['profile']
    assert provenance['input_sha256'] == audit['files'][row['input_features']]['sha256']
    assert provenance['graph_sha256'] == audit['files'][f"{row['profile']}/audio.pt2"]['sha256']
    assert len(loads) == 1 and loads[0].parent.name == row['profile']


def test_default_still_selects_original_silence(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    _, values, expected, provenance = source.load_frozen_input(tmp_path)
    assert provenance['case_id'] == 'silence' and values[0, 0, 0] == 1 and expected[0, 0] == .5


@pytest.mark.parametrize('name', ['unknown', '', None])
def test_unknown_case_is_rejected_before_graph_deserialization_without_silence_fallback(tmp_path, monkeypatch, name):
    _, _, loads = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='case'):
        source.load_frozen_input(tmp_path, case_id=name)
    assert loads == []


def test_duplicate_case_is_rejected_before_deserialization(tmp_path, monkeypatch):
    _, _, loads = fixture(tmp_path, monkeypatch, duplicate=True)
    with pytest.raises(ValueError, match='case'):
        source.load_frozen_input(tmp_path, case_id='impulse')
    assert loads == []


def test_selected_input_hash_failure_does_not_fall_back_to_silence(tmp_path, monkeypatch):
    _, _, loads = fixture(tmp_path, monkeypatch)
    path = tmp_path / FULL / 'noise-features-fp32.bin'
    content = bytearray(path.read_bytes()); content[4] ^= 1; path.write_bytes(content)
    with pytest.raises(ValueError, match='hash'):
        source.load_frozen_input(tmp_path, case_id='noise')
    assert loads == []


def test_padded_graph_is_audited_independently_of_full_profile(tmp_path, monkeypatch):
    _, _, loads = fixture(tmp_path, monkeypatch)
    path = tmp_path / PADDED / 'audio.pt2'; path.write_bytes(path.read_bytes() + b'changed')
    source.load_frozen_input(tmp_path, case_id='silence')
    assert len(loads) == 1
    with pytest.raises(ValueError, match='hash'):
        source.load_frozen_input(tmp_path, case_id='padded_noise')
    assert len(loads) == 1
