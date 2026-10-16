import hashlib
import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from embeddinggemma_tools import audio_layer_replay
from test_audio_encoder import audio_models, mask
from test_audio_layer_replay import bundle


NAMES = ('layer0.conv', 'layer0.norm', 'layer0.act', 'layer1.conv',
         'layer1.norm', 'layer1.act', 'input_proj_linear')


def capture():
    function = getattr(audio_layer_replay, 'capture_subsample_parts', None)
    assert callable(function), 'Pinned subsample replay is missing'
    return function


def fixture():
    audio, _ = audio_models()
    features = torch.randn(1, 25, 8, generator=torch.Generator().manual_seed(17))
    frame_mask = mask().bool()
    with torch.no_grad():
        expected, _ = audio.subsample_conv_projection(features, frame_mask)
    return audio.subsample_conv_projection, features, frame_mask, expected


def test_subsample_capture_retains_input_and_exact_pinned_final():
    run = capture()
    module, features, frame_mask, expected = fixture()
    before = features.clone()
    values = run(module, features, frame_mask, expected)
    assert tuple(values) == NAMES
    assert torch.equal(features, before)
    assert torch.equal(values['input_proj_linear'], expected)
    assert values['layer0.conv'].shape == (1, 8, 13, 4)
    assert values['layer0.norm'].shape == (1, 13, 4, 8)


@pytest.mark.parametrize('fail', [False, True])
def test_subsample_capture_restores_hooks_on_success_and_mismatch(fail):
    run = capture()
    module, features, frame_mask, expected = fixture()
    external = module.layer0.conv.register_forward_hook(lambda *args: None)
    before = {name: set(module.get_submodule(name)._forward_hooks) for name in NAMES}
    if fail:
        with pytest.raises(ValueError, match='pinned subsample'):
            run(module, features, frame_mask, expected + 1)
    else:
        run(module, features, frame_mask, expected)
    assert before == {name: set(module.get_submodule(name)._forward_hooks) for name in NAMES}
    external.remove()


def test_subsample_capture_rejects_invalid_mask():
    run = capture()
    module, features, _, expected = fixture()
    with pytest.raises(ValueError):
        run(module, features, torch.ones(1, 24, dtype=torch.bool), expected)


def producer_bundle(tmp_path):
    snapshot, context = bundle(tmp_path)
    module, features, _, _ = fixture()
    with safe_open(snapshot / 'model.safetensors', framework='pt', device='cpu') as weights:
        state = {key: weights.get_tensor(key) for key in weights.keys()}
    state.update({'audio_tower.subsample_conv_projection.' + key: value.contiguous()
                  for key, value in module.state_dict().items()})
    save_file(state, snapshot / 'model.safetensors')
    for name in ['audio-stage-reference.json', 'audio-stage-validation.json']:
        path = context / name
        value = json.loads(path.read_text())
        value['metadata']['input_sha256']['model.safetensors'] = hashlib.sha256((snapshot / 'model.safetensors').read_bytes()).hexdigest()
        path.write_text(json.dumps(value))
    path = tmp_path / 'features.bin'
    features.numpy().astype('<f4').tofile(path)
    return snapshot, context, path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_producer_captures_subsample_only_after_exact_native_context(tmp_path):
    snapshot, context, features, digest = producer_bundle(tmp_path)
    report = audio_layer_replay.prepare(snapshot, context, tmp_path / 'out', reference_source='a' * 40,
                                       diagnostic_source='c' * 40, subsample_input=features,
                                       subsample_input_sha256=digest)
    assert report['subsample_native_final_exact_match']
    assert report['subsample_input']['sha256'] == digest
    assert len(report['parts']) == 14
    assert all('subsample.' + name in report['parts'] for name in NAMES)
    assert not report['full_encoder_validated']


def test_producer_rejects_wrong_feature_hash_before_output(tmp_path):
    snapshot, context, features, _ = producer_bundle(tmp_path)
    with pytest.raises(ValueError, match='subsample input'):
        audio_layer_replay.prepare(snapshot, context, tmp_path / 'out', reference_source='a' * 40,
                                  diagnostic_source='c' * 40, subsample_input=features,
                                  subsample_input_sha256='0' * 64)
    assert not (tmp_path / 'out').exists()


def test_cli_passes_pinned_subsample_input_and_hash(tmp_path, monkeypatch):
    seen = {}
    def prepare(*args, **kwargs):
        seen.update(kwargs)
        return {'success': True}
    monkeypatch.setattr(audio_layer_replay, 'prepare', prepare)
    assert audio_layer_replay.main(['--snapshot', str(tmp_path), '--context', str(tmp_path),
                                   '--output', str(tmp_path / 'out'), '--report', str(tmp_path / 'report.json'),
                                   '--reference-source', 'a' * 40, '--diagnostic-source', 'c' * 40,
                                   '--subsample-input', str(tmp_path / 'features.bin'),
                                   '--subsample-input-sha256', 'd' * 64]) == 0
    assert seen['subsample_input'] == tmp_path / 'features.bin'
    assert seen['subsample_input_sha256'] == 'd' * 64
