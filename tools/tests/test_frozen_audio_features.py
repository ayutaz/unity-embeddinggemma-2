from types import SimpleNamespace

import pytest
import torch

from embeddinggemma_tools import audio_embeddings


def forward_function():
    function = getattr(audio_embeddings, "official_forward_from_frozen_features", None)
    assert callable(function), "Explicit frozen-feature official backbone reference is missing"
    return function


class OfficialBoundary:
    def get_audio_features(self, *args, **kwargs):
        pytest.fail("Frozen-feature mode must never execute or silently accept a changed encoder")

    def __call__(self, *, input_ids, attention_mask, input_features, input_features_mask):
        output = self.get_audio_features(input_features, input_features_mask, return_dict=True)
        features = output.pooler_output[output.attention_mask]
        return SimpleNamespace(audio_hidden_states=features, last_hidden_state=features.unsqueeze(0))


def inputs():
    return dict(ids=torch.tensor([[1, 11, 11, 2]]), mask=torch.ones(1, 4, dtype=torch.int64),
                input_features=torch.zeros(1, 7, 3), frame_mask=torch.tensor([[1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool),
                frozen_features=torch.arange(8, dtype=torch.float32).reshape(2, 4))


def test_explicit_frozen_boundary_uses_official_forward_and_restores_method():
    core = OfficialBoundary()
    original = core.get_audio_features.__func__
    kwargs = inputs()
    output = forward_function()(core, **kwargs)
    assert torch.equal(output.audio_hidden_states, kwargs["frozen_features"])
    assert core.get_audio_features.__func__ is original
    assert "get_audio_features" not in core.__dict__


def test_failure_restores_existing_instance_override():
    class Broken(OfficialBoundary):
        def __call__(self, **kwargs):
            raise RuntimeError("official backbone failed")
    core = Broken()
    sentinel = lambda *args, **kwargs: None
    core.get_audio_features = sentinel
    with pytest.raises(RuntimeError, match="official backbone failed"):
        forward_function()(core, **inputs())
    assert core.get_audio_features is sentinel


@pytest.mark.parametrize("problem", ["nonfinite", "count", "prefix", "empty"])
def test_invalid_frozen_features_or_mask_never_enter_official_forward(problem):
    kwargs = inputs()
    if problem == "nonfinite": kwargs["frozen_features"][0, 0] = float("nan")
    elif problem == "count": kwargs["frozen_features"] = torch.ones(3, 4)
    elif problem == "prefix": kwargs["frame_mask"] = torch.tensor([[1, 0, 1, 1, 1, 0, 0]], dtype=torch.bool)
    else: kwargs["frame_mask"].zero_()
    with pytest.raises(ValueError, match="Frozen audio"):
        forward_function()(OfficialBoundary(), **kwargs)


def test_official_forward_cannot_replace_the_frozen_features():
    class Wrong(OfficialBoundary):
        def __call__(self, **kwargs):
            output = super().__call__(**kwargs)
            output.audio_hidden_states = output.audio_hidden_states + 0.01
            return output
    core = Wrong()
    with pytest.raises(ValueError, match="Frozen audio"):
        forward_function()(core, **inputs())
    assert "get_audio_features" not in core.__dict__
