import importlib
import importlib.util

import numpy as np
import pytest
import torch
from transformers import Gemma4AudioFeatureExtractor

FEATURE_ATOL = 0.004  # Pre-implementation numpy vs torch fp32 FFT baseline: max 0.002475.
FEATURE_RTOL = 1e-4


def module():
    name = "embeddinggemma_tools.audio"
    assert importlib.util.find_spec(name) is not None, "exportable audio preprocessing is missing"
    return importlib.import_module(name)


def reference(waveform, sample_mask, extractor=None):
    extractor = Gemma4AudioFeatureExtractor() if extractor is None else extractor
    features, mask = extractor._extract_spectrogram(waveform[0].numpy(), sample_mask[0].numpy())
    return torch.from_numpy(features.astype(np.float32) * mask[:, None]).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)


def waveform(name, length=1024):
    t = torch.arange(length).float() / 16000
    if name == "silence":
        return torch.zeros(1, length)
    if name == "impulse":
        return torch.nn.functional.one_hot(torch.tensor(170), length).float().unsqueeze(0) * 0.8
    if name == "sine":
        return (0.7 * torch.sin(2 * torch.pi * 440 * t)).unsqueeze(0)
    if name == "nyquist":
        return (torch.arange(length).remainder(2).float() * 0.6 - 0.3).unsqueeze(0)
    return (torch.rand(1, length, generator=torch.Generator().manual_seed(17)) * 1.6 - 0.8)


@pytest.mark.parametrize("name", ["silence", "impulse", "sine", "nyquist", "noise"])
def test_real_only_fft_log_magnitude_mel_matches_official_semicausal_preprocessor(name):
    samples = waveform(name)
    sample_mask = torch.ones_like(samples, dtype=torch.int64)
    expected, mask = reference(samples, sample_mask)
    actual, actual_mask = module().AudioPreprocessorExport(1024)(samples, sample_mask)
    torch.testing.assert_close(actual, expected, rtol=FEATURE_RTOL, atol=FEATURE_ATOL)
    assert torch.equal(actual_mask.bool(), mask)
    assert actual_mask.dtype == torch.int32
    if name == "silence":
        torch.testing.assert_close(actual, torch.full_like(actual, np.log(0.001)), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("valid", [161, 319, 320, 321, 479, 480, 481, 801, 1024])
def test_frame_end_mask_and_right_padding_match_official_at_sample_boundaries(valid):
    samples = waveform("noise")
    sample_mask = (torch.arange(1024)[None, :] < valid).to(torch.int64)
    samples = samples * sample_mask
    expected, mask = reference(samples, sample_mask)
    actual, actual_mask = module().AudioPreprocessorExport(1024)(samples, sample_mask)
    torch.testing.assert_close(actual, expected, rtol=FEATURE_RTOL, atol=FEATURE_ATOL)
    assert torch.equal(actual_mask.bool(), mask)
    assert torch.count_nonzero(actual[:, ~mask[0]]) == 0


def test_saved_audio_program_uses_real_tensor_ops_and_handles_changed_waveform_and_mask(tmp_path):
    audio = module()
    instance = audio.AudioPreprocessorExport(1024)
    first = waveform("noise")
    output = tmp_path / "audio-preprocessor.pt2"
    audio.export_audio_preprocessor_pt2(instance, first, torch.ones_like(first, dtype=torch.int64), output)
    changed = waveform("sine")
    sample_mask = (torch.arange(1024)[None, :] < 801).to(torch.int64)
    changed = changed * sample_mask
    expected, expected_mask = reference(changed, sample_mask)
    program = torch.export.load(output)
    actual, actual_mask = program.module()(changed, sample_mask)
    torch.testing.assert_close(actual, expected, rtol=FEATURE_RTOL, atol=FEATURE_ATOL)
    assert torch.equal(actual_mask.bool(), expected_mask)
    assert not any("fft" in str(node.target) or "complex" in str(node.target) or "unfold" in str(node.target)
                   for node in program.graph.nodes if node.op == "call_function")


@pytest.mark.parametrize("problem", ["wrong_shape", "half", "nan", "infinite", "amplitude", "mask_shape",
                                     "mask_dtype", "mask_nonbinary", "mask_hole", "too_short", "nonzero_padding"])
def test_invalid_waveform_or_sample_mask_is_rejected(problem):
    samples = waveform("noise")
    mask = torch.ones_like(samples, dtype=torch.int64)
    if problem == "wrong_shape":
        samples = samples[0]
    elif problem == "half":
        samples = samples.half()
    elif problem == "nan":
        samples[0, 0] = float("nan")
    elif problem == "infinite":
        samples[0, 0] = float("inf")
    elif problem == "amplitude":
        samples[0, 0] = 1.01
    elif problem == "mask_shape":
        mask = mask[:, :1000]
    elif problem == "mask_dtype":
        mask = mask.float()
    elif problem == "mask_nonbinary":
        mask[0, 0] = 2
    elif problem == "mask_hole":
        mask[0, 1] = 0
    elif problem == "too_short":
        mask[:, 160:] = 0
        samples = samples * mask
    elif problem == "nonzero_padding":
        mask[:, 801:] = 0
    with pytest.raises(ValueError, match="waveform|mask"):
        module().AudioPreprocessorExport(1024)(samples, mask)


@pytest.mark.parametrize("length", [0, 160, 1024.0, True])
def test_invalid_fixed_sample_length_is_rejected(length):
    with pytest.raises(ValueError, match="profile"):
        module().AudioPreprocessorExport(length)


@pytest.mark.parametrize("changes", [{"dither": 0.01}, {"preemphasis": 0.97}, {"sampling_rate": 8000},
                                    {"mel_floor": 0.1}, {"per_bin_mean": np.ones(128)}])
def test_unsupported_processor_configuration_is_explicitly_rejected(changes):
    with pytest.raises(ValueError, match="configuration"):
        module().AudioPreprocessorExport(1024, Gemma4AudioFeatureExtractor(**changes))
