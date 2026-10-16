import hashlib
import json

import numpy as np
import pytest
import torch

from test_audio_preprocessor import module


def require_preparation():
    audio = module()
    assert hasattr(audio, "prepare_audio_preprocessor"), "auditable audio preprocessing preparation is missing"
    return audio


def configuration(path):
    path.write_text(json.dumps({"feature_extractor": {"feature_extractor_type": "Gemma4AudioFeatureExtractor",
        "sampling_rate": 16000, "feature_size": 128, "frame_length": 320, "hop_length": 160, "fft_length": 512,
        "mel_floor": 0.001}}))
    return path


def test_preparation_exports_and_validates_all_signals_masks_and_hashes(tmp_path):
    audio = require_preparation()
    config = configuration(tmp_path / "processor_config.json")
    output = tmp_path / "out"
    report = audio.prepare_audio_preprocessor(config, output, num_samples=1024, source_commit="b" * 40)
    assert report["success"] is True and report["case_count"] == 6
    assert report["stage"] == "audio_preprocessing_only"
    assert report["unity_validated"] is False and report["encoders_validated"] is False
    reference = json.loads((output / "audio-reference.json").read_text())
    assert reference["metadata"]["source_commit"] == "b" * 40
    assert reference["metadata"]["processor_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert reference["metadata"]["feature_shape"] == [1, 6, 128]
    assert len(reference["cases"]) == 6
    for name, item in report["files"].items():
        assert (output / name).stat().st_size == item["bytes"]
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == item["sha256"]
    silence = np.fromfile(output / reference["cases"][0]["features"], dtype="<f4")
    np.testing.assert_allclose(silence, np.log(0.001), atol=1e-6, rtol=1e-6)
    assert json.loads((output / "audio-validation.json").read_text()) == report


def test_failed_preparation_clears_stale_audio_success(tmp_path):
    audio = require_preparation()
    output = tmp_path / "out"
    output.mkdir()
    (output / "audio-validation.json").write_text('{"success":true}')
    with pytest.raises(FileNotFoundError):
        audio.prepare_audio_preprocessor(tmp_path / "missing.json", output)
    assert not (output / "audio-validation.json").exists()


def test_wrong_saved_audio_output_is_rejected(tmp_path, monkeypatch):
    audio = require_preparation()
    config = configuration(tmp_path / "processor_config.json")
    class Wrong(torch.nn.Module):
        def forward(self, waveform, sample_mask):
            return torch.zeros(1, 6, 128), torch.ones(1, 6, dtype=torch.int32)
    def corrupt(wrapper, waveform, mask, output):
        torch.export.save(torch.export.export(Wrong(), (waveform, mask)), output)
    monkeypatch.setattr(audio, "export_audio_preprocessor_pt2", corrupt)
    with pytest.raises(ValueError, match="audio preprocessing export mismatch"):
        audio.prepare_audio_preprocessor(config, tmp_path / "out", num_samples=1024)
    assert not (tmp_path / "out/audio-validation.json").exists()


@pytest.mark.parametrize("problem", ["nonbinary_frame_mask", "half_features"])
def test_saved_output_dtype_and_binary_mask_are_audited_before_success(tmp_path, monkeypatch, problem):
    audio = require_preparation()
    config = configuration(tmp_path / "processor_config.json")
    class WrongContract(torch.nn.Module):
        def __init__(self, wrapper):
            super().__init__()
            self.wrapper = wrapper
        def forward(self, waveform, sample_mask):
            features, mask = self.wrapper(waveform, sample_mask)
            return (features, mask * 2) if problem == "nonbinary_frame_mask" else (features.half(), mask)
    def corrupt(wrapper, waveform, mask, output):
        audio._save_program(WrongContract(wrapper), (waveform, mask), output)
    monkeypatch.setattr(audio, "export_audio_preprocessor_pt2", corrupt)
    with pytest.raises(ValueError, match="audio preprocessing export mismatch"):
        audio.prepare_audio_preprocessor(config, tmp_path / "out", num_samples=1024)
    assert not (tmp_path / "out/audio-validation.json").exists()


@pytest.mark.parametrize("key,value", [("frame_length", 640), ("hop_length", 320), ("fft_length", 1024),
                                     ("feature_extractor_type", "OtherExtractor")])
def test_serialized_configuration_cannot_silently_override_or_ignore_fixed_coefficients(tmp_path, key, value):
    audio = require_preparation()
    config = configuration(tmp_path / "processor_config.json")
    settings = json.loads(config.read_text())
    settings["feature_extractor"][key] = value
    config.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="configuration"):
        audio.prepare_audio_preprocessor(config, tmp_path / "out", num_samples=1024)
    assert not (tmp_path / "out/audio-validation.json").exists()


def test_cli_downloads_only_pinned_processor_config_and_records_ci_source(tmp_path, monkeypatch, capsys):
    audio = module()
    assert hasattr(audio, "main"), "audio preprocessing reference CLI is missing"
    downloads, calls = [], []
    def download(*args, **kwargs):
        downloads.append((args, kwargs))
        return str(tmp_path / "processor_config.json")
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return dict(success=True, stage="audio_preprocessing_only", case_count=6, maximum_absolute_error=0.002,
                    unity_validated=False, encoders_validated=False)
    monkeypatch.setattr(audio, "hf_hub_download", download, raising=False)
    monkeypatch.setattr(audio, "prepare_audio_preprocessor", prepare, raising=False)
    monkeypatch.setenv("GITHUB_SHA", "c" * 40)
    audio.main(["--output", str(tmp_path / "out")])
    assert downloads[0][0] == ("google/embeddinggemma-2", "processor_config.json")
    assert downloads[0][1]["revision"] == "914f7f89142e33e77833254d9c9b90c3cef7303b"
    assert calls[0][1]["num_samples"] == 179200
    assert calls[0][1]["source_commit"] == "c" * 40
    assert json.loads(capsys.readouterr().out)["encoders_validated"] is False
