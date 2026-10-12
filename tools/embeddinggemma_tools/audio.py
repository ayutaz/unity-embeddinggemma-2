"""Fixed-profile, real-valued export of the pinned Gemma4 audio preprocessing."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
import torch
from torch import nn
from transformers import Gemma4AudioFeatureExtractor
from huggingface_hub import hf_hub_download

from .export import _save_program
from .prepare import MODEL_ID, MODEL_REVISION

FEATURE_ATOL = 0.004
FEATURE_RTOL = 1e-4


class AudioPreprocessorExport(nn.Module):
    """16 kHz mono PCM → log-magnitude HTK mel and int32 frame mask.

    A radix-two FFT uses real tensors, fixed gathers and ordinary arithmetic,
    avoiding complex / FFT operators unavailable in the Torch import path.
    Input sample masks describe a nonempty real prefix and right zero padding.
    Eager validation is outside the saved numerical graph.
    """

    def __init__(self, num_samples, extractor=None):
        super().__init__()
        if type(num_samples) is not int or num_samples < 161:
            raise ValueError("audio profile must contain at least 161 samples")
        extractor = Gemma4AudioFeatureExtractor() if extractor is None else extractor
        expected = dict(sampling_rate=16000, feature_size=128, frame_length=320, hop_length=160,
                        fft_length=512, min_frequency=0.0, max_frequency=8000.0, padding_value=0.0,
                        preemphasis=0.0, dither=0.0, input_scale_factor=1.0, mel_floor=0.001)
        if (any(getattr(extractor, name) != value for name, value in expected.items())
                or extractor.per_bin_mean is not None or extractor.per_bin_stddev is not None):
            raise ValueError("audio processor configuration differs from the fixed supported reference")
        self.num_samples = num_samples
        self.num_frames = (num_samples - 161) // 160 + 1
        frame_start = torch.arange(self.num_frames) * 160
        self.register_buffer("frame_indices", (frame_start[:, None] + torch.arange(320)).flatten())
        self.register_buffer("frame_ends", frame_start + 320)
        self.register_buffer("sample_positions", torch.arange(num_samples).unsqueeze(0))
        self.register_buffer("window", torch.from_numpy(extractor.window).float())
        self.register_buffer("mel_filters", torch.from_numpy(extractor.mel_filters).float())
        self.register_buffer("bit_reversal", torch.tensor([int(f"{index:09b}"[::-1], 2) for index in range(512)]))
        for level in range(1, 10):
            size = 2**level
            angle = -2 * torch.pi * torch.arange(size // 2, dtype=torch.float64) / size
            self.register_buffer(f"cos_{level}", angle.cos().float())
            self.register_buffer(f"sin_{level}", angle.sin().float())

    def validate_inputs(self, waveform, sample_mask):
        if waveform.dtype != torch.float32 or waveform.shape != (1, self.num_samples):
            raise ValueError("waveform must match the batch-one float32 sample profile")
        if not torch.isfinite(waveform).all() or torch.any(waveform.abs() > 1):
            raise ValueError("waveform must contain finite PCM amplitudes in [-1, 1]")
        if sample_mask.dtype != torch.int64 or sample_mask.shape != waveform.shape:
            raise ValueError("sample mask must match the waveform shape and be int64")
        if waveform.device != self.window.device or sample_mask.device != waveform.device:
            raise ValueError("waveform, mask and profile must be on the same device")
        if not torch.all((sample_mask == 0) | (sample_mask == 1)):
            raise ValueError("sample mask must be binary")
        valid = sample_mask.sum()
        if valid < 161 or not torch.equal(sample_mask.bool(), self.sample_positions < valid):
            raise ValueError("sample mask must describe a real prefix of at least 161 samples")
        if torch.any(waveform[sample_mask == 0] != 0):
            raise ValueError("waveform must be zero in sample mask padding")

    def forward(self, waveform, sample_mask):
        if not torch.compiler.is_compiling():
            self.validate_inputs(waveform, sample_mask)
        padded = torch.nn.functional.pad(waveform, (160, 0))
        frames = padded.index_select(1, self.frame_indices).reshape(1, self.num_frames, 320) * self.window
        real = torch.nn.functional.pad(frames, (0, 192)).index_select(-1, self.bit_reversal)
        imaginary = torch.zeros_like(real)
        for level in range(1, 10):
            size = 2**level
            half = size // 2
            real_blocks = real.reshape(1, self.num_frames, 512 // size, size)
            imaginary_blocks = imaginary.reshape(1, self.num_frames, 512 // size, size)
            a_real, b_real = real_blocks[..., :half], real_blocks[..., half:]
            a_imaginary, b_imaginary = imaginary_blocks[..., :half], imaginary_blocks[..., half:]
            cosine, sine = getattr(self, f"cos_{level}"), getattr(self, f"sin_{level}")
            rotated_real = b_real * cosine - b_imaginary * sine
            rotated_imaginary = b_real * sine + b_imaginary * cosine
            real = torch.cat((a_real + rotated_real, a_real - rotated_real), dim=-1).reshape(1, self.num_frames, 512)
            imaginary = torch.cat((a_imaginary + rotated_imaginary, a_imaginary - rotated_imaginary), dim=-1).reshape(
                1, self.num_frames, 512)
        magnitude = (real[..., :257].square() + imaginary[..., :257].square()).sqrt()
        features = (magnitude @ self.mel_filters + 0.001).log()
        padded_mask = torch.nn.functional.pad(sample_mask, (160, 0))
        frame_mask = padded_mask.index_select(1, self.frame_ends)
        return features * frame_mask.unsqueeze(-1), frame_mask.to(torch.int32)


def export_audio_preprocessor_pt2(wrapper, waveform, sample_mask, destination: Path):
    wrapper.validate_inputs(waveform, sample_mask)
    _save_program(wrapper.eval(), (waveform, sample_mask), destination)


def prepare_audio_preprocessor(processor_config, destination, *, num_samples=179200,
                               revision=MODEL_REVISION, source_commit="local"):
    processor_config, destination = Path(processor_config), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "audio-validation.json").unlink(missing_ok=True)
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("model revision must be an immutable lowercase commit SHA")
    settings = json.loads(processor_config.read_text(encoding="utf-8"))["feature_extractor"]
    serialized = dict(feature_extractor_type="Gemma4AudioFeatureExtractor", frame_length=320, hop_length=160,
                      fft_length=512)
    if any(settings.get(name) != value for name, value in serialized.items()):
        raise ValueError("serialized audio configuration differs from the supported reference")
    extractor = Gemma4AudioFeatureExtractor(**{name: value for name, value in settings.items()
                                              if name not in serialized})
    wrapper = AudioPreprocessorExport(num_samples, extractor)
    t = torch.arange(num_samples).float() / 16000
    signal_table = dict(
        silence=torch.zeros(1, num_samples),
        impulse=torch.nn.functional.one_hot(torch.tensor(min(170, num_samples - 1)), num_samples).float()[None] * 0.8,
        sine=(0.7 * torch.sin(2 * torch.pi * 440 * t))[None],
        nyquist=(torch.arange(num_samples).remainder(2).float() * 0.6 - 0.3)[None],
        noise=torch.rand(1, num_samples, generator=torch.Generator().manual_seed(17)) * 1.6 - 0.8)
    signal_table["padded_noise"] = signal_table["noise"].clone()
    inputs = []
    for case_id, samples in signal_table.items():
        valid_length = max(161, num_samples * 3 // 4) if case_id == "padded_noise" else num_samples
        sample_mask = (torch.arange(num_samples)[None] < valid_length).to(torch.int64)
        inputs.append((case_id, samples * sample_mask, sample_mask))
    graph = destination / "audio-preprocessor.pt2"
    export_audio_preprocessor_pt2(wrapper, inputs[0][1], inputs[0][2], graph)
    exported = torch.export.load(graph).module()
    rows, produced, errors = [], [graph], []
    with torch.no_grad():
        for case_id, samples, sample_mask in inputs:
            raw, mask = extractor._extract_spectrogram(samples[0].numpy(), sample_mask[0].numpy())
            expected = torch.from_numpy(raw.astype(np.float32) * mask[:, None])[None]
            expected_mask = torch.from_numpy(mask)[None]
            actual, actual_mask = exported(samples, sample_mask)
            if (actual.dtype != torch.float32 or actual_mask.dtype != torch.int32
                    or actual.shape != expected.shape or actual_mask.shape != expected_mask.shape
                    or not torch.isfinite(actual).all()
                    or not torch.equal(actual_mask, expected_mask.to(torch.int32))
                    or not torch.allclose(actual, expected, rtol=FEATURE_RTOL, atol=FEATURE_ATOL)):
                raise ValueError(f"audio preprocessing export mismatch: {case_id}")
            error = (actual - expected).abs().max().item()
            errors.append(error)
            paths = {name: destination / f"{case_id}-{suffix}.bin" for name, suffix in (
                ("waveform", "pcm-fp32"), ("sample_mask", "samples-int32"),
                ("features", "features-fp32"), ("frame_mask", "frames-int32"))}
            for name, tensor, dtype in (("waveform", samples, "<f4"), ("sample_mask", sample_mask, "<i4"),
                                        ("features", expected, "<f4"), ("frame_mask", expected_mask, "<i4")):
                tensor.numpy().astype(dtype).tofile(paths[name])
                produced.append(paths[name])
            rows.append(dict(id=case_id, valid_samples=int(sample_mask.sum()), maximum_absolute_error=error,
                             **{name: path.name for name, path in paths.items()}))
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = dict(model_id=MODEL_ID, model_revision=revision, source_commit=source_commit,
                    stage="audio_preprocessing_only", num_samples=num_samples, sampling_rate=16000,
                    feature_shape=[1, wrapper.num_frames, 128], sample_shape=[1, num_samples],
                    frame_mask_shape=[1, wrapper.num_frames], processor_sha256=digest(processor_config),
                    feature_rtol=FEATURE_RTOL, feature_atol=FEATURE_ATOL,
                    fixture_provenance="procedurally_generated_pcm_no_user_audio",
                    waveform_encoding="little_endian_float32", mask_encoding="little_endian_int32",
                    reference="official_Gemma4AudioFeatureExtractor",
                    versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")})
    reference_path = destination / "audio-reference.json"
    reference_path.write_text(json.dumps(dict(metadata=metadata, cases=rows), indent=2, allow_nan=False) + "\n",
                              encoding="utf-8")
    config_copy = destination / "processor_config.json"
    shutil.copyfile(processor_config, config_copy)
    produced.extend((reference_path, config_copy))
    report = dict(success=True, stage=metadata["stage"], case_count=len(rows), unity_validated=False,
                  encoders_validated=False, maximum_absolute_error=max(errors), metadata=metadata,
                  files={path.name: dict(bytes=path.stat().st_size, sha256=digest(path)) for path in produced})
    (destination / "audio-validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                                      encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processor-config", type=Path, help="Existing pinned processor config; no weights are loaded")
    parser.add_argument("--num-samples", type=int, default=179200)
    parser.add_argument("--revision", default=MODEL_REVISION)
    args = parser.parse_args(argv)
    if re.fullmatch(r"[0-9a-f]{40}", args.revision) is None:
        parser.error("revision must be an immutable lowercase commit SHA")
    config = args.processor_config
    if config is None:
        config = Path(hf_hub_download(MODEL_ID, "processor_config.json", revision=args.revision))
    report = prepare_audio_preprocessor(config, args.output, num_samples=args.num_samples,
                                        revision=args.revision, source_commit=os.environ.get("GITHUB_SHA", "local"))
    print(json.dumps({name: report[name] for name in (
        "success", "stage", "case_count", "maximum_absolute_error", "unity_validated", "encoders_validated")}))


if __name__ == "__main__":
    main()
