"""Weights-free official image processor reference for Unity GPU acceptance."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import torch
from transformers import Gemma4ImageProcessor


def prepare_reference(configuration, destination, *, revision, source_commit, cases=None):
    configuration, destination = Path(configuration), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    receipt = destination / "image-preprocessing-reference.json"
    receipt.unlink(missing_ok=True)
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None
           for value in (revision, source_commit)):
        raise ValueError("model and source require immutable lowercase commit SHAs")
    settings = json.loads(configuration.read_text(encoding="utf-8"))["image_processor"]
    required = dict(image_processor_type="Gemma4ImageProcessor", do_convert_rgb=True,
                    do_resize=True, do_rescale=True, do_normalize=False, resample=3,
                    rescale_factor=1 / 255, patch_size=16, pooling_kernel_size=3, max_soft_tokens=280)
    if any(settings.get(key) != value for key, value in required.items()):
        raise ValueError("processor must use the pinned 16px / 280-token RGB bicubic profile")
    cases = {"square": (96, 96), "landscape": (96, 160), "portrait": (160, 96),
             "odd": (97, 161), "downsample": (801, 1335), "identity": (768, 768)} if cases is None else cases
    if (not isinstance(cases, dict) or not cases or any(
            not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_-]*", name) is None
            or not isinstance(size, (tuple, list)) or len(size) != 2
            or any(type(side) is not int or side <= 0 or side > 8192 for side in size)
            for name, size in cases.items())):
        raise ValueError("cases require safe names and two positive image dimensions up to 8192")
    processor = Gemma4ImageProcessor(**{key: value for key, value in settings.items()
                                     if key not in ("image_processor_type", "image_seq_length")})
    files, rows = {}, []
    def write(name, data):
        path = destination / name
        path.write_bytes(data)
        files[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    for name, (height, width) in cases.items():
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        raw = torch.stack((x * 7 % 256, y * 11 % 256, (x + y) * 5 % 256)).to(torch.uint8)
        inputs = processor(images=raw, return_tensors="pt")
        pixels = inputs["pixel_values"]
        positions = inputs["image_position_ids"]
        actual = positions[0][positions[0, :, 0] >= 0]
        write(name + "-rgb.bin", raw.permute(1, 2, 0).contiguous().numpy().tobytes())
        write(name + "-pixels.bin", pixels.numpy().astype("<f4").tobytes())
        write(name + "-positions.bin", positions.numpy().astype("<i4").tobytes())
        rows.append(dict(id=name, width=width, height=height,
            resized_width=(int(actual[:, 0].max()) + 1) * 16,
            resized_height=(int(actual[:, 1].max()) + 1) * 16,
            soft_tokens=int(inputs["num_soft_tokens_per_image"].item()), pixel_shape=list(pixels.shape),
            rgb_file=name + "-rgb.bin", pixels_file=name + "-pixels.bin", positions_file=name + "-positions.bin"))
    write("processor_config.json", configuration.read_bytes())
    report = dict(success=True, stage="image_preprocessing_only", weights_loaded=False,
        unity_validated=False, encoder_validated=False, end_to_end_validated=False,
        metadata=dict(model_revision=revision, source_commit=source_commit,
            processor_sha256=hashlib.sha256(configuration.read_bytes()).hexdigest(),
            provenance="Generated RGB coordinate pattern; no collected user images.",
            versions={name: importlib.metadata.version(name) for name in ("torch", "torchvision", "transformers")},
            tolerance=dict(atol=1e-6, rtol=0)), cases=rows, files=files)
    receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
