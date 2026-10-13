import hashlib
import json

import numpy as np
import pytest
import torch
from transformers import Gemma4ImageProcessor

from embeddinggemma_tools import image_preprocessing


def config(path):
    settings = dict(image_processor_type="Gemma4ImageProcessor", do_convert_rgb=True,
                    do_resize=True, do_rescale=True, do_normalize=False, resample=3,
                    rescale_factor=1 / 255, patch_size=16, pooling_kernel_size=3, max_soft_tokens=280)
    path.write_text(json.dumps(dict(image_processor=settings)), encoding="utf-8")
    return settings


def test_reference_preserves_official_patch_values_positions_and_input_provenance(tmp_path):
    configuration = tmp_path / "processor.json"
    settings = config(configuration)
    report = image_preprocessing.prepare_reference(configuration, tmp_path / "out", revision="a" * 40,
        source_commit="b" * 40, cases={"odd": (17, 29)})
    assert report["success"] is True
    assert report["stage"] == "image_preprocessing_only"
    assert report["weights_loaded"] is False and report["unity_validated"] is False
    row = report["cases"][0]
    y, x = torch.meshgrid(torch.arange(17), torch.arange(29), indexing="ij")
    raw = torch.stack((x * 7 % 256, y * 11 % 256, (x + y) * 5 % 256)).to(torch.uint8)
    np.testing.assert_array_equal(np.fromfile(tmp_path / "out" / row["rgb_file"], dtype=np.uint8)
        .reshape(17, 29, 3), raw.permute(1, 2, 0).numpy())
    expected = Gemma4ImageProcessor(**{k: v for k, v in settings.items() if k != "image_processor_type"})(
        images=raw, return_tensors="pt")
    np.testing.assert_array_equal(np.fromfile(tmp_path / "out" / row["pixels_file"], dtype="<f4")
        .reshape(1, 2520, 768), expected["pixel_values"].numpy())
    np.testing.assert_array_equal(np.fromfile(tmp_path / "out" / row["positions_file"], dtype="<i4")
        .reshape(1, 2520, 2), expected["image_position_ids"].numpy())
    assert row["soft_tokens"] == expected["num_soft_tokens_per_image"].item()
    assert report["metadata"]["source_commit"] == "b" * 40
    assert report["metadata"]["model_revision"] == "a" * 40
    assert report["metadata"]["processor_sha256"] == hashlib.sha256(configuration.read_bytes()).hexdigest()
    assert report["metadata"]["tolerance"] == dict(atol=1e-6, rtol=0)
    for name, evidence in report["files"].items():
        data = (tmp_path / "out" / name).read_bytes()
        assert evidence == dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    assert json.loads((tmp_path / "out/image-preprocessing-reference.json").read_text()) == report


@pytest.mark.parametrize("field,value", [("revision", "main"), ("source_commit", "local")])
def test_mutable_provenance_is_rejected_and_stale_success_removed(tmp_path, field, value):
    configuration = tmp_path / "processor.json"
    config(configuration)
    output = tmp_path / "out"
    output.mkdir()
    stale = output / "image-preprocessing-reference.json"
    stale.write_text('{"success":true}')
    provenance = dict(revision="a" * 40, source_commit="b" * 40)
    provenance[field] = value
    with pytest.raises(ValueError, match="immutable"):
        image_preprocessing.prepare_reference(configuration, output, **provenance)
    assert not stale.exists()


@pytest.mark.parametrize("key,value", [("resample", 2), ("do_normalize", True),
    ("patch_size", 8), ("max_soft_tokens", 70), ("rescale_factor", 1)])
def test_incompatible_processor_is_rejected_before_reference_generation(tmp_path, key, value):
    configuration = tmp_path / "processor.json"
    settings = config(configuration)
    settings[key] = value
    configuration.write_text(json.dumps(dict(image_processor=settings)))
    with pytest.raises(ValueError, match="processor"):
        image_preprocessing.prepare_reference(configuration, tmp_path / "out", revision="a" * 40,
                                             source_commit="b" * 40)


@pytest.mark.parametrize("cases", [{}, {"../escape": (17, 29)}, {"odd": (0, 29)},
    {"odd": (True, 29)}, {"odd": (17, 29, 3)}])
def test_invalid_fixture_geometry_and_paths_are_rejected(tmp_path, cases):
    configuration = tmp_path / "processor.json"
    config(configuration)
    with pytest.raises(ValueError, match="cases"):
        image_preprocessing.prepare_reference(configuration, tmp_path / "out", revision="a" * 40,
                                             source_commit="b" * 40, cases=cases)


def test_processor_failure_never_leaves_success_receipt(tmp_path, monkeypatch):
    configuration = tmp_path / "processor.json"
    config(configuration)
    def fail(*args, **kwargs):
        raise RuntimeError("processor execution failed")
    monkeypatch.setattr(Gemma4ImageProcessor, "__call__", fail)
    with pytest.raises(RuntimeError, match="processor execution failed"):
        image_preprocessing.prepare_reference(configuration, tmp_path / "out", revision="a" * 40,
                                             source_commit="b" * 40, cases={"odd": (17, 29)})
    assert not (tmp_path / "out/image-preprocessing-reference.json").exists()
