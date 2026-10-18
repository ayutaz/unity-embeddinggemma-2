import importlib
import json

import pytest
import torch

from test_vision import positions, vision_models


def preparation_module():
    from embeddinggemma_tools import prepare
    assert hasattr(prepare, "prepare_vision"), "official image processor / vision preparation gate is missing"
    return prepare


def fake_snapshot(path):
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"mock preparation weights only")
    (path / "config.json").write_text("{}")
    (path / "processor_config.json").write_text(json.dumps({"image_processor": {
        "image_processor_type": "Gemma4ImageProcessor", "patch_size": 2,
        "pooling_kernel_size": 3, "max_soft_tokens": 280}}))
    return path


class TinyProcessor:
    def __call__(self, images, return_tensors):
        assert return_tensors == "pt"
        # Orchestration fixture, not a test of official image preprocessing.
        values = torch.linspace(0, 1, 216).reshape(1, 18, 12)
        values = values * (0.7 + float(images.float().mean()) / 1000)
        return {"pixel_values": values, "image_position_ids": positions(width=3),
                "num_soft_tokens_per_image": torch.tensor([1])}


def test_preparation_exports_official_vision_features_and_records_stage_hashes(tmp_path, monkeypatch):
    module = preparation_module()
    snapshot = fake_snapshot(tmp_path / "snapshot")
    monkeypatch.setattr(module, "load_vision_models", lambda _: vision_models())
    monkeypatch.setattr(module, "load_image_processor", lambda _: TinyProcessor())
    result = module.prepare_vision(snapshot, tmp_path / "out", revision="a" * 40, source_commit="b" * 40,
                                   profiles={"square": (12, 12)})
    assert result["success"] is True
    assert result["stage"] == "vision_encoder_projector_only"
    assert result["unity_validated"] is False and result["end_to_end_validated"] is False
    assert result["profile_count"] == 1 and result["case_count"] == 2
    assert result["minimum_cosine"] >= 0.999999
    reference = json.loads((tmp_path / "out/vision-reference.json").read_text())
    assert reference["metadata"]["model_revision"] == "a" * 40
    assert reference["metadata"]["source_commit"] == "b" * 40
    assert reference["profiles"]["square"]["feature_count"] == 1
    assert reference["profiles"]["square"]["pixel_shape"] == [1, 18, 12]
    assert len(reference["cases"]) == 2
    assert reference["cases"][0]["features"] != reference["cases"][1]["features"]
    for name in result["files"]:
        assert (tmp_path / "out" / name).is_file()
        assert len(result["files"][name]["sha256"]) == 64
    assert json.loads((tmp_path / "out/vision-validation.json").read_text()) == result


def test_failure_removes_stale_vision_success(tmp_path, monkeypatch):
    module = preparation_module()
    snapshot = fake_snapshot(tmp_path / "snapshot")
    output = tmp_path / "out"
    output.mkdir()
    (output / "vision-validation.json").write_text('{"success":true}')
    def fail(_):
        raise RuntimeError("missing vision weights")
    monkeypatch.setattr(module, "load_vision_models", fail)
    with pytest.raises(RuntimeError, match="missing vision weights"):
        module.prepare_vision(snapshot, output)
    assert not (output / "vision-validation.json").exists()


def test_default_generated_images_pass_only_the_exact_reviewed_publication_policy(tmp_path, monkeypatch):
    from pathlib import Path
    from embeddinggemma_tools.privacy_artifacts import audit
    module=preparation_module()
    snapshot=fake_snapshot(tmp_path/'snapshot')
    monkeypatch.setattr(module,'load_vision_models',lambda _:vision_models())
    monkeypatch.setattr(module,'load_image_processor',lambda _:TinyProcessor())
    policy=Path(__file__).parents[2]/'.privacy-reviewed-assets.json'
    (tmp_path/'.privacy-reviewed-assets.json').write_bytes(policy.read_bytes())
    result=module.prepare_vision(snapshot,tmp_path/'artifacts/vision-encoder',source_commit='b'*40)
    assert result['success'] and result['case_count']==6
    assert not audit(tmp_path,['artifacts/vision-encoder/'])


def test_wrong_saved_vision_output_is_rejected_without_success_receipt(tmp_path, monkeypatch):
    module = preparation_module()
    snapshot = fake_snapshot(tmp_path / "snapshot")
    monkeypatch.setattr(module, "load_vision_models", lambda _: vision_models())
    monkeypatch.setattr(module, "load_image_processor", lambda _: TinyProcessor())
    class Wrong(torch.nn.Module):
        def forward(self, pixels):
            return torch.zeros(1, 16)
    def corrupt(wrapper, pixels, output):
        torch.export.save(torch.export.export(Wrong(), (pixels,)), output)
    monkeypatch.setattr(module, "export_vision_pt2", corrupt)
    with pytest.raises(ValueError, match="vision export mismatch"):
        module.prepare_vision(snapshot, tmp_path / "out", profiles={"square": (12, 12)})
    assert not (tmp_path / "out/vision-validation.json").exists()


def test_cli_downloads_pinned_processor_and_runs_vision_gate(tmp_path, monkeypatch, capsys):
    cli = importlib.import_module("embeddinggemma_tools.__main__")
    calls, downloads = [], []
    def download(model, **kwargs):
        downloads.append((model, kwargs))
        return str(tmp_path / "snapshot")
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return dict(success=True, stage="vision_encoder_projector_only", profile_count=3, case_count=6,
                    minimum_cosine=1.0, unity_validated=False, end_to_end_validated=False)
    monkeypatch.setattr(cli, "snapshot_download", download)
    monkeypatch.setattr(cli, "prepare_vision", prepare, raising=False)
    monkeypatch.setenv("GITHUB_SHA", "c" * 40)
    cli.main(["prepare-vision", "--output", str(tmp_path / "out")])
    assert downloads[0][1]["revision"] == "914f7f89142e33e77833254d9c9b90c3cef7303b"
    assert "processor_config.json" in downloads[0][1]["allow_patterns"]
    assert calls[0][1]["source_commit"] == "c" * 40
    assert json.loads(capsys.readouterr().out)["end_to_end_validated"] is False


def test_official_processor_square_rgb_patch_order_padding_and_actual_soft_token_count(tmp_path):
    module = preparation_module()
    snapshot = fake_snapshot(tmp_path / "snapshot")
    processor = module.load_image_processor(snapshot)
    red = torch.zeros(3, 12, 12, dtype=torch.uint8)
    red[0] = 255
    actual = processor(images=red, return_tensors="pt")
    assert actual["pixel_values"].shape == (1, 2520, 12)
    assert actual["num_soft_tokens_per_image"].tolist() == [256]  # 280 is a cap, not the actual count.
    torch.testing.assert_close(actual["pixel_values"][0, :2304], torch.tensor([1., 0., 0.] * 4).expand(2304, 12))
    assert torch.count_nonzero(actual["pixel_values"][0, 2304:]) == 0
    assert actual["image_position_ids"][0, 0].tolist() == [0, 0]
    assert actual["image_position_ids"][0, 47].tolist() == [47, 0]
    assert actual["image_position_ids"][0, 48].tolist() == [0, 1]
    assert actual["image_position_ids"][0, 2303].tolist() == [47, 47]
    assert torch.all(actual["image_position_ids"][0, 2304:] == -1)


@pytest.mark.parametrize("profiles", [{}, {"../escape": (12, 12)}, {"square": (0, 12)}, {"square": (12, -1)}])
def test_invalid_profile_definitions_fail_before_loading_models_or_writing_graphs(tmp_path, monkeypatch, profiles):
    module = preparation_module()
    snapshot = fake_snapshot(tmp_path / "snapshot")
    calls = []
    monkeypatch.setattr(module, "load_vision_models", lambda _: calls.append("loaded") or vision_models())
    monkeypatch.setattr(module, "load_image_processor", lambda _: TinyProcessor())
    with pytest.raises(ValueError, match="profile"):
        module.prepare_vision(snapshot, tmp_path / "out", profiles=profiles)
    assert calls == []
    assert not (tmp_path / "out/vision-validation.json").exists()
