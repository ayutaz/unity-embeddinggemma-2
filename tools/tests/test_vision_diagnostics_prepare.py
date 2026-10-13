import json

import pytest

from embeddinggemma_tools import prepare, vision_diagnostics
from test_vision import vision_models
from test_vision_prepare import TinyProcessor, fake_snapshot


@pytest.mark.parametrize("success", [False, True])
def test_prepare_diagnoses_only_first_case_and_preserves_diagnostic_failure(tmp_path, monkeypatch, success):
    snapshot = fake_snapshot(tmp_path / "snapshot")
    monkeypatch.setattr(prepare, "load_vision_models", lambda _: vision_models())
    monkeypatch.setattr(prepare, "load_image_processor", lambda _: TinyProcessor())
    calls = []
    def diagnostic(vision, projection, pixels, coords, destination, original, **kwargs):
        calls.append((destination, original, kwargs))
        destination.mkdir(parents=True, exist_ok=True)
        result = dict(success=success, stage="vision_encoder_diagnostics_only")
        (destination / "vision-diagnostics.json").write_text(json.dumps(result))
        return result
    monkeypatch.setattr(vision_diagnostics, "prepare_vision_diagnostics", diagnostic)
    destination = tmp_path / "out"
    arguments = dict(revision="a" * 40, source_commit="b" * 40,
                     profiles={"square": (12, 12), "landscape": (12, 20)}, diagnostics=True)
    if success:
        result = prepare.prepare_vision(snapshot, destination, **arguments)
        assert result["success"] is True
        assert "diagnostics/vision-diagnostics.json" in result["files"]
    else:
        with pytest.raises(ValueError, match="diagnostic"):
            prepare.prepare_vision(snapshot, destination, **arguments)
        assert not (destination / "vision-validation.json").exists()
        assert json.loads((destination / "diagnostics/vision-diagnostics.json").read_text())["success"] is False
    assert len(calls) == 1
    assert calls[0][1] == destination / "square/vision.pt2"
    assert "additional_pixels" in calls[0][2], "Both real inputs for the first failing profile must be diagnosed."
    assert len(calls[0][2]["additional_pixels"]) == 1
    assert calls[0][2]["source_commit"] == "b" * 40 and calls[0][2]["revision"] == "a" * 40


def test_diagnostics_are_not_generated_when_not_requested(tmp_path, monkeypatch):
    snapshot = fake_snapshot(tmp_path / "snapshot")
    monkeypatch.setattr(prepare, "load_vision_models", lambda _: vision_models())
    monkeypatch.setattr(prepare, "load_image_processor", lambda _: TinyProcessor())
    def unexpected(*args, **kwargs):
        pytest.fail("Unchanged vision preparation must not produce an extra heavy diagnostic graph.")
    monkeypatch.setattr(vision_diagnostics, "prepare_vision_diagnostics", unexpected)
    prepare.prepare_vision(snapshot, tmp_path / "out", profiles={"square": (12, 12)})


def test_cli_diagnostics_flag_uses_existing_snapshot_and_fixed_source(tmp_path, monkeypatch, capsys):
    from embeddinggemma_tools import __main__ as cli
    calls = []
    def run(snapshot, output, **kwargs):
        calls.append((snapshot, output, kwargs))
        return dict(success=True, stage="vision_encoder_projector_only", profile_count=3, case_count=6,
                    minimum_cosine=1, unity_validated=False, end_to_end_validated=False)
    monkeypatch.setattr(cli, "prepare_vision", run)
    monkeypatch.setenv("GITHUB_SHA", "c" * 40)
    cli.main(["prepare-vision", "--snapshot", str(tmp_path / "snapshot"), "--output", str(tmp_path / "out"), "--diagnostics"])
    assert calls[0][2]["diagnostics"] is True
    assert calls[0][2]["source_commit"] == "c" * 40
    assert json.loads(capsys.readouterr().out)["unity_validated"] is False
