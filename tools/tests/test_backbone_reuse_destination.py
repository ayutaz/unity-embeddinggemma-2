import pytest

from embeddinggemma_tools import audio_embeddings, image_embeddings


@pytest.mark.parametrize("modality", ["audio", "image"])
@pytest.mark.parametrize("relation", ["same", "destination_inside", "source_inside"])
def test_reuse_rejects_overlapping_destination_before_changing_receipt(tmp_path, modality, relation):
    source = tmp_path / "reference"
    destination = source
    if relation == "destination_inside":
        destination = source / "new-role"
    elif relation == "source_inside":
        source = destination / "old-role"
    source.mkdir(parents=True)
    receipt = source / (modality + "-embedding-validation.json")
    expected = b'{"success": true, "immutable": true}'
    receipt.write_bytes(expected)
    module = audio_embeddings if modality == "audio" else image_embeddings
    run = getattr(module, "prepare_" + modality + "_embeddings")
    with pytest.raises(ValueError, match="backbone.*separate"):
        run(tmp_path / "snapshot", tmp_path / "encoder", destination, role="raw", backbone_source=source,
            revision="a" * 40, source_commit="b" * 40)
    assert receipt.read_bytes() == expected
