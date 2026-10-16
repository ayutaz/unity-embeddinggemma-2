import io

import pytest

from embeddinggemma_tools import artifact_zip


@pytest.mark.parametrize("limit", [None, 2 * 1024 ** 3])
def test_cli_requires_explicit_large_model_selection_limit(tmp_path, monkeypatch, limit):
    seen = {}
    reader = artifact_zip.RangeReader(4, lambda start, stop: b"data"[start:stop])
    monkeypatch.setattr(artifact_zip, "github_artifact_reader", lambda *_: (reader, dict(id=7, name="fixed")))
    def extract(source, destination, names, **kwargs):
        seen.update(kwargs)
        destination.mkdir()
        return dict(selected_count=1, range_bytes=4, range_requests=1, files={})
    monkeypatch.setattr(artifact_zip, "extract_members", extract)
    arguments = ["--repo", "owner/repo", "--artifact-id", "7", "--output", str(tmp_path / "out"), "--member", "model.pt2"]
    if limit is not None:
        arguments.extend(["--max-entry-bytes", str(limit)])
    artifact_zip.main(arguments)
    assert seen.get("max_entry_bytes", 64 * 1024 ** 2) == (limit or 64 * 1024 ** 2)
    reader.close()
