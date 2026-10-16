import io
import importlib
import zipfile

import pytest

def module():
    try:
        return importlib.import_module("embeddinggemma_tools.artifact_zip")
    except ModuleNotFoundError:
        pytest.fail("Bounded artifact ZIP selection API is missing")


def archive_bytes():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        with archive.open("metadata/ref.json", "w", force_zip64=True) as member:
            member.write(b'{"success":false}')
        archive.writestr("large/model.onnx", bytes(range(256)) * 4096)
    return stream.getvalue()


def test_selected_zip64_member_uses_bounded_ranges_and_preserves_failed_status(tmp_path):
    artifact_zip = module()
    data, calls = archive_bytes(), []
    def fetch(start, stop):
        calls.append((start, stop))
        return data[start:stop]
    reader = artifact_zip.RangeReader(len(data), fetch, chunk_bytes=31)
    report = artifact_zip.extract_members(reader, tmp_path, ["metadata/ref.json"])
    assert (tmp_path / "metadata/ref.json").read_bytes() == b'{"success":false}'
    assert not (tmp_path / "large/model.onnx").exists()
    assert max(stop - start for start, stop in calls) <= 31
    assert report["selected_count"] == 1 and report["range_bytes"] < len(data)
    assert len(report["files"]["metadata/ref.json"]["sha256"]) == 64


def test_short_range_does_not_silently_corrupt_a_member():
    artifact_zip = module()
    reader = artifact_zip.RangeReader(100, lambda start, stop: b"x" * (stop-start-1), chunk_bytes=10)
    with pytest.raises(ValueError, match="range length"):
        reader.read(10)


@pytest.mark.parametrize("name", ["../escape.json", "/absolute.json", "metadata\\ref.json"])
def test_member_path_is_refused_before_any_network_or_write(tmp_path, name):
    artifact_zip = module()
    reader = artifact_zip.RangeReader(100, lambda *args: pytest.fail("audit names before network"))
    with pytest.raises(ValueError, match="member path"):
        artifact_zip.extract_members(reader, tmp_path, [name])


def test_member_size_limit_is_checked_before_reading_its_payload(tmp_path):
    artifact_zip = module()
    data = archive_bytes()
    reader = artifact_zip.RangeReader(len(data), lambda start, stop: data[start:stop])
    with pytest.raises(ValueError, match="member size"):
        artifact_zip.extract_members(reader, tmp_path, ["large/model.onnx"], max_entry_bytes=100)
    assert not (tmp_path / "large/model.onnx").exists()
