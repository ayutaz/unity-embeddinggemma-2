import io
import zipfile

import pytest

from embeddinggemma_tools import artifact_subset


def required(name):
    value = getattr(artifact_subset, name, None)
    assert value is not None, f"selective artifact download is missing: {name}"
    return value


def test_seek_cross_block_eof_and_cache_have_file_semantics():
    reader_type = required("RangeReader")
    data = bytes(range(40)); fetched = []
    def fetch(start, stop):
        fetched.append((start, stop)); return data[start:stop]
    reader = reader_type(fetch, len(data), block_size=8)
    reader.seek(5)
    assert reader.read(12) == data[5:17]
    reader.seek(-3, io.SEEK_CUR)
    assert reader.read(2) == data[14:16]
    assert fetched.count((8, 16)) == 1
    reader.seek(-2, io.SEEK_END)
    assert reader.read(9) == data[-2:]
    assert reader.read(1) == b""
    assert reader.tell() == len(data)


def test_incomplete_range_response_cannot_be_accepted_as_file_data():
    reader_type = required("RangeReader")
    reader = reader_type(lambda start, stop: b"", 10, block_size=8)
    with pytest.raises(OSError, match="range"):
        reader.read(1)


def test_unrelated_model_payload_is_not_downloaded(tmp_path):
    reader_type, extract = required("RangeReader"), required("extract_subset")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("landscape/vision.pt2", b"x" * 1024 * 1024)
        archive.writestr("diagnostics/native-stage-0.bin", b"native pixels")
        archive.writestr("diagnostics/vision-diagnostics.json", b"{}")
    data, requests = stream.getvalue(), []
    def fetch(start, stop):
        requests.append((start, stop)); return data[start:stop]
    report = extract(reader_type(fetch, len(data), block_size=512), tmp_path / "out", "diagnostics/")
    assert sorted(report) == ["native-stage-0.bin", "vision-diagnostics.json"]
    assert (tmp_path / "out/native-stage-0.bin").read_bytes() == b"native pixels"
    assert not (tmp_path / "out/landscape").exists()
    assert sum(stop - start for start, stop in requests) < len(data) / 10


def test_archive_path_escape_is_rejected(tmp_path):
    extract = required("extract_subset")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("diagnostics/../../escape.bin", b"bad")
    stream.seek(0)
    with pytest.raises(ValueError, match="path"):
        extract(stream, tmp_path / "out", "diagnostics/")
    assert not (tmp_path / "escape.bin").exists()


def test_completed_entry_is_reused_only_when_size_and_crc_match(tmp_path):
    reader_type, extract = required("RangeReader"), required("extract_subset")
    value = b"native reference" * 100000
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("diagnostics/native.bin", value)
    destination = tmp_path / "out"; destination.mkdir()
    (destination / "native.bin").write_bytes(value)
    data, requested = stream.getvalue(), []
    def fetch(start, stop):
        requested.append((start, stop)); return data[start:stop]
    extract(reader_type(fetch, len(data), block_size=512), destination, "diagnostics/")
    assert sum(stop - start for start, stop in requested) < len(data) / 10, "Completed native reference was downloaded again."
