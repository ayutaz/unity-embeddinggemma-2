import hashlib
import inspect
import io
import threading
import time
import zipfile

import pytest

from embeddinggemma_tools import artifact_zip
from test_artifact_zip import archive_bytes


def test_member_payload_is_read_in_bounded_blocks(tmp_path, monkeypatch):
    data = archive_bytes()
    original = zipfile.ZipExtFile.read
    def bounded(self, size=-1):
        assert 0 < size <= 32 * 1024 * 1024, "Artifact members must be streamed, never read in one unbounded allocation"
        return original(self, size)
    monkeypatch.setattr(zipfile.ZipExtFile, "read", bounded)
    reader = artifact_zip.RangeReader(len(data), lambda start, stop: data[start:stop])
    result = artifact_zip.extract_members(reader, tmp_path, ["large/model.onnx"])
    assert result["files"]["large/model.onnx"]["sha256"] == hashlib.sha256(bytes(range(256))*4096).hexdigest()


def test_ranges_use_at_most_four_parallel_requests_and_preserve_byte_order():
    assert "max_parallel" in inspect.signature(artifact_zip.RangeReader).parameters, "Bounded parallel ranges are missing"
    lock, active, maximum = threading.Lock(), 0, 0
    def fetch(start, stop):
        nonlocal active, maximum
        with lock: active += 1; maximum = max(maximum, active)
        time.sleep(0.01)
        with lock: active -= 1
        return bytes(range(start, stop))
    reader = artifact_zip.RangeReader(100, fetch, chunk_bytes=10, max_parallel=4)
    assert reader.read(100) == bytes(range(100))
    assert 1 < maximum <= 4


def test_verified_cached_member_is_not_downloaded_again(tmp_path):
    data = archive_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        spec = archive.getinfo("metadata/ref.json")
    start = spec.header_offset + 30 + len(spec.filename.encode()) + 20  # forced ZIP64 local header extra
    stop = start + spec.compress_size
    def fetch(left, right):
        assert right <= start or left >= stop, "Verified cached member must not be downloaded again"
        return data[left:right]
    path = tmp_path / "metadata/ref.json"; path.parent.mkdir(); path.write_bytes(b'{"success":false}')
    reader = artifact_zip.RangeReader(len(data), fetch)
    result = artifact_zip.extract_members(reader, tmp_path, ["metadata/ref.json"])
    assert result["files"]["metadata/ref.json"]["cache_reused"] is True


def test_crc_failure_preserves_old_target_and_removes_only_new_partial_file(tmp_path):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("payload.bin", b"expected-payload")
    raw = bytearray(stream.getvalue()); raw[30+len("payload.bin")] ^= 1
    path = tmp_path / "payload.bin"; path.write_bytes(b"old")
    reader = artifact_zip.RangeReader(len(raw), lambda start, stop: bytes(raw[start:stop]))
    with pytest.raises(zipfile.BadZipFile):
        artifact_zip.extract_members(reader, tmp_path, ["payload.bin"])
    assert path.read_bytes() == b"old" and not list(tmp_path.glob('*.partial'))
