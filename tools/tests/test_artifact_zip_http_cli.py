import io
import json

import pytest

from embeddinggemma_tools import artifact_zip
from test_artifact_zip import archive_bytes


def fetcher():
    function = getattr(artifact_zip, "http_range_fetcher", None)
    assert callable(function), "HTTP range response validation is missing"
    return function


class Response(io.BytesIO):
    def __init__(self, raw, status, content_range):
        super().__init__(raw)
        self.status = status
        self.headers = {"Content-Range": content_range}


def test_http_range_requires_exact_206_content_range_and_bounded_body():
    requests = []
    def open_request(request, timeout):
        requests.append(request)
        assert timeout == 120
        return Response(b"abc", 206, "bytes 2-4/10")
    fetch = fetcher()("https://example.test/artifact", 10, opener=open_request)
    assert fetch(2, 5) == b"abc"
    assert requests[0].get_header("Range") == "bytes=2-4"


@pytest.mark.parametrize("status,header,raw", [(200, "bytes 2-4/10", b"abc"),
                                                (206, "bytes 0-2/10", b"abc"),
                                                (206, "bytes 2-4/10", b"abcd")])
def test_http_response_cannot_silently_fall_back_to_entire_archive(status, header, raw):
    fetch = fetcher()("https://example.test/artifact", 10,
                      opener=lambda *args, **kwargs: Response(raw, status, header))
    with pytest.raises(ValueError, match="HTTP range"):
        fetch(2, 5)


def test_cli_selects_members_and_records_actual_artifact_identity(tmp_path, monkeypatch):
    function = getattr(artifact_zip, "main", None)
    assert callable(function), "Artifact ZIP selection CLI is missing"
    data, calls = archive_bytes(), []
    def reader(repo, artifact_id):
        calls.append((repo, artifact_id))
        return artifact_zip.RangeReader(len(data), lambda start, stop: data[start:stop]), {
            "id": artifact_id, "name": "audio-runtime-native-" + "b" * 40}
    monkeypatch.setattr(artifact_zip, "github_artifact_reader", reader, raising=False)
    function(["--repo", "owner/repo", "--artifact-id", "123", "--output", str(tmp_path),
              "--member", "metadata/ref.json"])
    assert calls == [("owner/repo", 123)]
    report = json.loads((tmp_path / "range-receipt.json").read_text())
    assert report["artifact_id"] == 123
    assert report["artifact_name"] == "audio-runtime-native-" + "b" * 40
    assert report["files"]["metadata/ref.json"]["bytes"] == 17
