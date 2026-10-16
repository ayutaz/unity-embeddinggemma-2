import copy
import hashlib
import json

import pytest

from embeddinggemma_tools import audio_pinned_context
from test_audio_pinned_context import reference


def main():
    function = getattr(audio_pinned_context, "main", None)
    assert callable(function), "Pinned audio context gate CLI is missing"
    return function


def write(directory, ref, audit):
    directory.mkdir()
    raw = json.dumps(ref).encode()
    (directory / "audio-encoder-reference.json").write_bytes(raw)
    audit = copy.deepcopy(audit)
    audit["files"]["audio-encoder-reference.json"] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    (directory / "audio-encoder-validation.json").write_text(json.dumps(audit))


def test_cli_retains_full_failure_report_and_exits_nonzero(tmp_path):
    run = main()
    ref, audit = reference()
    write(tmp_path / "pinned", ref, audit)
    changed = copy.deepcopy(ref)
    changed["cases"][0]["features"][0][0] += 0.1
    write(tmp_path / "current", changed, audit)
    with pytest.raises(SystemExit) as failure:
        run(["--current", str(tmp_path / "current"), "--pinned", str(tmp_path / "pinned"),
             "--output", str(tmp_path / "context.json")])
    assert failure.value.code == 1
    report = json.loads((tmp_path / "context.json").read_text())
    assert report["context_validated"] is False and len(report["cases"]) == 6
    assert report["reference_hashes_audited"] is True


def test_cli_rejects_reference_corruption_before_accepting_context(tmp_path):
    run = main()
    ref, audit = reference()
    write(tmp_path / "pinned", ref, audit)
    write(tmp_path / "current", ref, audit)
    (tmp_path / "current/audio-encoder-reference.json").write_bytes(b'{}')
    (tmp_path / "context.json").write_text('{"context_validated":true}')
    with pytest.raises(ValueError, match="reference hash"):
        run(["--current", str(tmp_path / "current"), "--pinned", str(tmp_path / "pinned"),
             "--output", str(tmp_path / "context.json")])
    assert not (tmp_path / "context.json").exists()
