import hashlib
import importlib
import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from embeddinggemma_tools.prepare import prepare
from embeddinggemma_tools.__main__ import main
from test_model import save_snapshot, tiny_model


def make_snapshot(path):
    save_snapshot(path, tiny_model())
    backend = Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "cat": 4}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="<bos> $A <eos>", special_tokens=[("<bos>", 2), ("<eos>", 1)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                       unk_token="<unk>", bos_token="<bos>", eos_token="<eos>")
    tokenizer.save_pretrained(path)


def test_offline_prepare_produces_reloadable_model_and_reproducible_reference(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    make_snapshot(snapshot)
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"id": "cat", "text": "cat", "role": "raw"},
                                {"id": "long", "text": "cat ", "role": "raw", "repeat": 10}]))
    output = tmp_path / "output"
    summary = prepare(snapshot, cases, output, sequence_length=6, revision="0" * 40)
    reference = json.loads((output / "reference.json").read_text())
    assert reference["cases"][0]["input_ids"] == [2, 4, 1, 0, 0, 0]
    assert reference["metadata"]["sequence_length"] == 6
    assert reference["metadata"]["model_revision"] == "0" * 40
    assert reference["metadata"]["embedding_dimension"] == 24
    assert reference["metadata"]["versions"]["torch"] == torch.__version__
    assert summary["case_count"] == 2
    assert summary["minimum_cosine"] >= 0.999999
    for name in ("model.pt2", "reference.json", "tokenizer.json"):
        assert summary["sha256"][name] == hashlib.sha256((output / name).read_bytes()).hexdigest()
    assert json.loads((output / "export-validation.json").read_text()) == summary
    reloaded_tokenizer = Tokenizer.from_file(str(output / "tokenizer.json"))
    for case in reference["cases"]:
        encoded = reloaded_tokenizer.encode(case["formatted_text"])
        assert encoded.ids == case["input_ids"]
        assert encoded.attention_mask == case["attention_mask"]
    program = torch.export.load(output / "model.pt2").module()
    for case in reference["cases"]:
        with torch.no_grad():
            vector = program(torch.tensor([case["input_ids"]]), torch.tensor([case["attention_mask"]]))[0]
        torch.testing.assert_close(vector, torch.tensor(case["embedding"]), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("length", [0, 1, 8193])
def test_cli_rejects_invalid_sequence_length_before_downloading(length):
    with pytest.raises(SystemExit) as failure:
        main(["prepare", "--output", "unused", "--sequence-length", str(length)])
    assert failure.value.code == 2


def test_cli_rejects_mutable_revision_before_downloading():
    with pytest.raises(SystemExit) as failure:
        main(["prepare", "--output", "unused", "--revision", "main"])
    assert failure.value.code == 2


@pytest.mark.parametrize("failure_stage", ["load_text_model", "export_pt2"])
def test_failed_rerun_invalidates_previous_success_report(tmp_path, monkeypatch, failure_stage):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    make_snapshot(snapshot)
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"id": "cat", "text": "cat", "role": "raw"}]))
    output = tmp_path / "output"
    output.mkdir()
    report = output / "export-validation.json"
    report.write_text(json.dumps({"minimum_cosine": 1.0, "metadata": {"source_commit": "previous"}}))

    def fail(*args, **kwargs):
        raise RuntimeError("current generation failed")

    module = importlib.import_module("embeddinggemma_tools.prepare")
    monkeypatch.setattr(module, failure_stage, fail)
    with pytest.raises(RuntimeError, match="current generation failed"):
        prepare(snapshot, cases, output, sequence_length=6, revision="0" * 40)
    assert not report.exists(), "A failed generation must not leave a previous success report."
