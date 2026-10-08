import json
from types import SimpleNamespace

import pytest
import torch

from embeddinggemma_tools.reference import build_reference, load_cases, write_reference


class StubTokenizer:
    def __call__(self, text, **kwargs):
        assert kwargs == {"padding": "max_length", "truncation": True, "max_length": 4, "return_tensors": "pt"}
        assert text == "task: search result | query: 猫"
        return {"input_ids": torch.tensor([[2, 3, 1, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0]])}


class StubModel(torch.nn.Module):
    config = SimpleNamespace(embedding_dim=3)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=torch.tensor([[[3., 0., 0.], [0., 4., 0.],
                                                               [0., 0., 0.], [999., 999., 999.]]]))


def test_reference_matches_masked_mean_and_official_normalization():
    rows = build_reference(StubModel(), StubTokenizer(), [{"id": "ja", "text": "猫", "role": "query"}], 4)
    assert len(rows) == 1
    assert rows[0]["id"] == "ja"
    assert rows[0]["formatted_text"] == "task: search result | query: 猫"
    assert rows[0]["input_ids"] == [2, 3, 1, 0]
    assert rows[0]["attention_mask"] == [1, 1, 1, 0]
    assert rows[0]["embedding"] == pytest.approx([0.6, 0.8, 0.0])


def test_case_loader_expands_long_inputs_and_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"id": "long", "text": "猫", "role": "raw", "repeat": 150}]))
    assert load_cases(path)[0]["text"] == "猫" * 150
    path.write_text(json.dumps([{"id": "same", "text": "a", "role": "raw"}] * 2))
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path)


@pytest.mark.parametrize("repeat", [0, -1, True, 1.5])
def test_invalid_repeat_is_rejected(tmp_path, repeat):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"id": "bad", "text": "a", "role": "raw", "repeat": repeat}]))
    with pytest.raises(ValueError, match="repeat"):
        load_cases(path)


def test_case_loader_rejects_empty_suite(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="empty"):
        load_cases(path)


def test_reference_writer_preserves_unicode_and_metadata(tmp_path):
    rows = build_reference(StubModel(), StubTokenizer(), [{"id": "ja", "text": "猫", "role": "query"}], 4)
    output = tmp_path / "output" / "reference.json"
    write_reference(output, rows, {"model_revision": "abc", "sequence_length": 4})
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["metadata"]["model_revision"] == "abc"
    assert data["cases"] == rows
    assert "猫" in output.read_text(encoding="utf-8")


def test_nonfinite_reference_is_rejected():
    class BrokenModel(StubModel):
        def forward(self, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.full((1, 4, 3), float("nan")))

    with pytest.raises(ValueError, match="finite"):
        build_reference(BrokenModel(), StubTokenizer(), [{"id": "ja", "text": "猫", "role": "query"}], 4)
