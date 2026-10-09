import copy
import json

import pytest

from embeddinggemma_tools.search import load_search_cases, rank_documents, generate_search_reference, validate_search_reference


def suite():
    return {"documents": [{"id": "doc-b", "title": "猫", "text": "猫と暮らす"},
                          {"id": "doc-a", "title": "Cats", "text": "Living with cats"}],
            "queries": [{"id": "query-ja", "text": "猫について"}]}


def write_suite(tmp_path, data=None):
    path = tmp_path / "search.json"
    path.write_text(json.dumps(data if data is not None else suite()), encoding="utf-8")
    return path


def test_load_search_cases_preserves_titles_and_roles(tmp_path):
    data = load_search_cases(write_suite(tmp_path))
    assert data == suite()


@pytest.mark.parametrize("fault", ["empty_docs", "empty_queries", "duplicate", "empty_text", "bad_id", "bad_title", "overlap"])
def test_invalid_search_suite_is_rejected_before_inference(tmp_path, fault):
    data = suite()
    if fault == "empty_docs": data["documents"] = []
    elif fault == "empty_queries": data["queries"] = []
    elif fault == "duplicate": data["documents"][1]["id"] = "doc-b"
    elif fault == "empty_text": data["queries"][0]["text"] = " \n "
    elif fault == "bad_id": data["documents"][0]["id"] = "猫"
    elif fault == "bad_title": data["documents"][0]["title"] = 3
    elif fault == "overlap": data["queries"][0]["id"] = "doc-b"
    with pytest.raises(ValueError):
        load_search_cases(write_suite(tmp_path, data))


def test_cosine_normalizes_and_ties_use_ordinal_id_not_input_order():
    docs = [{"id": "b", "embedding": [3.0, 0.0]}, {"id": "a", "embedding": [6.0, 0.0]},
            {"id": "c", "embedding": [-2.0, 0.0]}]
    ranked = rank_documents([9.0, 0.0], docs)
    assert [row["document_id"] for row in ranked] == ["a", "b", "c"]
    assert [row["score"] for row in ranked] == pytest.approx([1, 1, -1])
    assert docs[0]["embedding"] == [3.0, 0.0]


@pytest.mark.parametrize("vector", [[], [0, 0], [float("nan"), 1], [float("inf"), 1], [1]])
def test_invalid_vectors_are_rejected(vector):
    with pytest.raises(ValueError):
        rank_documents(vector, [{"id": "a", "embedding": [1.0, 0.0]}])


def test_generation_uses_document_and_query_prompts_and_validation_detects_tampering(tmp_path, monkeypatch):
    def fake_reference(model, tokenizer, cases, sequence_length):
        assert sequence_length == 128
        assert [case["role"] for case in cases] == ["document", "document", "query"]
        assert cases[0]["title"] == "猫"
        return [{**case, "embedding": [1.0, 0.0], "input_ids": [1] * 128,
                 "attention_mask": [1] * 128, "formatted_text": case["text"]} for case in cases]
    monkeypatch.setattr("embeddinggemma_tools.search.build_reference", fake_reference)
    metadata = {"model_revision": "a" * 40, "source_commit": "b" * 40, "embedding_dimension": 2}
    path = tmp_path / "search-reference.json"
    reference = generate_search_reference(None, None, write_suite(tmp_path), path, 128, metadata)
    assert json.loads(path.read_text(encoding="utf-8")) == reference
    assert reference["metadata"] == metadata
    assert [row["document_id"] for row in reference["queries"][0]["ranking"]] == ["doc-a", "doc-b"]
    validate_search_reference(reference, suite(), metadata)
    for fault in ("rank", "score", "source", "input"):
        corrupt = copy.deepcopy(reference)
        if fault == "rank": corrupt["queries"][0]["ranking"].reverse()
        elif fault == "score": corrupt["queries"][0]["ranking"][0]["score"] = 0.5
        elif fault == "source": corrupt["metadata"]["source_commit"] = "c" * 40
        elif fault == "input": corrupt["documents"][0]["text"] = "different"
        with pytest.raises(ValueError):
            validate_search_reference(corrupt, suite(), metadata)
