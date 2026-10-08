import pytest
import torch

from embeddinggemma_tools.text import format_text, pool_and_normalize


@pytest.mark.parametrize(
    "text,role,title,expected",
    [
        ("猫", "query", None, "task: search result | query: 猫"),
        ("猫", "document", None, "title: none | text: 猫"),
        ("cat", "document", "動物", "title: 動物 | text: cat"),
        ("  \n😀", "raw", None, "  \n😀"),
        ("", "query", None, "task: search result | query: "),
    ],
)
def test_format_text_preserves_content(text, role, title, expected):
    assert format_text(text, role, title) == expected


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError, match="role"):
        format_text("cat", "guess")


def test_title_is_only_valid_for_documents():
    with pytest.raises(ValueError, match="title"):
        format_text("cat", "query", "title")


def test_pooling_uses_mask_and_normalizes_each_row():
    hidden = torch.tensor([[[3.0, 0.0], [0.0, 4.0], [999.0, 999.0]],
                           [[0.0, 5.0], [999.0, 999.0], [999.0, 999.0]]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    actual = pool_and_normalize(hidden, mask)
    torch.testing.assert_close(actual, torch.tensor([[0.6, 0.8], [0.0, 1.0]]))


def test_truncation_is_applied_before_final_normalization():
    actual = pool_and_normalize(torch.tensor([[[3.0, 4.0, 12.0]]]), torch.ones(1, 1), 2)
    torch.testing.assert_close(actual, torch.tensor([[0.6, 0.8]]))


def test_all_padding_is_finite_zero():
    actual = pool_and_normalize(torch.ones(1, 2, 3), torch.zeros(1, 2))
    torch.testing.assert_close(actual, torch.zeros(1, 3))


@pytest.mark.parametrize("dimensions", [0, -1, 4])
def test_invalid_dimensions_are_rejected(dimensions):
    with pytest.raises(ValueError, match="dimensions"):
        pool_and_normalize(torch.ones(1, 2, 3), torch.ones(1, 2), dimensions)
