import pytest
import torch

from embeddinggemma_tools import reference_context_failure as context


def probe(*args, **kwargs):
    function = getattr(context, "probe_cpu_threads", None)
    assert callable(function), "Controlled CPU thread context probe is missing"
    return function(*args, **kwargs)


def test_probe_compares_each_actual_thread_context_without_replacing_expected():
    initial = torch.get_num_threads()
    expected = torch.tensor([2.0])
    report = probe(lambda: torch.tensor([float(torch.get_num_threads())]), (), expected, thread_counts=(1, 2, 4))
    assert report["original_threads"] == initial and torch.get_num_threads() == initial
    assert [row["actual_threads"] for row in report["conditions"]] == [1, 2, 4]
    assert [row["mismatched_elements"] for row in report["conditions"]] == [1, 0, 1]
    assert expected.tolist() == [2.0]
    assert report["full_encoder_validated"] is False and report["end_to_end_validated"] is False


def test_probe_restores_threads_and_retains_failed_execution_separately():
    initial = torch.get_num_threads()
    def evaluate():
        if torch.get_num_threads() == 2:
            raise RuntimeError("fixture execution failure")
        return torch.tensor([1.0])
    report = probe(evaluate, (), torch.tensor([1.0]), thread_counts=(1, 2, 4))
    assert torch.get_num_threads() == initial
    assert report["conditions"][1]["error"] == "RuntimeError: fixture execution failure"
    assert report["conditions"][1]["executed"] is False
    assert report["conditions"][2]["mismatched_elements"] == 0


@pytest.mark.parametrize("counts", [(0,), (True,), (), (2, 2)])
def test_invalid_probe_thread_counts_are_rejected_before_execution(counts):
    initial = torch.get_num_threads()
    with pytest.raises(ValueError, match="thread"):
        probe(lambda: pytest.fail("Invalid thread probe executed"), (), torch.ones(1), thread_counts=counts)
    assert torch.get_num_threads() == initial
