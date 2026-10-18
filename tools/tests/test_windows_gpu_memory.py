"""Missing counters are unavailable, not zero memory or GPU acceptance."""
import pytest
from embeddinggemma_tools.windows_gpu_memory import summarize_gpu_memory


def row(pid, adapter, value, status=0):
    return {'name': f'pid_{pid}_luid_0x0_0x{adapter}_phys_0', 'bytes': value, 'status': status}


def test_exact_pid_and_sum_distinct_adapter_allocations():
    result = summarize_gpu_memory(12, [row(12, 'a', 1024), row(12, 'b', 2048), row(123, 'a', 9999)],
                                  [row(12, 'a', 128), row(12, 'b', 256)])
    assert result['dedicated_bytes'] == 3072
    assert result['shared_bytes'] == 384
    assert result['available'] is True
    assert result['gpu_execution_validated'] is False
    assert result['scope'] == 'sampled_process_allocations_not_capacity_or_peak_or_gpu_time'


def test_missing_instance_is_unavailable_not_zero():
    result = summarize_gpu_memory(12, [row(123, 'a', 9999)], [])
    assert result['dedicated_bytes'] is None
    assert result['shared_bytes'] is None
    assert result['available'] is False


def test_valid_zero_is_distinct_from_absent():
    result = summarize_gpu_memory(12, [row(12, 'a', 0, 1)], [row(12, 'a', 0)])
    assert result['available'] is True
    assert result['dedicated_bytes'] == 0


def test_invalid_status_does_not_publish_partial_total():
    result = summarize_gpu_memory(12, [row(12, 'a', 10), row(12, 'b', 20, 0xc0000bc6)], [row(12, 'a', 1)])
    assert result['available'] is False
    assert result['dedicated_bytes'] is None
    assert result['shared_bytes'] == 1


@pytest.mark.parametrize('value', [-1, 1.5, True])
def test_malformed_size_is_not_a_memory_observation(value):
    with pytest.raises(ValueError):
        summarize_gpu_memory(12, [row(12, 'a', value)], [row(12, 'a', 1)])


def test_duplicate_instance_rejected_instead_of_double_counted():
    with pytest.raises(ValueError):
        summarize_gpu_memory(12, [row(12, 'a', 1), row(12, 'a', 1)], [row(12, 'a', 1)])


@pytest.mark.parametrize('pid', [0, -1, True, '12'])
def test_invalid_pid_rejected(pid):
    with pytest.raises(ValueError):
        summarize_gpu_memory(pid, [], [])
