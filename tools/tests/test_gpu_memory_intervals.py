"""Counter timestamps, exact PID, incomplete data and observed maxima stay distinct."""
import pytest
from embeddinggemma_tools import windows_gpu_memory as memory

def counter(pid,value,kind='dedicated usage',status=0):
    return dict(name=f'pid_{pid}_luid_0x0_0xaa_phys_0',counter=kind,bytes=value,status=status)

def observation(second,value=100,pid=12,source='counter_sample_set',error=None):
    return dict(pid=pid,utc=f'2026-10-18T00:00:{second:02d}+00:00',timestamp_source=source,error=error,
        rows=[counter(pid,value),counter(pid,value//2,'shared usage')])

def interval(name,start,end):
    return dict(id=name,begin_utc=f'2026-10-18T09:00:{start:02d}+09:00',end_utc=f'2026-10-18T09:00:{end:02d}+09:00')

def test_exact_pid_timezone_and_half_open_boundaries():
    result=memory.summarize_gpu_memory_intervals(12,[observation(0,100),observation(10,200),observation(11,999,123),observation(20,300)],
        [interval('a',0,10),interval('b',10,20)])
    a,b=result['intervals']
    assert a['largest_observed_dedicated_bytes']==100 and b['largest_observed_dedicated_bytes']==200
    assert [a['sample_count'],b['sample_count']]==[1,1]
    assert result['guaranteed_peak_measured'] is False

def test_no_sample_and_a_real_zero_are_different():
    result=memory.summarize_gpu_memory_intervals(12,[observation(2,0)],[interval('zero',0,5),interval('missing',5,10)])
    zero,missing=result['intervals']
    assert zero['available'] is True and zero['largest_observed_dedicated_bytes']==0
    assert missing['available'] is False and missing['largest_observed_dedicated_bytes'] is None

def test_request_start_timestamp_is_not_valid_phase_attribution():
    result=memory.summarize_gpu_memory_intervals(12,[observation(2,100,source='request_start')],[interval('a',0,5)])
    assert result['rejected_timestamp_observations']==1
    assert result['intervals'][0]['largest_observed_dedicated_bytes'] is None

def test_invalid_status_and_failed_query_do_not_become_zero():
    bad=observation(2);bad['rows'][0]['status']=0xc0000bc6
    result=memory.summarize_gpu_memory_intervals(12,[bad,observation(3,error='counter unavailable')],[interval('a',0,5)])['intervals'][0]
    assert result['available'] is False
    assert result['largest_observed_dedicated_bytes'] is None
    assert result['largest_observed_shared_bytes']==50
    assert result['sample_count']==2 and result['available_sample_count']==0

def test_integral_counter_double_can_be_read_without_truncating_fractional_values():
    sample=observation(2);sample['rows'][0]['bytes']=100.0
    result=memory.summarize_gpu_memory_intervals(12,[sample],[interval('a',0,5)])
    assert result['intervals'][0]['largest_observed_dedicated_bytes']==100
    sample['rows'][0]['bytes']=100.5
    with pytest.raises(ValueError):memory.summarize_gpu_memory_intervals(12,[sample],[interval('a',0,5)])

def test_duplicate_instance_or_sample_timestamp_cannot_inflate_coverage():
    sample=observation(2);sample['rows'].append(counter(12,100))
    with pytest.raises(ValueError):memory.summarize_gpu_memory_intervals(12,[sample],[interval('a',0,5)])
    with pytest.raises(ValueError):memory.summarize_gpu_memory_intervals(12,[observation(2),observation(2)],[interval('a',0,5)])

@pytest.mark.parametrize('bad',[
    dict(id='naive',begin_utc='2026-10-18T00:00:00',end_utc='2026-10-18T00:00:05Z'),
    interval('reversed',5,0),dict(id='',begin_utc='2026-10-18T00:00:00Z',end_utc='2026-10-18T00:00:05Z')])
def test_bad_interval_boundaries_rejected(bad):
    with pytest.raises(ValueError):memory.summarize_gpu_memory_intervals(12,[],[bad])

def test_duplicate_interval_ids_and_invalid_pid_rejected():
    with pytest.raises(ValueError):memory.summarize_gpu_memory_intervals(12,[],[interval('a',0,5),interval('a',5,10)])
    with pytest.raises(ValueError):memory.summarize_gpu_memory_intervals(True,[],[])

def test_maximum_is_observed_and_gaps_remain_visible():
    result=memory.summarize_gpu_memory_intervals(12,[observation(1,200),observation(4,100)],[interval('a',0,5)])['intervals'][0]
    assert result['largest_observed_dedicated_bytes']==200
    assert result['maximum_observed_sample_gap_seconds']==3
