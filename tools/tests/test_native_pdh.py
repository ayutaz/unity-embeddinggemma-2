"""Independent protocol fixtures; these do not execute a GPU or a provider."""
from copy import deepcopy

import pytest

from embeddinggemma_tools.native_pdh import PdhGpuQuery

TYPE = 0x20510500  # PERF_100NSEC_TIMER
NAME = 'pid_12_luid_0x0_0x1_phys_0_eng_0_engtype_3D'


class Provider:
    def __init__(self):
        self.opened = self.closed = self.collected = 0
        self.metadata = dict(counter_type=TYPE,time_base=10_000_000)
        self.packet = dict(query_timestamp_100ns=134000000000000000,instances=[
            dict(name=NAME,raw_value=100,timestamp_100ns=134000000000000000,status=0)])
        self.error = None

    def open(self):
        self.opened += 1
        return self.metadata

    def collect(self):
        self.collected += 1
        if self.error: raise self.error
        return deepcopy(self.packet)

    def close(self):
        self.closed += 1


def test_reuses_one_query_and_keeps_raw_clock_and_exact_pid_only():
    provider = Provider()
    provider.packet['instances'].append(dict(name=NAME.replace('pid_12_','pid_123_'),raw_value=-999))
    with PdhGpuQuery(12,provider) as query:
        first = query.sample()
        provider.packet['instances'][0]['raw_value'] = 200
        second = query.sample()
    assert provider.opened == provider.closed == 1 and provider.collected == 2
    assert first['pid'] == 12 and len(first['rows']) == 1 and first['rows'][0]['raw_value'] == 100
    assert second['rows'][0]['raw_value'] == 200
    assert first['query_timestamp_raw_100ns'] == first['rows'][0]['timestamp_raw_100ns'] == 134000000000000000
    assert first['counter_type'] == TYPE and first['time_base'] == 10_000_000
    assert first['timestamp_mapping_verified'] is False and first['gpu_execution_validated'] is False
    assert first['request_end_utc'] >= first['request_begin_utc']
    assert first['collection_wall_ns'] >= 0 and first['collection_process_cpu_ns'] >= 0
    assert 'active_seconds' not in first and 'total_gpu_seconds' not in first


@pytest.mark.parametrize('pid',[0,-1,True,'12'])
def test_invalid_pid_is_rejected_before_open(pid):
    provider = Provider()
    with pytest.raises(ValueError): PdhGpuQuery(pid,provider)
    assert provider.opened == provider.closed == 0


@pytest.mark.parametrize(('field','value'),[('counter_type',1),('time_base',1)])
def test_unverified_type_or_timebase_closes_query_and_rejects(field,value):
    provider = Provider(); provider.metadata[field] = value
    with pytest.raises(ValueError):
        with PdhGpuQuery(12,provider): pytest.fail('Unverified counters must not enter.')
    assert provider.opened == provider.closed == 1


def test_missing_engine_is_empty_and_invalid_owned_status_is_preserved():
    provider = Provider()
    with PdhGpuQuery(12,provider) as query:
        provider.packet['instances'] = []
        assert query.sample()['rows'] == []
        provider.packet['instances'] = [dict(name=NAME,raw_value=10,timestamp_100ns=134000000000000000,status=0xc0000bc6)]
        assert query.sample()['rows'][0]['status'] == 0xc0000bc6


def test_provider_failure_closes_query_without_fake_sample():
    provider = Provider(); provider.error = OSError('provider failure')
    with pytest.raises(OSError,match='provider failure'):
        with PdhGpuQuery(12,provider) as query: query.sample()
    assert provider.closed == 1


@pytest.mark.parametrize(('field','value'),[('raw_value',-1),('timestamp_100ns',True),('name','pid_12_unknown')])
def test_invalid_owned_raw_metadata_is_rejected(field,value):
    provider = Provider(); provider.packet['instances'][0][field] = value
    with pytest.raises(ValueError):
        with PdhGpuQuery(12,provider) as query: query.sample()
    assert provider.closed == 1


def test_closed_query_cannot_collect_again_or_close_twice():
    provider = Provider()
    with PdhGpuQuery(12,provider) as query: query.sample()
    with pytest.raises(RuntimeError): query.sample()
    query.close()
    assert provider.closed == 1
