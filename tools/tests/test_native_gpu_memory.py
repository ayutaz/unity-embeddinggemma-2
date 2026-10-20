"""Persistent memory counters retain clocks, identity and missing-data semantics."""
from copy import deepcopy
import pytest
from embeddinggemma_tools.native_gpu_memory import PdhGpuMemoryQuery
from embeddinggemma_tools.windows_gpu_engine import _ticks
from embeddinggemma_tools.windows_gpu_memory import summarize_gpu_memory_intervals

NAME='pid_12_luid_0x0_0xaa_phys_0'

class Identity:
    created_filetime=42
    def __init__(self): self.alive=True;self.closed=False
    def is_alive(self): return self.alive
    def close(self): self.closed=True

class Backend:
    def __init__(self,change=None): self.change=change;self.closed=False;self.opens=0
    def open(self):
        self.opens+=1
        return {'dedicated usage':0x10100,'shared usage':0x10100}
    def collect(self):
        from datetime import datetime,timezone
        tick=_ticks(datetime.now(timezone.utc).isoformat())
        packet=dict(query_timestamp_100ns=tick,instances={kind:[dict(name=NAME,raw_value=value,status=0,timestamp_100ns=tick)] for kind,value in (('dedicated usage',123),('shared usage',45))})
        if self.change: self.change(packet)
        return packet
    def close(self): self.closed=True

def query(backend=None,identity=None,**kwargs):
    return PdhGpuMemoryQuery(12,backend=backend or Backend(),identity=identity or Identity(),utc_offset_minutes=0,**kwargs)

def test_persistent_query_pairs_owned_byte_counts_without_promoting_a_peak():
    backend=Backend();identity=Identity()
    with query(backend,identity) as collector:
        first=collector.sample();second=collector.sample()
    assert backend.opens==1 and backend.closed and identity.closed
    assert [r['bytes'] for r in first['rows']]==[123,45]
    assert first['pid']==12 and first['process_created_utc_filetime']==42
    assert first['timestamp_source']=='counter_sample_set'
    assert first['guaranteed_peak_measured'] is False
    assert first['GPU_execution_validated'] is False
    result=summarize_gpu_memory_intervals(12,[first,second],[dict(id='calls',begin_utc=first['request_begin_utc'],end_utc=second['request_end_utc'])])
    assert result['intervals'][0]['available'] and result['intervals'][0]['largest_observed_dedicated_bytes']==123

def test_foreign_rows_are_ignored_and_absent_owned_rows_remain_unknown():
    def foreign(p):
        for rows in p['instances'].values():rows[0]['name']=NAME.replace('pid_12_','pid_123_')
    with query(Backend(foreign)) as collector: result=collector.sample()
    assert result['rows']==[] and result['available'] is False
    assert result['dedicated_bytes'] is None and result['shared_bytes'] is None

def test_valid_zero_and_missing_shared_counter_are_distinct():
    def zero(p):
        for rows in p['instances'].values():rows[0]['raw_value']=0
    with query(Backend(zero)) as collector: result=collector.sample()
    assert result['available'] and result['dedicated_bytes']==result['shared_bytes']==0
    with query(Backend(lambda p:p['instances'].__setitem__('shared usage',[]))) as collector: result=collector.sample()
    assert result['available'] is False and result['shared_bytes'] is None

def test_different_adapter_sets_cannot_be_reported_as_a_paired_sample():
    def mismatch(p):p['instances']['shared usage'][0]['name']=NAME.replace('phys_0','phys_1')
    with query(Backend(mismatch)) as collector: result=collector.sample()
    assert result['available'] is False
    assert result['error']=='allocation_adapter_sets_differ'
    combined=summarize_gpu_memory_intervals(12,[result],[dict(id='calls',begin_utc=result['request_begin_utc'],end_utc=result['request_end_utc'])])
    assert combined['intervals'][0]['available'] is False

@pytest.mark.parametrize('fault',['duplicate','malformed','negative','fractional','bool','invalid_status','wrong_query_clock','wrong_item_clock'])
def test_invalid_owned_rows_or_unverified_clocks_fail_closed(fault):
    def corrupt(p):
        row=p['instances']['dedicated usage'][0]
        if fault=='duplicate':p['instances']['dedicated usage'].append(deepcopy(row))
        elif fault=='malformed':row['name']='pid_12_invalid'
        elif fault=='negative':row['raw_value']=-1
        elif fault=='fractional':row['raw_value']=1.5
        elif fault=='bool':row['raw_value']=True
        elif fault=='invalid_status':row['status']=0xc0000bc6
        elif fault=='wrong_query_clock':p['query_timestamp_100ns']+=100_000_000
        else:row['timestamp_100ns']-=100_000_000
    with query(Backend(corrupt)) as collector:
        with pytest.raises(ValueError):collector.sample()

def test_declared_local_filetime_offset_must_match_actual_call_clock():
    offset=540*60*10_000_000
    def local(p):
        p['query_timestamp_100ns']+=offset
        for rows in p['instances'].values():rows[0]['timestamp_100ns']+=offset
    with PdhGpuMemoryQuery(12,backend=Backend(local),identity=Identity(),utc_offset_minutes=540) as collector:
        assert collector.sample()['clock_consistency_checked']
    with query(Backend(local)) as collector:
        with pytest.raises(ValueError):collector.sample()

def test_process_exit_during_collection_cannot_observe_a_reused_pid():
    identity=Identity()
    with query(Backend(lambda _:setattr(identity,'alive',False)),identity) as collector:
        with pytest.raises(ValueError):collector.sample()

def test_metadata_mismatch_closes_both_resources():
    backend=Backend();backend.open=lambda:{'dedicated usage':0x20510500,'shared usage':0x10100}
    identity=Identity()
    with pytest.raises(ValueError):query(backend,identity)
    assert backend.closed and identity.closed

@pytest.mark.parametrize('pid',[0,-1,True,'12'])
def test_pid_must_be_an_exact_positive_integer(pid):
    with pytest.raises(ValueError):PdhGpuMemoryQuery(pid,backend=Backend(),identity=Identity(),utc_offset_minutes=0)

def test_closed_query_cannot_collect_more_data():
    collector=query();collector.close();collector.close()
    with pytest.raises(RuntimeError):collector.sample()
