"""Clock uncertainty and per-engine gaps are not whole-stage SDK GPU times."""
from copy import deepcopy

import pytest

from embeddinggemma_tools.native_pdh import summarize_pdh_intervals
from embeddinggemma_tools.windows_gpu_engine import _ticks,_iso

TICKS = _ticks('2026-10-18T00:00:00Z')
OFFSET = 540*60*10_000_000
NAME = 'pid_12_luid_0x0_0x1_phys_0_eng_0_engtype_3D'


def sample(second,active):
    tick = TICKS+second*10_000_000
    return dict(pid=12,process_created_utc_filetime=42,counter_type=0x20510500,time_base=10_000_000,
        utc_offset_minutes_at_collection=540,query_timestamp_raw_100ns=tick+OFFSET,
        request_begin_utc=_iso(tick+500),request_end_utc=_iso(tick+25000),
        process_handle_alive_before=True,process_handle_alive_after=True,
        rows=[dict(name=NAME,raw_value=active,timestamp_raw_100ns=tick+OFFSET+100,status=0)])


def summarize(rows,bounds=None):
    return summarize_pdh_intervals(12,42,rows,[bounds or dict(id='load',begin_utc=_iso(TICKS-100000),end_utc=_iso(TICKS+30100000))],utc_offset_minutes=540)


def test_item_clock_differs_from_query_clock_and_engines_are_not_summed():
    rows = [sample(0,10),sample(1,2_000_010),sample(3,6_000_010)]
    for row in rows:
        other = deepcopy(row['rows'][0]); other['name'] = NAME.replace('eng_0_','eng_1_')
        row['rows'].append(other)
    result = summarize(rows)
    assert [e['active_seconds'] for e in result['intervals'][0]['engines']] == [.6,.6]
    assert result['intervals'][0]['engines'][0]['first_counter_sample_utc'] == _iso(TICKS+100)
    assert result['clock_tolerance_ticks'] == 10000
    assert result['full_stage_gpu_time_measured'] is False and result['exclusive_sdk_stage_time_measured'] is False
    assert 'total_gpu_seconds' not in result['intervals'][0]


def test_acquisition_must_fit_inside_stage_with_one_ms_clock_margin():
    bounds = dict(id='release',begin_utc=_iso(TICKS),end_utc=_iso(TICKS+2*10_000_000))
    result = summarize([sample(0,0),sample(1,10),sample(2,20)],bounds)['intervals'][0]
    assert result['sample_count'] == 1
    assert result['engines'][0]['active_seconds'] is None
    assert result['engines'][0]['rejection_reasons'] == ['insufficient_samples']


@pytest.mark.parametrize('change',['status','missing','reset','exceeds','dead'])
def test_status_gap_reset_or_process_exit_never_produces_endpoint_only_time(change):
    rows = [sample(0,100),sample(1,200),sample(3,300)]
    if change == 'status': rows[1]['rows'][0]['status'] = 0xc0000bc6
    if change == 'missing': rows[1]['rows'] = []
    if change == 'reset': rows[1]['rows'][0]['raw_value'] = 1
    if change == 'exceeds': rows[1]['rows'][0]['raw_value'] = 100_000_000
    if change == 'dead': rows[1]['process_handle_alive_after'] = False
    engine = summarize(rows)['intervals'][0]['engines'][0]
    assert not engine['available'] and engine['active_seconds'] is None and engine['rejection_reasons']


@pytest.mark.parametrize('change',['offset','clock','creation','type','duplicate'])
def test_unverified_clock_identity_or_duplicate_engine_is_rejected(change):
    row = sample(1,100)
    if change == 'offset': row['utc_offset_minutes_at_collection'] = 0
    if change == 'clock': row['rows'][0]['timestamp_raw_100ns'] += 2*10_000_000
    if change == 'creation': row['process_created_utc_filetime'] = 43
    if change == 'type': row['counter_type'] = 1
    if change == 'duplicate': row['rows'] *= 2
    with pytest.raises(ValueError): summarize([row])


def test_valid_zero_foreign_pid_and_no_engine_have_distinct_meanings():
    rows = [sample(0,10),sample(3,10)]
    foreign = sample(1,999);foreign['pid'] = 123
    result = summarize([rows[0],foreign,rows[1]])['intervals'][0]
    assert result['engines'][0]['available'] and result['engines'][0]['active_seconds'] == 0
    for row in rows: row['rows'] = []
    assert summarize(rows)['intervals'][0]['engines'] == []
