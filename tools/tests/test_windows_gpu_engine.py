"""Do not invent durations from missing samples, engine sums or PID reuse."""
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from embeddinggemma_tools.windows_gpu_engine import summarize_engine_intervals

CREATED = '2026-10-17T23:50:00Z'
BEGIN = '2026-10-18T00:00:00Z'
EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)
delta = datetime.fromisoformat(BEGIN.replace('Z', '+00:00')) - EPOCH
TICKS = (delta.days * 86400 + delta.seconds) * 10_000_000


def observation(second, active, *, engine=0, pid=12):
    return dict(pid=pid, process_created_utc=CREATED, timestamp_source='counter_sample_set',
        timestamp_100ns=TICKS + second * 10_000_000, rows=[dict(
        name=f'pid_{pid}_luid_0x0_0x1_phys_0_eng_{engine}_engtype_3D',
        counter='utilization percentage', counter_type='Timer100Ns', time_base=10_000_000,
        raw_value=active, timestamp_100ns=TICKS + second * 10_000_000, status=0)])


def interval(begin=0, end=3):
    return dict(id='load', begin_utc=f'2026-10-18T00:00:0{begin}Z', end_utc=f'2026-10-18T00:00:0{end}Z')


def summarize(rows, bounds=None):
    return summarize_engine_intervals(12, CREATED, rows, [bounds or interval()])


def test_uses_raw_timer_delta_at_exact_pid_and_inside_bounds_without_engine_sum():
    rows = [observation(0, 10), observation(1, 2_000_010), observation(3, 6_000_010), observation(4, 7_000_010)]
    rows += [observation(0, 9_000_000, pid=123), observation(3, 19_000_000, pid=123)]
    result = summarize(rows)
    row = result['intervals'][0]
    engine = row['engines'][0]
    assert engine['active_seconds'] == .6
    assert engine['covered_elapsed_seconds'] == 3
    assert engine['sample_count'] == 3
    assert row['boundary_coverage_complete'] is True
    assert result['exclusive_sdk_stage_time_measured'] is False
    assert result['gpu_execution_validated'] is False
    assert 'total_gpu_seconds' not in row and 'total_gpu_seconds' not in result


def test_keeps_overlapping_engines_separate():
    begin, end = observation(0, 0), observation(3, 20_000_000)
    begin['rows'] += observation(0, 0, engine=1)['rows']
    end['rows'] += observation(3, 20_000_000, engine=1)['rows']
    engines = summarize([begin, end])['intervals'][0]['engines']
    assert [r['active_seconds'] for r in engines] == [2, 2]


def test_partial_coverage_is_not_interpolated_to_stage_boundaries():
    result = summarize([observation(1, 0), observation(2, 1_000_000)])['intervals'][0]
    assert result['engines'][0]['active_seconds'] == .1
    assert result['engines'][0]['covered_elapsed_seconds'] == 1
    assert result['boundary_coverage_complete'] is False


def test_no_pair_and_valid_zero_are_different():
    missing = summarize([observation(1, 10)])['intervals'][0]['engines'][0]
    zero = summarize([observation(1, 10), observation(2, 10)])['intervals'][0]['engines'][0]
    assert missing['available'] is False and missing['active_seconds'] is None
    assert 'insufficient_samples' in missing['rejection_reasons']
    assert zero['available'] is True and zero['active_seconds'] == 0


@pytest.mark.parametrize('change', ['missing_engine', 'error', 'invalid_status', 'reset', 'exceeds_elapsed'])
def test_discontinuities_cannot_publish_an_endpoint_only_duration(change):
    rows = [observation(0, 100), observation(1, 200), observation(3, 300)]
    if change == 'missing_engine': rows[1]['rows'] = []
    if change == 'error': rows[1]['error'] = 'provider failed'
    if change == 'invalid_status': rows[1]['rows'][0]['status'] = 0xc0000bc6
    if change == 'reset': rows[1]['rows'][0]['raw_value'] = 1
    if change == 'exceeds_elapsed': rows[1]['rows'][0]['raw_value'] = 100_000_000
    result = summarize(rows)['intervals'][0]['engines'][0]
    assert result['available'] is False and result['active_seconds'] is None
    assert result['rejection_reasons']


def test_pid_reuse_and_request_start_timestamps_do_not_supply_samples():
    rows = [observation(0, 10), observation(3, 100)]
    rows[0]['process_created_utc'] = '2026-10-17T23:49:00Z'
    rows[1]['timestamp_source'] = 'request_start'
    result = summarize(rows)
    assert result['rejected_observations'] == 2
    assert result['intervals'][0]['engines'] == []


@pytest.mark.parametrize(('field', 'value'), [('raw_value', -1), ('raw_value', True),
    ('timestamp_100ns', True), ('time_base', 1), ('counter_type', 'NumberOfItems64')])
def test_malformed_timer_metadata_is_rejected(field, value):
    row = observation(1, 10)
    row['rows'][0][field] = value
    with pytest.raises(ValueError): summarize([row])


def test_duplicate_engine_or_counter_time_is_rejected():
    row = observation(1, 10)
    duplicate = deepcopy(row)
    duplicate['rows'] *= 2
    with pytest.raises(ValueError): summarize([duplicate])
    with pytest.raises(ValueError): summarize([row, row])


def test_seven_digit_utc_boundary_does_not_include_a_counter_sample_before_begin():
    bounds = interval()
    bounds['begin_utc'] = '2026-10-18T00:00:00.0000001Z'
    result = summarize([observation(0, 10), observation(1, 20), observation(3, 30)], bounds)['intervals'][0]
    assert result['engines'][0]['sample_count'] == 2
    assert result['engines'][0]['active_seconds'] == .000001
    assert result['boundary_coverage_complete'] is False
