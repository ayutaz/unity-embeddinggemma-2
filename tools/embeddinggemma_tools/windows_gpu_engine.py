"""Per-engine raw Timer100Ns observations; never exclusive SDK stage timings."""

from datetime import datetime, timedelta, timezone
import re

EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)
FREQUENCY = 10_000_000
INSTANCE = re.compile(r'pid_(\d+)_luid_0x[0-9a-f]+_0x[0-9a-f]+_phys_\d+_eng_\d+_engtype_[a-z0-9_]+', re.I)
UTC = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.(\d{1,7}))?(?:Z|[+-]\d{2}:\d{2})')


def _ticks(value):
    match = UTC.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError('An aware ISO timestamp with at most seven fractional digits is required')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)
    delta = parsed - EPOCH
    extra = int(match[1][6]) if match[1] and len(match[1]) == 7 else 0
    return (delta.days * 86400 + delta.seconds) * FREQUENCY + delta.microseconds * 10 + extra


def _natural(value):
    if type(value) is not int or not 0 <= value <= 2**64 - 1:
        raise ValueError('Nonnegative integer counter metadata is required')
    return value


def _iso(ticks):
    instant = EPOCH + timedelta(seconds=ticks // FREQUENCY)
    return instant.strftime('%Y-%m-%dT%H:%M:%S') + f'.{ticks % FREQUENCY:07d}Z'


def summarize_engine_intervals(pid, created_utc, observations, intervals):
    """Keep each engine separate and use only sample pairs inside the interval.

    Read the raw numerator of the documented Timer100Ns counter, not cooked
    percentage or the untyped Running Time counter. No endpoint interpolation,
    process reuse, reset, missing-engine bridge or exclusive SDK attribution.
    """
    if type(pid) is not int or pid <= 0:
        raise ValueError('A positive process ID is required')
    created = _ticks(created_utc)
    snapshots, rejected, untrusted = [], 0, False
    last = None
    for observation in observations:
        if type(observation.get('pid')) is not int or observation['pid'] != pid:
            continue
        if _ticks(observation.get('process_created_utc')) != created:
            rejected += 1
            continue
        if observation.get('timestamp_source') != 'counter_sample_set':
            rejected += 1
            untrusted = True
            continue
        instant = _natural(observation.get('timestamp_100ns'))
        if last is not None and instant <= last:
            raise ValueError('Distinct increasing counter sample timestamps are required')
        last = instant
        rows = {}
        for row in observation.get('rows', []):
            name = row.get('name')
            match = INSTANCE.fullmatch(name) if isinstance(name, str) else None
            if match is None:
                raise ValueError('A complete GPU process/adapter/engine identity is required')
            if int(match[1]) != pid:
                continue
            if row.get('counter', '').casefold() != 'utilization percentage':
                continue
            if row.get('counter_type') != 'Timer100Ns' or type(row.get('time_base')) is not int or row['time_base'] != FREQUENCY:
                raise ValueError('Only raw Timer100Ns data with the verified time base is supported')
            if _natural(row.get('timestamp_100ns')) != instant:
                raise ValueError('Counter timestamps must match the actual sample set')
            if name in rows:
                raise ValueError('Duplicate engine instance in one sample')
            rows[name] = dict(raw=_natural(row.get('raw_value')), status=_natural(row.get('status')))
        snapshots.append((instant, rows, bool(observation.get('error'))))
    result, ids = [], set()
    for interval in intervals:
        name = interval.get('id')
        if not isinstance(name, str) or not name.strip() or name in ids:
            raise ValueError('Distinct nonempty interval IDs are required')
        ids.add(name)
        begin, end = _ticks(interval['begin_utc']), _ticks(interval['end_utc'])
        if end < begin:
            raise ValueError('Interval end precedes beginning')
        samples = [row for row in snapshots if begin <= row[0] <= end]
        engines = []
        for engine in sorted({engine for _, rows, _ in samples for engine in rows}):
            reasons = set()
            if untrusted:
                reasons.add('untrusted_timestamp_observation')
            if len(samples) < 2:
                reasons.add('insufficient_samples')
            if any(error for _, _, error in samples):
                reasons.add('provider_error')
            if any(engine not in rows for _, rows, _ in samples):
                reasons.add('missing_engine')
            pairs = [(instant, rows[engine]) for instant, rows, _ in samples if engine in rows]
            if any(value['status'] not in (0, 1) for _, value in pairs):
                reasons.add('invalid_status')
            for (t0, a), (t1, b) in zip(pairs, pairs[1:]):
                if b['raw'] < a['raw']:
                    reasons.add('counter_reset')
                elif b['raw'] - a['raw'] > t1 - t0:
                    reasons.add('active_time_exceeds_elapsed')
            available = not reasons
            engines.append(dict(name=engine, available=available, sample_count=len(pairs),
                first_counter_sample_utc=_iso(pairs[0][0]), last_counter_sample_utc=_iso(pairs[-1][0]),
                covered_elapsed_seconds=(pairs[-1][0] - pairs[0][0]) / FREQUENCY if len(pairs) >= 2 else None,
                active_seconds=(pairs[-1][1]['raw'] - pairs[0][1]['raw']) / FREQUENCY if available else None,
                rejection_reasons=sorted(reasons)))
        result.append(dict(id=name, begin_utc=interval['begin_utc'], end_utc=interval['end_utc'],
            sample_count=len(samples), engines=engines,
            boundary_coverage_complete=bool(samples and engines and all(e['available'] for e in engines)
                and samples[0][0] == begin and samples[-1][0] == end)))
    return dict(pid=pid, process_created_utc=created_utc, intervals=result, rejected_observations=rejected,
        exclusive_sdk_stage_time_measured=False, gpu_execution_validated=False,
        scope='Per-process, per-engine raw Timer100Ns deltas between counter samples inside each interval. Engines may overlap and are never summed; no boundary interpolation or SDK-exclusive time.')
