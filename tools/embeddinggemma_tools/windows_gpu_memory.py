"""Windows per-process GPU memory observations; never GPU execution timings."""

from datetime import datetime, timezone
import math


def summarize_gpu_memory(pid, dedicated, shared):
    if type(pid) is not int or pid <= 0:
        raise ValueError('A positive process ID is required')

    def total(rows):
        matching = [row for row in rows if row['name'].startswith(f'pid_{pid}_')]
        if not matching:
            return None
        names = [row['name'] for row in matching]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate GPU memory instance')
        valid = True
        for row in matching:
            if row['status'] not in (0, 1):
                valid = False
                continue
            if type(row['bytes']) is not int or row['bytes'] < 0:
                raise ValueError('Invalid allocation byte count')
        return sum(row['bytes'] for row in matching) if valid else None

    dedicated_bytes, shared_bytes = total(dedicated), total(shared)
    return dict(pid=pid, dedicated_bytes=dedicated_bytes, shared_bytes=shared_bytes,
                available=dedicated_bytes is not None and shared_bytes is not None,
                gpu_execution_validated=False,
                scope='sampled_process_allocations_not_capacity_or_peak_or_gpu_time')


def summarize_gpu_memory_intervals(pid, observations, intervals):
    """Correlate actual counter sample times with half-open UTC intervals.

    Report sampled maxima and gaps, never a guaranteed peak or zero for missing
    data. Request-start timestamps cannot establish stage attribution.
    """
    summarize_gpu_memory(pid, [], [])

    def utc(value):
        if not isinstance(value, str):
            raise ValueError('An aware ISO timestamp is required')
        try:
            parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError as error:
            raise ValueError('An aware ISO timestamp is required') from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError('A timezone is required')
        return parsed.astimezone(timezone.utc)

    records, seen, rejected = [], set(), 0
    for observation in observations:
        if type(observation.get('pid')) is not int or observation['pid'] != pid:
            continue
        if observation.get('timestamp_source') != 'counter_sample_set':
            rejected += 1
            continue
        instant = utc(observation['utc'])
        if instant in seen:
            raise ValueError('Duplicate counter sample timestamp')
        seen.add(instant)
        dedicated, shared = [], []
        if not observation.get('error'):
            for row in observation.get('rows', []):
                if not row['name'].startswith(f'pid_{pid}_'):
                    continue
                counter = row['counter'].casefold()
                if counter not in ('dedicated usage', 'shared usage'):
                    continue
                value, status = row['bytes'], row['status']
                if type(value) is float and math.isfinite(value) and value.is_integer():
                    value = int(value)
                if type(value) is not int or value < 0 or type(status) is not int:
                    raise ValueError('Invalid GPU allocation observation')
                target = dedicated if counter == 'dedicated usage' else shared
                target.append(dict(name=row['name'], bytes=value, status=status))
        records.append((instant, summarize_gpu_memory(pid, dedicated, shared)))
    records.sort(key=lambda pair: pair[0])
    rows, names = [], set()
    for interval in intervals:
        name = interval['id']
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError('Distinct nonempty interval IDs are required')
        names.add(name)
        begin, end = utc(interval['begin_utc']), utc(interval['end_utc'])
        if end < begin:
            raise ValueError('Interval end precedes its beginning')
        samples = [(instant, value) for instant, value in records if begin <= instant < end]
        paired = [(instant, value) for instant, value in samples if value['available']]
        dedicated = [value['dedicated_bytes'] for _, value in samples if value['dedicated_bytes'] is not None]
        shared = [value['shared_bytes'] for _, value in samples if value['shared_bytes'] is not None]
        gaps = [(b[0]-a[0]).total_seconds() for a, b in zip(paired, paired[1:])]
        rows.append(dict(id=name, begin_utc=begin.isoformat(), end_utc=end.isoformat(),
            sample_count=len(samples), available_sample_count=len(paired), available=bool(paired),
            largest_observed_dedicated_bytes=max(dedicated, default=None),
            largest_observed_shared_bytes=max(shared, default=None),
            first_paired_sample_utc=paired[0][0].isoformat() if paired else None,
            last_paired_sample_utc=paired[-1][0].isoformat() if paired else None,
            maximum_observed_sample_gap_seconds=max(gaps, default=None)))
    return dict(pid=pid, intervals=rows, rejected_timestamp_observations=rejected,
        guaranteed_peak_measured=False, gpu_execution_validated=False,
        scope='counter_sample_timestamp_correlated_process_allocations_not_capacity_or_guaranteed_peak_or_gpu_time')
