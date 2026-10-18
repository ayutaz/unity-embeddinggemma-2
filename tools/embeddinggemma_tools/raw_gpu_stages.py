"""Correlate complete target-PID GPU records with validated native-QPC stage windows."""

def summarize_raw_gpu_stages(pid, records, stages, bridges):
    """Keep per-queue unions; target-process activity is not exclusive SDK/kernel time."""
    def integer(row, key, minimum=0):
        value = row.get(key)
        if type(value) is not int or value < minimum:
            raise ValueError(f'Missing or invalid integer: {key}')
        return value

    if type(pid) is not int or pid <= 0:
        raise ValueError('Positive target PID required')
    if not 2 <= len(records) <= 200002 or not stages or len(bridges) < 2:
        raise ValueError('Complete finite capture, stages and clock bridges required')
    first, last = records[0], records[-1]
    if first.get('kind') != 'capture_begin' or last.get('kind') != 'capture_end':
        raise ValueError('Capture begin/end required')
    if any(row.get('pid') != pid for row in records):
        raise ValueError('Records must belong to the same target PID')
    frequency = integer(first, 'qpc_frequency', 1)
    integer(first, 'target_creation_filetime', 1)
    if integer(last, 'qpc_frequency', 1) != frequency:
        raise ValueError('Capture frequency changed')
    begin = integer(first, 'qpc_before')
    end = integer(last, 'qpc_after')
    if begin > integer(first, 'qpc_after') or integer(last, 'qpc_before') > end or begin > end:
        raise ValueError('Capture clocks are unordered')
    for key in ('capture_valid', 'session_stopped', 'target_exited'):
        if last.get(key) is not True:
            raise ValueError(f'Capture evidence absent: {key}')
    if last.get('stop_had_failure') is not False:
        raise ValueError('Stop failure or missing stop evidence')
    for key in ('events_lost', 'buffers_lost', 'decode_failures', 'invalid_records',
                'dropped_records', 'diagnostics', 'open_owned_executions', 'start_status',
                'd3d_enable_status', 'dxg_enable_status', 'd3d_rundown_status',
                'dxg_rundown_status', 'stop_status', 'process_trace_status', 'close_trace_status'):
        if integer(last, key) != 0:
            raise ValueError(f'Incomplete capture: {key}')
    if integer(last, 'record_count') != len(records)-2:
        raise ValueError('Capture row count mismatch')
    if any(row.get('kind') in ('capture_begin', 'capture_end') for row in records[1:-1]):
        raise ValueError('Multiple captures cannot be combined')

    previous_bridge = -1
    for bridge in bridges:
        b = integer(bridge, 'native_qpc_before')
        sample = integer(bridge, 'counter_ticks')
        e = integer(bridge, 'native_qpc_after')
        if not begin <= b <= sample <= e <= end or b < previous_bridge:
            raise ValueError('Native/runtime QPC bridge is unordered or outside capture')
        if bridge.get('validated') is not True or bridge.get('high_resolution') is not True or bridge.get('clock_source') != 'QueryPerformanceCounter':
            raise ValueError('High-resolution native/runtime bridge required')
        if any(integer(bridge, key, 1) != frequency
               for key in ('frequency_hz', 'native_frequency_hz', 'counter_frequency_hz')):
            raise ValueError('Runtime/native frequency mismatch')
        previous_bridge = b

    to_ns = lambda ticks: ticks*1_000_000_000//frequency
    work = []
    for row in records[1:-1]:
        if row.get('kind') != 'gpu_work':
            continue
        b, e = integer(row, 'begin_ns'), integer(row, 'end_ns')
        queue = integer(row, 'hardware_queue', 1)
        if not to_ns(begin) <= b <= e <= to_ns(end):
            raise ValueError('GPU interval unordered or outside observed capture')
        work.append((queue, b, e))

    def union_ns(spans):
        total, current = 0, None
        for b, e in sorted(spans):
            if current is None:
                current = (b, e)
            elif b <= current[1]:
                current = (current[0], max(current[1], e))
            else:
                total += current[1]-current[0]
                current = (b, e)
        return total + (current[1]-current[0] if current else 0)

    result, ids, previous_end = [], set(), begin
    for stage in stages:
        name = stage.get('id')
        if not isinstance(name, str) or not name or name in ids:
            raise ValueError('Unique nonempty stage IDs required')
        ids.add(name)
        b, e = integer(stage, 'begin_ticks'), integer(stage, 'end_ticks')
        if stage.get('timestamp_unit') != 'qpc_ticks' or stage.get('clock_source') != 'QueryPerformanceCounter' or integer(stage, 'frequency_hz', 1) != frequency:
            raise ValueError('Raw stage QPC frequency/units required')
        if not previous_end <= b <= e <= end:
            raise ValueError('Stages unordered, overlapping or outside capture')
        previous_end = e
        if not integer(bridges[0], 'native_qpc_after') <= b <= e <= integer(bridges[-1], 'native_qpc_before'):
            raise ValueError('Clock bridges must cover the measured stages')
        b, e = to_ns(b), to_ns(e)
        queues, count, crossing = {}, 0, 0
        if b < e:
            for queue, work_begin, work_end in work:
                if work_begin < e and work_end > b:
                    queues.setdefault(queue, []).append((max(b, work_begin), min(e, work_end)))
                    count += 1
                    crossing += work_begin < b or work_end > e
        result.append(dict(id=name, begin_ns=b, end_ns=e, gpu_work_records=count,
            boundary_crossing_records=crossing,
            queues=[dict(hardware_queue=queue, observed_union_ns=union_ns(spans))
                    for queue, spans in sorted(queues.items())]))
    return dict(raw_stage_correlation_accepted=True, exclusive_sdk_gpu_time_measured=False,
        scope='Target-process OnGpuWork intervals clipped to raw QPC API windows; per-queue unions, no engine sum or pure-kernel attribution.',
        pid=pid, frequency_hz=frequency, stages=result)
