"""Windows per-process GPU memory observations; never GPU execution timings."""


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
