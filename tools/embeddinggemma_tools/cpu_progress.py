"""Durable, path-free progress for one CPU comparison process."""
import ctypes
import json
import os
from pathlib import Path
import re
import sys
import time

from .bounded_process import write_json_atomic

PHASES = {'starting', 'load_part_begin', 'onnx_parse_begin', 'onnx_parse_end',
          'session_create_begin', 'session_create_end', 'load_part_end',
          'inference_part_begin', 'inference_part_end', 'release_part_begin', 'release_part_end', 'complete', 'failed'}
CASES = {'silence', 'impulse', 'sine', 'nyquist', 'noise', 'padded_noise'}


def own_process_memory():
    """Return whole-process resident/peak bytes; unavailable values stay null."""
    rss = peak = None
    if os.name == 'nt':
        class Counters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in ('peak', 'rss', 'peak_paged',
                'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
        from ctypes import wintypes
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        value = Counters(); value.cb = ctypes.sizeof(value)
        if get_memory(get_process(), ctypes.byref(value), value.cb):
            rss, peak = int(value.rss), int(value.peak)
    else:
        try:
            status = Path('/proc/self/status').read_text()
            values = {k: int(v) * 1024 for k, v in re.findall(r'^(VmRSS|VmHWM):\s+(\d+) kB$', status, re.M)}
            rss, peak = values.get('VmRSS'), values.get('VmHWM')
        except OSError:
            try:
                import resource
                peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                if sys.platform != 'darwin':
                    peak *= 1024
            except (ImportError, OSError):
                pass
    return dict(rss_bytes=rss if rss and rss > 0 else None,
                peak_rss_bytes=peak if peak and peak > 0 else None)


class ProgressJournal:
    def __init__(self, path, *, source, case_id, memory=own_process_memory):
        if re.fullmatch('[a-f0-9]{40}', source or '') is None or case_id not in CASES:
            raise ValueError('A registered source and original case are required')
        self.path, self.memory, self.started = Path(path), memory, time.perf_counter()
        if self.path.exists() or self.path.with_suffix(self.path.suffix + '.pending').exists():
            raise FileExistsError('Preserve the existing CPU progress evidence')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.report = dict(schema=1, scope='Whole owned CPU process progress and sampled resident/peak memory; not SDK-exclusive allocation',
            source=source, case_id=case_id, terminal=False, success=False, events=[],
            GPU_validated=False, complete_performance_accepted=False)
        with self.path.open('x', encoding='utf-8') as stream:
            json.dump(self.report, stream)
        self.record('starting')

    def record(self, phase, *, part=None, round_name=None):
        if (phase not in PHASES or part is not None and (type(part) is not int or not 0 <= part < 5)
                or round_name not in (None, 'first', 'warm', 'repeat-1', 'repeat-2')):
            raise ValueError('Only registered CPU progress fields may be published')
        memory = self.memory()
        event = dict(phase=phase, part=part, round=round_name,
                     elapsed_seconds=time.perf_counter() - self.started,
                     rss_bytes=memory['rss_bytes'], peak_rss_bytes=memory['peak_rss_bytes'])
        if any(v is not None and (type(v) is not int or v <= 0)
               for v in (event['rss_bytes'], event['peak_rss_bytes'])):
            raise ValueError('Memory observations must be positive bytes or explicit missing values')
        self.report['events'].append(event)
        write_json_atomic(self.path, self.report)
        print(json.dumps(event, allow_nan=False), flush=True)

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        self.report.update(terminal=True, success=kind is None)
        if kind is not None:
            name = kind.__name__
            self.report['error_class'] = name if name in {'RuntimeError', 'ValueError',
                'TypeError', 'MemoryError', 'OSError', 'FileExistsError', 'FileNotFoundError',
                'InvalidArgument', 'NotImplemented', 'Fail'} else 'OtherError'
        self.record('complete' if kind is None else 'failed')
        return False
