"""Persistent read-only GPU Engine raw samples; no inferred SDK GPU times.

Both provider clocks are retained unchanged. A consumer must verify their
mapping to UTC before using them with stage boundaries.
"""
import ctypes as C
from datetime import datetime, timezone
import os
import time

from .windows_gpu_engine import INSTANCE

TIMER100NS = 0x20510500
FREQUENCY = 10_000_000
MORE_DATA = 0x800007D2
MAX_BUFFER = 16*1024*1024


def _natural(value):
    if type(value) is not int or not 0 <= value <= 2**64-1:
        raise ValueError('Raw counter metadata must be nonnegative integers')
    return value


class PdhGpuQuery:
    def __init__(self, pid, backend=None):
        if type(pid) is not int or pid <= 0:
            raise ValueError('An exact positive process ID is required')
        self.pid, self.backend, self.closed = pid, backend or _NativePdh(), True
        try:
            metadata = self.backend.open()
            if metadata != dict(counter_type=TIMER100NS,time_base=FREQUENCY):
                raise ValueError('The provider must declare PERF_100NSEC_TIMER and its verified time base')
            self.closed = False
        except BaseException:
            self.backend.close()
            raise

    def sample(self):
        if self.closed:
            raise RuntimeError('The owned PDH query is closed')
        begin = datetime.now(timezone.utc).isoformat()
        wall, cpu = time.perf_counter_ns(), time.process_time_ns()
        packet = self.backend.collect()
        elapsed_cpu, elapsed_wall = time.process_time_ns()-cpu, time.perf_counter_ns()-wall
        end = datetime.now(timezone.utc).isoformat()
        rows = []
        for item in packet['instances']:
            name = item.get('name')
            if not isinstance(name,str) or not name.lower().startswith(f'pid_{self.pid}_'):
                continue
            match = INSTANCE.fullmatch(name)
            if match is None or int(match[1]) != self.pid:
                raise ValueError('Owned rows require the full process/adapter/engine identity')
            rows.append(dict(name=name,raw_value=_natural(item['raw_value']),
                timestamp_raw_100ns=_natural(item['timestamp_100ns']),status=_natural(item['status'])))
        return dict(pid=self.pid,query_timestamp_raw_100ns=_natural(packet['query_timestamp_100ns']),
            request_begin_utc=begin,request_end_utc=end,collection_wall_ns=elapsed_wall,
            collection_process_cpu_ns=elapsed_cpu,rows=rows,counter_type=TIMER100NS,time_base=FREQUENCY,
            timestamp_mapping_verified=False,gpu_execution_validated=False,
            scope='Raw native PDH data for exact PID only. Query/row clocks remain unconverted; no stage duration or engine sum.')

    def close(self):
        if not self.closed:
            self.closed = True
            self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()


class _FileTime(C.Structure):
    _fields_ = [('low',C.c_uint32),('high',C.c_uint32)]


class _RawCounter(C.Structure):
    _fields_ = [('status',C.c_uint32),('timestamp',_FileTime),('first',C.c_int64),
                ('second',C.c_int64),('multiple',C.c_uint32)]


class _RawItem(C.Structure):
    _fields_ = [('name',C.c_wchar_p),('value',_RawCounter)]


class _NativePdh:
    def __init__(self):
        if os.name != 'nt':
            raise OSError('Native PDH collection requires Windows; this is not a skipped GPU pass')
        self.dll = C.WinDLL('pdh')
        handle, dword, quad = C.c_void_p, C.c_uint32, C.c_int64
        bindings = dict(PdhOpenQueryW=([C.c_wchar_p,C.c_size_t,C.POINTER(handle)],dword),
            PdhAddEnglishCounterW=([handle,C.c_wchar_p,C.c_size_t,C.POINTER(handle)],dword),
            PdhGetCounterInfoW=([handle,C.c_ubyte,C.POINTER(dword),C.c_void_p],dword),
            PdhGetCounterTimeBase=([handle,C.POINTER(quad)],dword),
            PdhCollectQueryDataWithTime=([handle,C.POINTER(quad)],dword),
            PdhGetRawCounterArrayW=([handle,C.POINTER(dword),C.POINTER(dword),C.c_void_p],dword),
            PdhCloseQuery=([handle],dword))
        for name,(args,result) in bindings.items():
            function = getattr(self.dll,name)
            function.argtypes, function.restype = args,result
        self.query, self.counter = handle(),handle()

    @staticmethod
    def check(status):
        if status:
            raise OSError(f'Native PDH returned status 0x{status:08x}')

    @staticmethod
    def buffer(size):
        if not 8 <= size <= MAX_BUFFER:
            raise ValueError('The provider returned an invalid or oversized PDH buffer')
        return C.create_string_buffer(size)

    def open(self):
        self.check(self.dll.PdhOpenQueryW(None,0,C.byref(self.query)))
        self.check(self.dll.PdhAddEnglishCounterW(self.query,r'\GPU Engine(*)\Utilization Percentage',0,C.byref(self.counter)))
        length = C.c_uint32()
        if self.dll.PdhGetCounterInfoW(self.counter,False,C.byref(length),None) != MORE_DATA:
            raise OSError('PDH counter metadata size was not returned')
        info = self.buffer(length.value)
        self.check(self.dll.PdhGetCounterInfoW(self.counter,False,C.byref(length),info))
        kind = C.c_uint32.from_buffer(info,4).value
        frequency = C.c_int64()
        self.check(self.dll.PdhGetCounterTimeBase(self.counter,C.byref(frequency)))
        return dict(counter_type=kind,time_base=frequency.value)

    def collect(self):
        stamp = C.c_int64()
        self.check(self.dll.PdhCollectQueryDataWithTime(self.query,C.byref(stamp)))
        length,count = C.c_uint32(),C.c_uint32()
        status = self.dll.PdhGetRawCounterArrayW(self.counter,C.byref(length),C.byref(count),None)
        if status != MORE_DATA:
            self.check(status)
            if count.value == 0:
                return dict(query_timestamp_100ns=stamp.value,instances=[])
            raise OSError('The provider did not size the raw PDH array')
        buffer = self.buffer(length.value)
        self.check(self.dll.PdhGetRawCounterArrayW(self.counter,C.byref(length),C.byref(count),buffer))
        if count.value*C.sizeof(_RawItem) > C.sizeof(buffer):
            raise ValueError('The PDH item count exceeds its allocated buffer')
        items = (_RawItem*count.value).from_buffer(buffer)
        return dict(query_timestamp_100ns=stamp.value,instances=[dict(name=item.name,
            raw_value=item.value.first,status=item.value.status,
            timestamp_100ns=(item.value.timestamp.high<<32)|item.value.timestamp.low) for item in items])

    def close(self):
        if self.query.value:
            query,self.query = self.query,C.c_void_p()
            self.check(self.dll.PdhCloseQuery(query))
