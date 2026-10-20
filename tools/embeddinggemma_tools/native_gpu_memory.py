"""Persistent, read-only Windows GPU allocation samples for one process lifetime.

These are observed process allocations, never guaranteed peaks or SDK-exclusive
memory. Raw provider FILETIMEs use the declared local offset and must also fit
the actual UTC collection envelope. No inference backend is loaded or changed.
"""
import ctypes as C
from datetime import datetime, timezone
import os
import re
import time

from .native_pdh import _NativePdh, _RawItem, MORE_DATA
from .windows_gpu_engine import _ticks, _iso
from .windows_gpu_memory import summarize_gpu_memory

RAW_BYTES = 0x10100
KINDS = ('dedicated usage', 'shared usage')
INSTANCE = re.compile(r'pid_(\d+)_luid_0x[0-9a-f]+_0x[0-9a-f]+_phys_\d+', re.I)


class PdhGpuMemoryQuery:
    def __init__(self, pid, *, backend=None, identity=None, utc_offset_minutes):
        if type(pid) is not int or pid <= 0:
            raise ValueError('An exact positive process ID is required')
        if type(utc_offset_minutes) is not int or not -840 <= utc_offset_minutes <= 840:
            raise ValueError('Declare the observed local UTC offset')
        self.pid, self.offset = pid, utc_offset_minutes
        self.closed = True
        self.identity, self.backend = None, None
        try:
            self.identity = identity if identity is not None else _ProcessIdentity(pid)
            if type(self.identity.created_filetime) is not int or self.identity.created_filetime <= 0 or not self.identity.is_alive():
                raise ValueError('A live process creation identity is required')
            self.backend = backend if backend is not None else _MemoryBackend()
            if self.backend.open() != {kind: RAW_BYTES for kind in KINDS}:
                raise ValueError('Both counters must declare raw allocation byte counts')
            self.closed = False
        except BaseException:
            if self.backend is not None: self.backend.close()
            if self.identity is not None: self.identity.close()
            raise

    def sample(self):
        if self.closed:
            raise RuntimeError('The memory query is closed')
        if not self.identity.is_alive():
            raise ValueError('The original process lifetime has ended')
        begin = datetime.now(timezone.utc).isoformat()
        wall, cpu = time.perf_counter_ns(), time.process_time_ns()
        packet = self.backend.collect()
        elapsed_wall, elapsed_cpu = time.perf_counter_ns()-wall, time.process_time_ns()-cpu
        end = datetime.now(timezone.utc).isoformat()
        if not self.identity.is_alive():
            raise ValueError('The original process ended during collection')
        offset = self.offset*60*10_000_000
        lower, upper = _ticks(begin)-10000, _ticks(end)+10000
        def mapped(value):
            if type(value) is not int or not 0 <= value <= 2**64-1 or not lower <= value-offset <= upper:
                raise ValueError('The raw provider clock is outside the actual collection envelope')
            return value-offset
        moment = mapped(packet['query_timestamp_100ns'])
        rows, grouped = [], {}
        if set(packet['instances']) != set(KINDS):
            raise ValueError('Both allocation counter arrays are required')
        for kind in KINDS:
            found, names = [], set()
            for row in packet['instances'][kind]:
                name = row.get('name')
                if not isinstance(name,str) or not name.casefold().startswith(f'pid_{self.pid}_'):
                    continue
                match = INSTANCE.fullmatch(name)
                if match is None or int(match[1]) != self.pid or name.casefold() in names:
                    raise ValueError('Distinct full owned process and adapter identities are required')
                names.add(name.casefold())
                value, status = row['raw_value'], row['status']
                if type(status) is not int or status not in (0,1) or type(value) is not int or not 0 <= value <= 2**63-1:
                    raise ValueError('Invalid raw allocation counter data')
                mapped(row['timestamp_100ns'])
                found.append(dict(name=name,bytes=value,status=status))
                rows.append(dict(name=name,counter=kind,bytes=value,status=status,timestamp_raw_100ns=row['timestamp_100ns']))
            grouped[kind] = found
        summary = summarize_gpu_memory(self.pid,grouped[KINDS[0]],grouped[KINDS[1]])
        paired = {r['name'].casefold() for r in grouped[KINDS[0]]} == {r['name'].casefold() for r in grouped[KINDS[1]]}
        if not paired: summary['available'] = False
        return dict(summary,utc=_iso(moment),timestamp_source='counter_sample_set',rows=rows,
            error=None if paired else 'allocation_adapter_sets_differ',
            query_timestamp_raw_100ns=packet['query_timestamp_100ns'],utc_offset_minutes=self.offset,
            process_created_utc_filetime=self.identity.created_filetime,process_handle_alive_before=True,
            process_handle_alive_after=True,request_begin_utc=begin,request_end_utc=end,
            collection_wall_ns=elapsed_wall,collection_process_cpu_ns=elapsed_cpu,clock_consistency_checked=True,
            clock_consistency_margin_ticks=10000,guaranteed_peak_measured=False,GPU_execution_validated=False)

    def close(self):
        if not self.closed:
            self.closed = True
            try: self.backend.close()
            finally: self.identity.close()

    def __enter__(self): return self
    def __exit__(self,*_): self.close()


class _ProcessIdentity:
    def __init__(self,pid):
        if os.name != 'nt': raise OSError('Windows process identity is required')
        self.dll = C.WinDLL('kernel32',use_last_error=True)
        bindings = dict(OpenProcess=([C.c_uint32,C.c_int,C.c_uint32],C.c_void_p),
            GetProcessTimes=([C.c_void_p,*([C.POINTER(C.c_uint64)]*4)],C.c_int),
            WaitForSingleObject=([C.c_void_p,C.c_uint32],C.c_uint32),CloseHandle=([C.c_void_p],C.c_int))
        for name,(args,result) in bindings.items():
            function=getattr(self.dll,name);function.argtypes,function.restype=args,result
        self.handle=self.dll.OpenProcess(0x1000|0x100000,False,pid)
        if not self.handle: raise OSError('The target process could not be opened read-only')
        try:
            created,ended,kernel,user=(C.c_uint64() for _ in range(4))
            if not self.dll.GetProcessTimes(self.handle,C.byref(created),C.byref(ended),C.byref(kernel),C.byref(user)):
                raise OSError('Process creation identity could not be read')
            self.created_filetime=created.value
        except BaseException:
            self.close();raise
    def is_alive(self):
        status=self.dll.WaitForSingleObject(self.handle,0)
        if status not in (0,0x102): raise OSError('Process lifetime could not be verified')
        return status==0x102
    def close(self):
        if self.handle:
            handle,self.handle=self.handle,None
            if not self.dll.CloseHandle(handle): raise OSError('Process identity handle could not be closed')


class _MemoryBackend(_NativePdh):
    def open(self):
        self.check(self.dll.PdhOpenQueryW(None,0,C.byref(self.query)))
        self.counters={}
        metadata={}
        for kind in KINDS:
            handle=C.c_void_p()
            path='\\GPU Process Memory(*)\\'+kind.title()
            self.check(self.dll.PdhAddEnglishCounterW(self.query,path,0,C.byref(handle)))
            self.counters[kind]=handle
            length=C.c_uint32()
            if self.dll.PdhGetCounterInfoW(handle,False,C.byref(length),None)!=MORE_DATA:
                raise OSError('Memory counter metadata size was not returned')
            info=self.buffer(length.value)
            self.check(self.dll.PdhGetCounterInfoW(handle,False,C.byref(length),info))
            metadata[kind]=C.c_uint32.from_buffer(info,4).value
        return metadata

    def collect(self):
        stamp=C.c_int64()
        self.check(self.dll.PdhCollectQueryDataWithTime(self.query,C.byref(stamp)))
        arrays={}
        for kind,handle in self.counters.items():
            length,count=C.c_uint32(),C.c_uint32()
            status=self.dll.PdhGetRawCounterArrayW(handle,C.byref(length),C.byref(count),None)
            if status!=MORE_DATA:
                self.check(status)
                if count.value==0: arrays[kind]=[];continue
                raise OSError('Memory counter array size was not returned')
            buffer=self.buffer(length.value)
            self.check(self.dll.PdhGetRawCounterArrayW(handle,C.byref(length),C.byref(count),buffer))
            if count.value*C.sizeof(_RawItem)>C.sizeof(buffer):
                raise ValueError('Memory counter item count exceeds its allocated buffer')
            arrays[kind]=[dict(name=item.name,raw_value=item.value.first,status=item.value.status,
                timestamp_100ns=(item.value.timestamp.high<<32)|item.value.timestamp.low)
                for item in (_RawItem*count.value).from_buffer(buffer)]
        return dict(query_timestamp_100ns=stamp.value,instances=arrays)
