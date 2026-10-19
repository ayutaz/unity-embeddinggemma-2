"""Selected numerical CPU context without machine names, serials or paths."""
import json
import os
from pathlib import Path
import re
import sys

import torch

from .privacy import scan_bytes


CPU_FIELDS = frozenset({'vendor_id', 'cpu family', 'model', 'model name', 'stepping',
                        'CPU implementer', 'CPU architecture', 'CPU part'})
MATH_ENVIRONMENT = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'MKL_CBWR', 'MKL_DYNAMIC',
                    'MKL_ENABLE_INSTRUCTIONS', 'ATEN_CPU_CAPABILITY', 'DNNL_MAX_CPU_ISA')


def _windows_cpu_model():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            values = {}
            for name in ('ProcessorNameString', 'VendorIdentifier', 'Identifier'):
                try:
                    value, _ = winreg.QueryValueEx(key, name)
                except OSError:
                    continue
                if isinstance(value, str) and value.strip():
                    values[name] = value.strip()
    except (ImportError, OSError):
        return {}
    fields = {target: values[source] for source, target in (
        ('ProcessorNameString', 'model name'), ('VendorIdentifier', 'vendor_id')) if source in values}
    match = re.search(r'Family\s+(\d+)\s+Model\s+(\d+)\s+Stepping\s+(\d+)', values.get('Identifier', ''))
    if match:
        fields.update(zip(('cpu family', 'model', 'stepping'), match.groups()))
    return fields


def cpu_execution_context(*, cpuinfo_path=Path('/proc/cpuinfo'), environ=None):
    environment = os.environ if environ is None else environ
    fields = {}
    try:
        text = Path(cpuinfo_path).read_text(encoding='utf-8')
        for line in text.splitlines():
            if not line.strip() and fields:
                break
            name, separator, value = line.partition(':')
            if separator and name.strip() in CPU_FIELDS:
                fields.setdefault(name.strip(), value.strip())
    except (OSError, UnicodeError):
        pass
    if not fields and sys.platform == 'win32' and Path(cpuinfo_path) == Path('/proc/cpuinfo'):
        fields = _windows_cpu_model()
    # The complete Torch build report may contain build-machine paths. Select
    # only library version lines and inspect those values before returning them.
    libraries = [line.strip().removeprefix('- ') for line in torch.__config__.show().splitlines()
                 if line.strip().startswith(('- Intel(R)', '- oneDNN'))]
    result = dict(cpu_model=fields, cpu_model_available=bool(fields.get('model name')),
        CPU_capability=torch.backends.cpu.get_cpu_capability(), torch_runtime_version=str(torch.__version__),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        torch_num_threads=torch.get_num_threads(), torch_num_interop_threads=torch.get_num_interop_threads(),
        mkl_available=bool(torch.backends.mkl.is_available()),
        mkldnn_available=bool(torch.backends.mkldnn.is_available()),
        mkldnn_enabled=bool(torch.backends.mkldnn.enabled), math_libraries=libraries,
        math_environment={name: environment.get(name) for name in MATH_ENVIRONMENT},
        flush_denormal_state=None)
    if scan_bytes(json.dumps(result, ensure_ascii=False).encode('utf-8'), '<cpu-execution-context>'):
        raise ValueError('Selected CPU execution context failed privacy inspection')
    return result
