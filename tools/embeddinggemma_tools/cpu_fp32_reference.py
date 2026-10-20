"""Optional strict CPU reference arithmetic; never loaded by the Unity SDK.

Explicit hardware FP32 FMA, sequential 192-wide partials and separate block
adds follow the existing reference lowering. A native helper is built locally,
hash-bound and used only when explicitly selected for numerical diagnosis.
"""
import ctypes as C
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import threading

import numpy as np

SOURCE=Path(__file__).resolve().parents[1]/'native_reference/block192_fma.cpp'

def _sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def _compiler():
    if os.name=='nt':
        where=Path(os.environ.get('ProgramFiles(x86)',''))/'Microsoft Visual Studio/Installer/vswhere.exe'
        if not where.is_file():raise RuntimeError('An installed MSVC C++ toolchain is required')
        root=Path(subprocess.check_output([str(where),'-latest','-requires','Microsoft.VisualStudio.Component.VC.Tools.x86.x64','-property','installationPath'],stderr=subprocess.PIPE).decode().strip())
        versions=[p for p in (root/'VC/Tools/MSVC').iterdir() if p.is_dir()]
        vc=max(versions,key=lambda p:tuple(int(v) for v in p.name.split('.')))
        sdk=Path(os.environ.get('ProgramFiles(x86)',''))/'Windows Kits/10'
        versions=[p for p in (sdk/'Include').iterdir() if p.is_dir() and (p/'ucrt').is_dir() and (sdk/'Lib'/p.name/'um/x64').is_dir()]
        version=max(versions,key=lambda p:tuple(int(v) for v in p.name.split('.'))).name
        executable=vc/'bin/Hostx64/x64/cl.exe'
        if not executable.is_file():raise RuntimeError('The x64 MSVC compiler is missing')
        environment=dict(os.environ)
        environment['PATH']=str(executable.parent)+os.pathsep+environment.get('PATH','')
        environment['INCLUDE']=os.pathsep.join(str(p) for p in (vc/'include',sdk/'Include'/version/'ucrt',sdk/'Include'/version/'um',sdk/'Include'/version/'shared'))
        environment['LIB']=os.pathsep.join(str(p) for p in (vc/'lib/x64',sdk/'Lib'/version/'ucrt/x64',sdk/'Lib'/version/'um/x64'))
        return executable,['/nologo','/LD','/O2','/MD','/EHsc','/std:c++17','/fp:strict','/arch:AVX2'],environment,'msvc'
    executable=shutil.which('c++')
    if not executable:raise RuntimeError('A local C++ compiler is required')
    return Path(executable),['-std=c++17','-O2','-shared','-fPIC','-pthread','-mavx2','-mfma','-mxsave','-fno-fast-math','-ffp-contract=off'],dict(os.environ),'c++'

class NativeFp32Reference:
    @classmethod
    def build(cls,directory):
        if platform.machine().casefold() not in ('amd64','x86_64'):
            raise RuntimeError('This explicit reference helper requires x64 AVX2/FMA')
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        executable,flags,environment,kind=_compiler()
        library=directory/('reference.dll' if os.name=='nt' else 'reference.so')
        manifest=directory/'build.json'
        expected=dict(source_sha256=_sha(SOURCE),compiler_binary_sha256=_sha(executable),compiler_kind=kind,compiler_flags=flags)
        if library.exists() or manifest.exists():
            if not library.is_file() or not manifest.is_file():raise ValueError('Incomplete reference build; preserve its diagnostics')
            stored=json.loads(manifest.read_bytes())
            if any(stored.get(k)!=v for k,v in expected.items()) or stored.get('native_library_sha256')!=_sha(library):
                raise ValueError('The retained native reference build differs')
        else:
            arguments=[str(executable),*flags,str(SOURCE)]
            arguments+=['/link','/OUT:'+str(library)] if os.name=='nt' else ['-o',str(library)]
            with (directory/'compiler-private.log').open('xb') as stream:
                result=subprocess.run(arguments,cwd=directory,env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
            if result.returncode or not library.is_file():raise RuntimeError('Native reference compilation failed; retain the private build log')
            stored=dict(expected,native_library_sha256=_sha(library))
            with manifest.open('x',encoding='utf-8') as stream:json.dump(stored,stream,indent=2)
        return cls(library,stored)

    def __init__(self,library,manifest):
        self.library=C.CDLL(str(Path(library).resolve()))
        ptr=C.POINTER(C.c_float)
        self.library.eg_ref_capabilities.argtypes=[];self.library.eg_ref_capabilities.restype=C.c_int
        self.library.eg_ref_mxcsr.argtypes=[];self.library.eg_ref_mxcsr.restype=C.c_uint32
        self.library.eg_ref_matmul.argtypes=[ptr,ptr,ptr,ptr,*([C.c_int64]*4),C.c_int];self.library.eg_ref_matmul.restype=C.c_int
        self.library.eg_ref_fma.argtypes=[ptr,ptr,ptr,ptr,C.c_int64,C.c_int];self.library.eg_ref_fma.restype=C.c_int
        if self.library.eg_ref_capabilities()!=1:raise RuntimeError('CPU FMA, AVX2 and OS vector state are required')
        if self.library.eg_ref_mxcsr()&0x6000:raise ValueError('Round-to-nearest is required for this explicit reference')
        self.configuration=dict(manifest,AVX2_FMA_and_OS_vector_state_verified=True,native_threads=4,
            executor='strict-native-block192-fma-reference-only',model_weights_changed=False,GPU_execution_validated=False,
            CPU_comparison_qualified=False,complete_performance_accepted=False)

    @staticmethod
    def _array(value):
        if not isinstance(value,np.ndarray) or value.dtype!=np.float32 or not value.size or not np.isfinite(value).all():
            raise ValueError('Finite nonempty CPU FP32 arrays are required')
        return np.ascontiguousarray(value)

    @staticmethod
    def _pointer(value):return value.ctypes.data_as(C.POINTER(C.c_float)) if value is not None else None

    def matmul(self,left,right,*,bias=None):
        a,b=self._array(left),self._array(right)
        if a.ndim not in (2,3) or b.ndim!=a.ndim or a.shape[-1]!=b.shape[-2] or (a.ndim==3 and a.shape[0]!=b.shape[0]):
            raise ValueError('Matching static matrix or batch dimensions are required')
        if any(v<=0 or v>2**31-1 for v in (*a.shape,*b.shape)):raise ValueError('Invalid reference matrix dimensions')
        if bias is not None:
            bias=self._array(bias)
            if bias.shape!=(b.shape[-1],):raise ValueError('Only a shared column bias is supported')
        output=np.empty((*a.shape[:-1],b.shape[-1]),np.float32)
        code=self.library.eg_ref_matmul(self._pointer(a),self._pointer(b),self._pointer(bias),self._pointer(output),
            a.shape[0] if a.ndim==3 else 1,a.shape[-2],a.shape[-1],b.shape[-1],4)
        if code or not np.isfinite(output).all():raise ValueError('Native reference matrix execution failed')
        return output

    def fma(self,a,b,c):
        values=[self._array(v) for v in (a,b,c)]
        try:values=[np.ascontiguousarray(v) for v in np.broadcast_arrays(*values)]
        except ValueError:raise ValueError('Reference FMA broadcast shapes differ') from None
        output=np.empty(values[0].shape,np.float32)
        code=self.library.eg_ref_fma(*(self._pointer(v) for v in (*values,output)),output.size,4)
        if code or not np.isfinite(output).all():raise ValueError('Native reference FMA execution failed')
        return output


_LINEAR_SCOPE=threading.Lock()


class TorchFp32Reference:
    """Explicit no-grad CPU diagnostic adapter, never an automatic fallback.

    Selecting linear_scope changes process-global functional.linear temporarily.
    Use only in an isolated, single-inference-owner reference worker. Nested or
    concurrent scopes fail rather than replacing another owner's function.
    """
    def __init__(self,native):
        if not isinstance(native,NativeFp32Reference):raise ValueError('An explicit native CPU reference is required')
        self.native=native
        self.calls=dict(linear=0,matmul=0,fma=0)

    @staticmethod
    def _tensor(value):
        import torch
        if (torch.is_grad_enabled() or not isinstance(value,torch.Tensor) or value.device.type!='cpu'
                or value.dtype!=torch.float32 or not value.numel()):
            raise ValueError('Explicit no-grad nonempty CPU FP32 tensors are required')
        return value.detach().contiguous().numpy()

    def linear(self,value,weight,bias=None):
        import torch
        a,b=self._tensor(value),self._tensor(weight)
        if a.ndim<1 or b.ndim!=2 or a.shape[-1]!=b.shape[1]:raise ValueError('Original linear dimensions differ')
        offset=self._tensor(bias) if bias is not None else None
        result=self.native.matmul(a.reshape(-1,a.shape[-1]),b.T,bias=offset)
        self.calls['linear']+=1
        return torch.from_numpy(result.reshape(*a.shape[:-1],b.shape[0]))

    def matmul(self,value,weight,*,bias=None):
        import torch
        a,b=self._tensor(value),self._tensor(weight)
        offset=self._tensor(bias) if bias is not None else None
        result=self.native.matmul(a,b,bias=offset)
        self.calls['matmul']+=1
        return torch.from_numpy(result)

    def fma(self,a,b,c):
        import torch
        result=self.native.fma(*(self._tensor(v) for v in (a,b,c)))
        self.calls['fma']+=1
        return torch.from_numpy(result)

    @contextmanager
    def linear_scope(self):
        import torch.nn.functional as functional
        if not _LINEAR_SCOPE.acquire(blocking=False):raise ValueError('A CPU reference linear scope is already active')
        original=functional.linear
        try:
            functional.linear=self.linear
            yield self
        finally:
            functional.linear=original
            _LINEAR_SCOPE.release()
