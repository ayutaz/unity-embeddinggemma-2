"""Explicit CPU numerical-reference executor, separate from the Unity GPU SDK."""
from contextlib import contextmanager
import copy
import importlib
import re
import threading

import torch

from .cpu_fp32_reference import NativeFp32Reference,TorchFp32Reference
from .frozen_audio_prefix_onnx import PrefixExport,extract_prefix,weight_digests
from .frozen_audio_segment_onnx import SegmentExport,extract_segment,PRECISE_MIDPOINT
from .native_regression_distribution import load_distribution_input

_SCOPE=threading.Lock()
FMA_MODULES=('export','native_silu_onnx','native_softmax_onnx','native_audio_primitives_onnx','native_tanh_onnx')


@contextmanager
def native_audio_fma_scope(function):
    """Use only in an isolated reference process; restore every shared function."""
    if not _SCOPE.acquire(blocking=False):raise ValueError('A native audio reference scope is already active')
    replaced=[]
    try:
        for name in FMA_MODULES:
            module=importlib.import_module('embeddinggemma_tools.'+name)
            replaced.append((module,module.fp32_fma));module.fp32_fma=function
        yield
    finally:
        for module,original in reversed(replaced):module.fp32_fma=original
        _SCOPE.release()


def _make_wrappers(program,operations,case_id):
    boundaries=('view_2','mul_77','mul_152','mul_227','slice_51' if case_id=='padded_noise' else 'select')
    parts=[PrefixExport(extract_prefix(program,boundaries[0]),'ordered_original_stem')]
    parts.extend(SegmentExport(extract_segment(program,a,b),PRECISE_MIDPOINT) for a,b in zip(boundaries,boundaries[1:]))
    for index,part in enumerate(parts):
        part.linear_candidate=operations.matmul
        part.biased_linear_candidate=lambda bias,a,b:operations.matmul(a,b,bias=bias)
        if index==0:part.convolution_candidate=operations.stem_convolution
        else:part.bmm_candidate=operations.matmul
    return parts


class NativeAudioCpuEncoder(torch.nn.Module):
    """Load hash-qualified original coefficients, then carry actual CPU tensors.

    Reference arrays returned by the admission loader are discarded. The native
    helper must be built explicitly before model-load timing starts. Original
    coefficient verification is explicit so it can remain outside the timers.
    """
    def __init__(self,candidate,original,*,source,case_id,native):
        super().__init__()
        self.parts=torch.nn.ModuleList();self.program=None;self._closed=True;self._configuration=None
        if not isinstance(source,str) or re.fullmatch('[0-9a-f]{40}',source) is None:
            raise ValueError('A full immutable executor source is required')
        if case_id not in ('silence','padded_noise') or not isinstance(native,NativeFp32Reference):
            raise ValueError('An explicit native reference and fixed audio case are required')
        try:
            program,_,_,provenance=load_distribution_input(candidate,original,case_id=case_id)
            profile='frames-1119-valid-'+('839' if case_id=='padded_noise' else '1119')
            if provenance['case_id']!=case_id or provenance['profile']!=profile:
                raise ValueError('Original coefficient profile differs')
            self.operations=TorchFp32Reference(native)
            self._original_weights=weight_digests(program)
            self.parts=torch.nn.ModuleList(_make_wrappers(program,self.operations,case_id))
            if len(self.parts)!=5:raise ValueError('Exactly five original partitions are required')
            # ExportedProgram.module() rejects train/eval calls. Keep this
            # original solely as the checksum owner; executable partitions
            # already register all active coefficients as ordinary modules.
            object.__setattr__(self,'program',program)
            self._shapes=[(1,280,1024)]*4+[(210 if case_id=='padded_noise' else 280,512)]
            self._configuration=dict(executor='native-fp32-reference',source=source,case_id=case_id,profile=profile,
                coefficient_graph_provenance=copy.deepcopy(provenance),native_reference=copy.deepcopy(native.configuration),
                maximum_owned_parts=5,execution_schedule='chain-major',references_used_as_inference_inputs=False,
                original_ONNX_bytes_executed=False,hardware_FMA_may_differ_from_bounded_Dekker_graph=True,
                original_coefficients_verified_unchanged=False,CPU_reference_comparison_qualified=False,
                GPU_validated=False,complete_performance_accepted=False)
            self._closed=False;self.eval()
        except Exception:
            self.close()
            raise ValueError('Native audio CPU reference loading failed') from None

    @property
    def configuration(self):
        result=copy.deepcopy(self._configuration)
        if result is not None:
            result.update(closed=self._closed,native_actual_calls=copy.deepcopy(self.operations.calls))
        return result

    def forward(self,value):
        if (self._closed or not isinstance(value,torch.Tensor) or value.device.type!='cpu'
                or value.dtype!=torch.float32 or tuple(value.shape)!=(1,1119,128) or not torch.isfinite(value).all()):
            raise ValueError('An open reference and actual finite CPU FP32 audio features are required')
        try:
            with torch.no_grad(),native_audio_fma_scope(self.operations.fma):
                actual=value
                for part,shape in zip(self.parts,self._shapes,strict=True):
                    actual=part(actual)
                    if isinstance(actual,tuple) and len(actual)==1:actual=actual[0]
                    if (not isinstance(actual,torch.Tensor) or actual.dtype!=torch.float32 or actual.device.type!='cpu'
                            or tuple(actual.shape)!=shape or not torch.isfinite(actual).all()):
                        raise ValueError('An original CPU partition returned a different contract')
                return actual
        except Exception:
            raise ValueError('Native audio CPU reference execution failed') from None

    def verify_coefficients(self):
        if self._closed:raise ValueError('The original coefficient owner is closed')
        same=weight_digests(self.program)==self._original_weights
        self._configuration['original_coefficients_verified_unchanged']=same
        if not same:raise ValueError('An original coefficient changed during reference execution')
        return True

    def close(self):
        self.parts=torch.nn.ModuleList();self.program=None;self._closed=True
