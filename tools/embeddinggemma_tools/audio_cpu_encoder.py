"""Audited five-part CPU reference adapter, outside the Unity GPU runtime."""
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import torch
from torch import nn

from .native_regression_distribution import audit_distribution_report
from .onnx_cpu_comparison import CpuOnnxSession, digest


CASES = frozenset(('silence', 'impulse', 'sine', 'nyquist', 'noise', 'padded_noise'))


def _qualify_retained_graph_producer(producer_source, execution_source, *, root=None):
    """Qualify reuse of unchanged exports; preserve their actual producer source.

    Compare every tracked generator module, reference and dependency input.
    Explicit CPU-only consumers and the independent allocation observer are excluded:
    none is imported by the raw graph producer. This is not an assertion that
    the consumer or the GPU SDK has been rerun at the newer execution source.
    """
    if any(not isinstance(value,str) or re.fullmatch('[a-f0-9]{40}',value) is None
           for value in (producer_source,execution_source)):
        raise ValueError('Full immutable producer and execution sources are required')
    root=Path(root) if root is not None else Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=root,stderr=subprocess.PIPE,timeout=30)
    try:
        if (git('rev-parse','HEAD').decode().strip()!=execution_source or git('status','--porcelain','--untracked-files=all')
                or git('cat-file','-t',producer_source).strip()!=b'commit'):
            raise ValueError('The executor must be clean and fixed at its actual source')
        paths=('tools/embeddinggemma_tools','tools/references','tools/pyproject.toml','tools/uv.lock')
        consumers={'tools/embeddinggemma_tools/'+name for name in
            ('audio_cpu_reference.py','audio_cpu_encoder.py','native_gpu_memory.py',
             'cpu_fp32_reference.py','native_audio_cpu_encoder.py')}
        def inventory(source):
            raw=git('ls-tree','-r','--format=%(objectmode) %(objectname) %(path)',source,'--',*paths)
            rows={}
            for line in raw.decode().splitlines():
                mode,blob,name=line.split(' ',2)
                if name not in consumers: rows[name]=(mode,blob)
            if not rows: raise ValueError('A nonempty graph producer input inventory is required')
            return rows
        producer,current=inventory(producer_source),inventory(execution_source)
        if producer!=current: raise ValueError('Retained graph producer inputs differ')
        digest=hashlib.sha256(json.dumps(producer,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return dict(producer_source=producer_source,execution_source=execution_source,
            producer_inputs_identical=True,producer_inputs_sha256=digest,
            graph_producer_reused=True,new_export_or_full_performance_accepted=False)
    except (OSError,subprocess.SubprocessError,UnicodeError) as error:
        raise ValueError('Retained graph producer source could not be qualified') from None


class AuditedAudioCpuEncoder(nn.Module):
    """Carry actual preprocessing output through five reviewed CPU sessions.

    Original reference data is consumed only by the admission auditor. No
    golden arrays are retained in the adapter or substituted in ``forward``.
    The owner must close the adapter, including after downstream failures.
    """

    def __init__(self, chain_report, original, *, source, retained_producer_source=None):
        super().__init__()
        self._sessions = []
        self._closed = True
        self._configuration = None
        if not isinstance(source, str) or re.fullmatch('[0-9a-f]{40}', source) is None:
            raise ValueError('An immutable execution source is required')
        try:
            path = Path(chain_report)
            if not path.is_file() or path.is_symlink():
                raise ValueError('An actual source report is required')
            report_hash = digest(path)
            audited = audit_distribution_report(path, original)
            case_id = audited['case_id']
            profile = 'frames-1119-valid-' + ('839' if case_id == 'padded_noise' else '1119')
            producer_source=source if retained_producer_source is None else retained_producer_source
            if (audited['original_tolerance_passed'] is not True or audited['source'] != producer_source
                    or case_id not in CASES or audited['profile'] != profile):
                raise ValueError('The original case, profile and execution source must match')
            qualification=None
            if retained_producer_source is not None:
                qualification=_qualify_retained_graph_producer(retained_producer_source,source)
            data = json.loads(path.read_bytes())
            selected = [row for row in data['cases'] if row['variant'] == 'ordered_original_midpoint_bmm']
            if len(selected) != 1 or len(selected[0]['parts']) != 5:
                raise ValueError('Exactly five ordered audited parts are required')
            shapes = [(1,1119,128)] + [(1,280,1024)]*4 + [(210 if case_id == 'padded_noise' else 280,512)]
            for index, part in enumerate(selected[0]['parts']):
                name = part['graph']
                if (type(part['index']) is not int or part['index'] != index or not isinstance(name, str) or name in ('','.','..')
                        or any(c in name for c in ('/','\\',':'))):
                    raise ValueError('Original graph order and relative basenames are required')
                graph = path.parent / name
                if not graph.is_file() or graph.is_symlink():
                    raise ValueError('An actual inline graph is required')
                session = CpuOnnxSession(graph, part['graph_sha256'],
                    graph_optimizations='disabled', node_order='priority')
                self._sessions.append(session)
                if session.input_shape != shapes[index] or session.output_shape != shapes[index+1]:
                    raise ValueError('The actual graph shape differs from its original profile')
            if digest(path) != report_hash:
                raise ValueError('The admitted source report changed during loading')
            self._configuration = dict(source=source, case_id=case_id, profile=profile,
                chain_report_sha256=report_hash,
                sessions=[copy.deepcopy(v.configuration) for v in self._sessions],
                original_reference_audit_passed=True, complete_graph_hashes_checked_at_load=True,
                execution_schedule='chain-major', maximum_owned_sessions=5,
                references_used_as_inference_inputs=False,
                CPU_reference_comparison_qualified=False, GPU_validated=False,
                complete_performance_accepted=False)
            if qualification is not None:
                self._configuration['retained_graph_producer_qualification']=qualification
            self._closed = False
            self.eval()
        except Exception:
            self.close()
            raise ValueError('Audited audio CPU encoder loading failed') from None

    @property
    def configuration(self):
        return copy.deepcopy(self._configuration)

    def forward(self, value):
        if (self._closed or not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                or value.dtype != torch.float32 or tuple(value.shape) != (1,1119,128)
                or not bool(torch.isfinite(value).all())):
            raise ValueError('An open CPU encoder and actual finite FP32 preprocessing tensor are required')
        actual = value.detach().contiguous().numpy()
        for session in self._sessions:
            actual = session.run(actual)
        return torch.from_numpy(np.array(actual, dtype=np.float32, copy=True, order='C'))

    def close(self):
        for session in self._sessions:
            session.close()
        self._sessions.clear()
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise ValueError('The CPU encoder is closed')
        return self

    def __exit__(self, *_):
        self.close()
