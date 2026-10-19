"""Audited five-part CPU reference adapter, outside the Unity GPU runtime."""
import copy
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn

from .native_regression_distribution import audit_distribution_report
from .onnx_cpu_comparison import CpuOnnxSession, digest


CASES = frozenset(('silence', 'impulse', 'sine', 'nyquist', 'noise', 'padded_noise'))


class AuditedAudioCpuEncoder(nn.Module):
    """Carry actual preprocessing output through five reviewed CPU sessions.

    Original reference data is consumed only by the admission auditor. No
    golden arrays are retained in the adapter or substituted in ``forward``.
    The owner must close the adapter, including after downstream failures.
    """

    def __init__(self, chain_report, original, *, source):
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
            if (audited['original_tolerance_passed'] is not True or audited['source'] != source
                    or case_id not in CASES or audited['profile'] != profile):
                raise ValueError('The original case, profile and execution source must match')
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
