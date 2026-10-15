from pathlib import Path

import pytest
import torch
import yaml

from embeddinggemma_tools import frozen_audio_chained_onnx as chain
from embeddinggemma_tools import frozen_audio_prefix_onnx as prefix

PADDED = 'frames-1119-valid-839'


def test_chained_cli_keeps_selected_case_and_profile_for_graph_boundaries_and_both_endpoints(tmp_path, monkeypatch):
    calls = []; marker = object(); values = torch.zeros(1)
    monkeypatch.setenv('GITHUB_SHA', 'a' * 40)
    def load(path, *, case_id):
        calls.append(('graph', case_id)); return marker, values, values, {'profile': PADDED}
    def boundary(program, stage, *, profile):
        assert program is marker; calls.append(('boundary', profile, stage)); return stage
    def stage(path, stage, *, case_id):
        calls.append(('stage', case_id, stage)); return prefix.FrozenStageReference(values, {'case_id': case_id})
    def run(*args, **kwargs):
        calls.append(('run', args[1].provenance['case_id'], args[2].provenance['case_id']))
        return dict(complete=True, context_qualified=False, cases=[])
    monkeypatch.setattr(chain, 'load_frozen_input', load)
    monkeypatch.setattr(chain, 'boundary_for_stage', boundary)
    monkeypatch.setattr(chain, 'load_frozen_stage_reference', stage)
    monkeypatch.setattr(chain, 'probe_chained_segments', run)
    assert chain.main(['--input', str(tmp_path), '--output', str(tmp_path / 'out'),
        '--frozen-stage-source', str(tmp_path), '--variant', 'native', '--case', 'padded_noise']) == 0
    assert ('graph', 'padded_noise') in calls
    assert ('run', 'padded_noise', 'padded_noise') in calls
    assert len([x for x in calls if x[0] == 'boundary' and x[1] == PADDED]) == 5


def test_prefix_cli_keeps_selected_raw_case_and_original_subsample_reference(tmp_path, monkeypatch):
    calls = []; marker = object(); values = torch.zeros(1)
    monkeypatch.setenv('GITHUB_SHA', 'a' * 40)
    def load(path, *, case_id):
        calls.append(('graph', case_id)); return marker, values, values, {'profile': PADDED}
    def boundary(program, stage, *, profile):
        calls.append(('boundary', profile, stage)); return 'view_2'
    def stage(path, stage, *, case_id):
        calls.append(('stage', case_id)); return prefix.FrozenStageReference(values, {'case_id': case_id})
    def run(*args, **kwargs):
        calls.append(('run', kwargs['frozen_stage'].provenance['case_id'])); return {'complete': True}
    monkeypatch.setattr(prefix, 'load_frozen_input', load)
    monkeypatch.setattr(prefix, 'boundary_for_stage', boundary)
    monkeypatch.setattr(prefix, 'load_frozen_stage_reference', stage)
    monkeypatch.setattr(prefix, 'probe_prefix', run)
    assert prefix.main(['--input', str(tmp_path), '--output', str(tmp_path / 'out'),
        '--frozen-stage-source', str(tmp_path), '--variant', 'native', '--case', 'padded_noise']) == 0
    assert ('graph', 'padded_noise') in calls and ('stage', 'padded_noise') in calls
    assert ('boundary', PADDED, 'subsample') in calls and ('run', 'padded_noise') in calls


def test_actions_matrix_has_all_six_original_cases_and_distinct_numeric_artifacts_without_changing_default():
    path = Path(__file__).resolve().parents[2] / '.github/workflows/frozen-audio-prefix-onnx.yml'
    text = path.read_text(encoding='utf-8'); workflow = yaml.safe_load(text)
    inputs = workflow.get('on', workflow.get(True))['workflow_dispatch']['inputs']
    assert inputs['case_id']['default'] == 'silence'
    assert inputs['case_id']['options'] == ['silence', 'all', 'impulse', 'sine', 'nyquist', 'noise', 'padded_noise']
    job = workflow['jobs']['probe']; matrix = job['strategy']['matrix']['case_id']
    assert 'inputs.case_id' in matrix and 'padded_noise' in matrix and 'noise' in matrix
    assert job['env']['PROBE_CASE'] == '${{ matrix.case_id }}'
    uploads = [x['with']['name'] for x in job['steps'] if x.get('with', {}).get('name', '').startswith('frozen-audio-prefix-onnx-')]
    assert len(uploads) == 2 and all('matrix.case_id' in name for name in uploads)
    assert '--case "$PROBE_CASE"' in text
    assert '${PROBE_PROFILE}/${PROBE_CASE}-features-fp32.bin' in text
    assert '${PROBE_PROFILE}/${PROBE_CASE}-stage-${stage_index}-fp32.bin' in text
