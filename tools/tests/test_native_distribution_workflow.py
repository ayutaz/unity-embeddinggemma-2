"""Required CI must explicitly execute and audit the new transport under unchanged gates."""
import copy
import json

import pytest

from test_selected_audio_generation import documents
from embeddinggemma_tools import native_regression_distribution as distribution
from embeddinggemma_tools import selected_audio_generation as selected


@pytest.mark.parametrize('command',['fetch-distribution','fetch-original-distribution'])
def test_selection_is_verified_before_any_distribution_download(monkeypatch,tmp_path,command):
    manifest,_=documents();calls=[]
    monkeypatch.setattr(selected,'checked_selection',lambda root=None:calls.append('selection') or manifest)
    monkeypatch.setattr(distribution,'fetch_candidate',lambda *a,**kw:calls.append(('candidate',kw['case_id'])))
    monkeypatch.setattr(distribution,'fetch_original',lambda *a,**kw:calls.append('original'))
    args=[command,'--output',str(tmp_path/'out')]
    if command=='fetch-distribution':args+=['--original',str(tmp_path/'original'),'--case','noise']
    assert selected.main(args)==0
    assert calls==['selection',('candidate','noise') if command=='fetch-distribution' else 'original']
    calls.clear()
    def reject(root=None):raise ValueError('Changed selection')
    monkeypatch.setattr(selected,'checked_selection',reject)
    with pytest.raises(ValueError):selected.main(args)
    assert not calls


def test_explicit_distribution_audit_cli_writes_actual_result_and_never_overwrites(monkeypatch,tmp_path):
    manifest,_=documents();monkeypatch.setattr(selected,'checked_selection',lambda root=None:manifest)
    result=dict(success=True,required_numeric_gate_passed=True,gpu_validated=False)
    monkeypatch.setattr(selected,'validate_current_distribution_reports',lambda *a,**kw:copy.deepcopy(result),raising=False)
    output=tmp_path/'audit.json'
    args=['audit-distribution','--reports',str(tmp_path),'--original',str(tmp_path/'original'),'--source','a'*40,'--output',str(output)]
    assert selected.main(args)==0 and json.loads(output.read_text())==result
    with pytest.raises(ValueError):selected.main(args)


@pytest.mark.parametrize('problem',[None,'source','values','cases','profiles','tolerances','success','GPU','transport'])
def test_new_selection_gate_requires_the_whole_matrix_and_exact_transport(monkeypatch,problem):
    manifest,_=documents();monkeypatch.setattr(selected,'checked_selection',lambda root=None:manifest)
    config,_=distribution.configuration()
    value=dict(success=True,source='a'*40,model_generation=manifest['model_generation'],
        candidate_contract_sha256=manifest['candidate_contract_sha256'],case_count=6,profile_count=2,
        values_compared=824320,candidate_tolerance_failures=0,gpu_validated=False,generation_qualified=False,
        graph_transport=distribution.TRANSPORT,graph_transport_contract_sha256=distribution.CONFIG_SHA,
        graph_distribution_inventory_sha256=config['inventory_sha256'])
    if problem:
        field,bad={'source':('source','b'*40),'values':('values_compared',824319),'cases':('case_count',5),
            'profiles':('profile_count',1),'tolerances':('candidate_tolerance_failures',1),'success':('success',False),
            'GPU':('gpu_validated',True),'transport':('graph_transport_contract_sha256','0'*64)}[problem]
        value[field]=bad
    monkeypatch.setattr(distribution,'audit_distribution_reports',lambda *a,**kw:copy.deepcopy(value))
    if problem:
        with pytest.raises(ValueError):selected.validate_current_distribution_reports([],'original',source='a'*40)
    else:
        result=selected.validate_current_distribution_reports([],'original',source='a'*40)
        assert result['required_numeric_gate_passed'] and result['graph_transport_contract_sha256']==distribution.CONFIG_SHA


def test_distribution_raw_cli_never_defaults_to_the_old_graph_loader(monkeypatch,tmp_path):
    from embeddinggemma_tools import frozen_audio_chained_onnx as chain
    from embeddinggemma_tools import audio_generation_candidate as candidate
    monkeypatch.setenv('GITHUB_SHA','a'*40);calls=[]
    class ReachedLoader(Exception):pass
    def load(directory,original,*,case_id):calls.append(case_id);raise ReachedLoader()
    monkeypatch.setattr(distribution,'load_distribution_input',load)
    monkeypatch.setattr(candidate,'load_candidate_input',lambda *a,**kw:pytest.fail('Old loader selected'))
    with pytest.raises(ReachedLoader):
        chain.main(['--input',str(tmp_path),'--output',str(tmp_path/'out'),'--candidate-distribution-directory',str(tmp_path),
                    '--raw-input','--case','noise','--variant','native'])
    assert calls==['noise']


def test_distribution_cli_requires_raw_mode(monkeypatch,tmp_path):
    from embeddinggemma_tools import frozen_audio_chained_onnx as chain
    monkeypatch.setenv('GITHUB_SHA','a'*40)
    with pytest.raises(ValueError,match='raw'):
        chain.main(['--input',str(tmp_path),'--output',str(tmp_path/'out'),'--candidate-distribution-directory',str(tmp_path),
                    '--frozen-stage-source',str(tmp_path)])


def test_old_and_distribution_graph_modes_are_mutually_exclusive(monkeypatch,tmp_path,capsys):
    from embeddinggemma_tools import frozen_audio_chained_onnx as chain
    monkeypatch.setenv('GITHUB_SHA','a'*40)
    with pytest.raises(SystemExit):
        chain.main(['--input',str(tmp_path),'--output',str(tmp_path/'out'),'--raw-input',
                    '--candidate-directory',str(tmp_path),'--candidate-distribution-directory',str(tmp_path)])
    assert 'not allowed with argument' in capsys.readouterr().err
