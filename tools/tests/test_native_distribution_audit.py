"""Distribution reports need their actual executed graph hash and the full original matrix."""
import json

import pytest

from test_audit_audio_raw_chain import evidence
from test_audio_generation_candidate import numeric_candidate
from embeddinggemma_tools import native_regression_distribution as distribution


@pytest.fixture
def report(numeric_candidate,monkeypatch):
    candidate,_,original,path,value=numeric_candidate
    inventory=dict(candidate_contract_sha256=candidate.CONTRACT_SHA,trace_removal_producer='c'*40,
                   files={'candidate-1119.pt2':dict(kind='pt2',sha256='6'*64,source_graph_sha256='e'*64)})
    config=dict(inventory_sha256='b'*64)
    monkeypatch.setattr(distribution,'configuration',lambda:(config,inventory));monkeypatch.setattr(distribution,'CONFIG_SHA','9'*64)
    for key in ('provenance','input_provenance','target_provenance'):
        value[key].update(graph_sha256='6'*64,executed_graph_sha256='6'*64,source_graph_sha256='e'*64,
                          graph_transport=distribution.TRANSPORT,graph_transport_contract_sha256='9'*64,
                          graph_distribution_inventory_sha256='b'*64,graph_distribution_producer='c'*40)
    path.write_text(json.dumps(value))
    return candidate,original,path,value


def test_explicit_audit_checks_original_captures_and_actual_clean_graph_identity(report):
    _,original,path,_=report
    result=distribution.audit_distribution_report(path,original)
    assert result['original_tolerance_passed'] and result['native_chain_control_independently_verified']
    assert result['executed_graph_sha256']=='6'*64 and result['source_graph_sha256']=='e'*64
    assert result['graph_transport_contract_sha256']=='9'*64
    assert result['gpu_validated'] is False and result['graphs_independently_rehashed'] is False


def test_existing_raw_gate_rejects_distribution_reports(report):
    candidate,original,path,_=report
    with pytest.raises(ValueError):candidate.audit_candidate_report(path,original)


@pytest.mark.parametrize('field',['graph_sha256','executed_graph_sha256','source_graph_sha256','graph_transport',
                                  'graph_transport_contract_sha256','graph_distribution_inventory_sha256','graph_distribution_producer'])
def test_every_input_target_and_job_provenance_must_match_the_pinned_distribution(report,field):
    _,original,path,value=report;value['target_provenance'][field]='changed';path.write_text(json.dumps(value))
    with pytest.raises(ValueError):distribution.audit_distribution_report(path,original)


def test_new_route_keeps_native_control_and_original_tolerance_gates(report):
    _,original,path,value=report;value['native_control_context']['threads']=1;path.write_text(json.dumps(value))
    with pytest.raises(ValueError):distribution.audit_distribution_report(path,original)


@pytest.mark.parametrize('problem',[None,'missing','duplicate','source','values','generation','transport'])
def test_distribution_aggregate_requires_every_original_case_and_one_transport(monkeypatch,problem):
    from embeddinggemma_tools import audio_generation_candidate as candidate
    names=['silence','impulse','sine','nyquist','noise','padded_noise']
    rows=[dict(case_id=name,profile='frames-1119-valid-839' if name=='padded_noise' else 'frames-1119-valid-1119',
        source='a'*40,original_tolerance_passed=True,values_compared=(210 if name=='padded_noise' else 280)*512,
        candidate_metrics=dict(mismatched_elements=0,exact_difference_count=3),native_context_qualified=False,
        model_generation='audio-test',candidate_contract_sha256='f'*64,graph_transport=distribution.TRANSPORT,
        graph_transport_contract_sha256='9'*64,graph_distribution_inventory_sha256='b'*64) for name in names]
    if problem=='missing':rows.pop()
    elif problem=='duplicate':rows[-1]=rows[0]
    elif problem=='source':rows[-1]['source']='d'*40
    elif problem=='values':rows[-1]['values_compared']+=1
    elif problem=='generation':rows[-1]['model_generation']='audio-other'
    elif problem=='transport':rows[-1]['graph_transport_contract_sha256']='0'*64
    monkeypatch.setattr(candidate,'candidate_contract',lambda:{'generation':'audio-test'})
    monkeypatch.setattr(candidate,'CONTRACT_SHA','f'*64);monkeypatch.setattr(distribution,'CONFIG_SHA','9'*64)
    monkeypatch.setattr(distribution,'configuration',lambda:(dict(inventory_sha256='b'*64),{}))
    monkeypatch.setattr(distribution,'audit_distribution_report',lambda index,original:rows[index])
    if problem:
        with pytest.raises(ValueError):distribution.audit_distribution_reports(list(range(len(rows))),'original',source='a'*40)
    else:
        result=distribution.audit_distribution_reports(list(range(6)),'original',source='a'*40)
        assert result['success'] and result['values_compared']==824320 and result['case_count']==6
        assert result['graph_transport_contract_sha256']=='9'*64 and result['gpu_validated'] is False
