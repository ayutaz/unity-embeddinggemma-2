import copy

import pytest

from embeddinggemma_tools import audio_raw_resume as probe
from embeddinggemma_tools import audit_audio_raw_chain as original_audit


@pytest.fixture
def audited_rows(monkeypatch):
    names=['silence','impulse','sine','nyquist','padded_noise']
    rows={name:dict(case_id=name,profile='frames-1119-valid-839' if name=='padded_noise' else 'frames-1119-valid-1119',
        source=probe.PREDECESSOR_SOURCE,values_compared=107520 if name=='padded_noise' else 143360,
        original_tolerance_passed=True,native_context_qualified=False,
        candidate_metrics=dict(mismatched_elements=0,exact_difference_count=3)) for name in names}
    resumed=dict(case_id='noise',profile='frames-1119-valid-1119',predecessor_source=probe.PREDECESSOR_SOURCE,
        job_source='b'*40,execution_sources=sorted([probe.PREDECESSOR_SOURCE,'b'*40]),values_compared=143360,
        original_tolerance_passed=True,predecessor_native_context_qualified=False,
        candidate_metrics=dict(mismatched_elements=0,exact_difference_count=4))
    calls=[]
    def audit(path,root,**kwargs):
        calls.append((path,kwargs));return copy.deepcopy(rows[path])
    monkeypatch.setattr(original_audit,'audit_report',audit)
    monkeypatch.setattr(probe,'audit_resume',lambda *a:copy.deepcopy(resumed))
    monkeypatch.setattr(probe,'verify_math_source',lambda:'fixed-math-manifest')
    return names,rows,resumed,calls


def test_all_six_requires_full_original_values_and_reports_two_execution_sources(audited_rows):
    names,_,_,calls=audited_rows
    r=probe.audit_six_with_resume(names,'resume','previous','original')
    assert r['success'] and r['values_compared']==824320 and r['case_count']==6 and r['profile_count']==2
    assert r['candidate_tolerance_failures']==0 and r['candidate_bit_differences']==19
    assert r['execution_sources']==sorted([probe.PREDECESSOR_SOURCE,'b'*40])
    assert not r['all_six_at_one_job_source'] and not r['gpu_validated'] and not r['default_export_promoted']
    assert all(options=={'expected_variant':'ordered_original_midpoint_bmm'} for _,options in calls)


@pytest.mark.parametrize('change',['duplicate','missing','source','profile','redistributed_values','noise_source'])
def test_six_case_coverage_cannot_hide_missing_profiles_sources_or_values(audited_rows,change):
    names,rows,resumed,_=audited_rows
    if change=='duplicate':names[-1]='silence'
    if change=='missing':names.pop()
    if change=='source':rows['sine']['source']='c'*40
    if change=='profile':rows['padded_noise']['profile']='frames-1119-valid-1119'
    if change=='redistributed_values':
        rows['sine']['values_compared']-=1;rows['padded_noise']['values_compared']+=1
    if change=='noise_source':resumed['predecessor_source']='c'*40
    with pytest.raises(ValueError):probe.audit_six_with_resume(names,'resume','previous','original')


def test_honest_last_case_numeric_failure_remains_failure(audited_rows):
    names,_,resumed,_=audited_rows
    resumed['original_tolerance_passed']=False;resumed['candidate_metrics']['mismatched_elements']=2
    r=probe.audit_six_with_resume(names,'resume','previous','original')
    assert not r['success'] and r['candidate_tolerance_failures']==2
