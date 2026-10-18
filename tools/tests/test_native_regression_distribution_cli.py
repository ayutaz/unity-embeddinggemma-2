"""Manual numerical transport must remain independent of generation adoption."""
import importlib
import json
import sys

import pytest


def entry():
    module=importlib.import_module('embeddinggemma_tools.native_regression_distribution')
    assert callable(getattr(module,'main',None)), 'Standalone unqualified distribution CLI is missing'
    return module


def test_manual_fetch_uses_explicit_distribution_without_loading_selection(monkeypatch,tmp_path,capsys):
    m=entry();calls=[]
    monkeypatch.setitem(sys.modules,'embeddinggemma_tools.selected_audio_generation',None)
    monkeypatch.setattr(m,'fetch_candidate',lambda output,original,*,case_id:calls.append((output,original,case_id)))
    assert m.main(['fetch','--output',str(tmp_path/'candidate'),'--original',str(tmp_path/'original'),'--case','padded_noise'])==0
    assert calls==[(tmp_path/'candidate',tmp_path/'original','padded_noise')]
    result=json.loads(capsys.readouterr().out)
    assert result['success'] and not result['generation_qualified'] and not result['gpu_validated']


def test_original_fetch_uses_public_distribution_without_candidate_or_selection(monkeypatch,tmp_path,capsys):
    m=entry();calls=[]
    monkeypatch.setitem(sys.modules,'embeddinggemma_tools.selected_audio_generation',None)
    monkeypatch.setattr(m,'fetch_original',lambda output:calls.append(output))
    assert m.main(['fetch-original','--output',str(tmp_path/'original')])==0
    assert calls==[tmp_path/'original'] and json.loads(capsys.readouterr().out)['success']


@pytest.mark.parametrize('success',[True,False])
def test_audit_routes_all_actual_reports_and_preserves_failed_result(monkeypatch,tmp_path,capsys,success):
    m=entry();reports=tmp_path/'reports';reports.mkdir();paths=[]
    for case in ['silence','impulse','sine','nyquist','noise','padded_noise']:
        p=reports/case/'frozen-audio-chained-onnx.json';p.parent.mkdir();p.write_text('{}');paths.append(p)
    result=dict(success=success,values_compared=824320,candidate_tolerance_failures=0 if success else 1,generation_qualified=False,gpu_validated=False)
    calls=[]
    def audit(actual,original,*,source):calls.append((actual,original,source));return result
    monkeypatch.setattr(m,'audit_distribution_reports',audit)
    out=tmp_path/'audit.json';source='a'*40
    assert m.main(['audit','--reports',str(reports),'--original',str(tmp_path/'original'),'--source',source,'--output',str(out)])==(0 if success else 1)
    assert calls==[(sorted(paths),tmp_path/'original',source)]
    assert json.loads(out.read_text())==result
    assert json.loads(capsys.readouterr().out)['success'] is success


def test_audit_never_overwrites_existing_evidence(monkeypatch,tmp_path):
    m=entry();out=tmp_path/'audit.json';out.write_bytes(b'original evidence')
    monkeypatch.setattr(m,'audit_distribution_reports',lambda *a,**k:pytest.fail('Existing evidence must be checked first'))
    with pytest.raises(ValueError,match='overwrite'):
        m.main(['audit','--reports',str(tmp_path),'--original',str(tmp_path),'--source','a'*40,'--output',str(out)])
    assert out.read_bytes()==b'original evidence'


@pytest.mark.parametrize('field',['generation_qualified','gpu_validated'])
def test_manual_numerical_transport_never_promotes_model_or_gpu_acceptance(monkeypatch,tmp_path,field):
    m=entry();out=tmp_path/'audit.json';result=dict(success=True,generation_qualified=False,gpu_validated=False);result[field]=True
    monkeypatch.setattr(m,'audit_distribution_reports',lambda *a,**k:result)
    with pytest.raises(ValueError,match='qualification'):
        m.main(['audit','--reports',str(tmp_path),'--original',str(tmp_path),'--source','a'*40,'--output',str(out)])
    assert not out.exists()
