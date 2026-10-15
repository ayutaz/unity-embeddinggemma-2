import copy

import pytest
import torch

from embeddinggemma_tools import audio_layer_replay


class Prefix(torch.nn.Module):
    def __init__(self,fail=False):
        super().__init__();self.conv=torch.nn.Conv2d(1,2,1,bias=False);self.fail=fail
    def forward(self,features,mask):
        if self.fail:raise RuntimeError('probe failure')
        return features[:,:2,:2].clone(),mask[:,:2]


@pytest.mark.parametrize('fail',[False,True])
def test_subsample_context_probe_retains_failure_and_restores_settings_inputs_and_weights(fail):
    probe=getattr(audio_layer_replay,'diagnose_subsample_context',None)
    assert callable(probe),'Bounded subsample CPU-context probe is missing'
    module=Prefix(fail);features=torch.arange(24,dtype=torch.float32).reshape(1,6,4)
    mask=torch.ones((1,6),dtype=torch.bool);expected=features[:,:2,:2].clone()
    original=features.clone();weights=copy.deepcopy(module.state_dict());threads=torch.get_num_threads();enabled=torch.backends.mkldnn.enabled
    report=probe(module,features,mask,expected)
    assert torch.get_num_threads()==threads and torch.backends.mkldnn.enabled==enabled
    assert torch.equal(features,original) and all(torch.equal(module.state_dict()[k],v) for k,v in weights.items())
    assert report['scope']=='subsample_cpu_context_diagnosis_only'
    assert not report['full_encoder_validated'] and not report['unity_validated']
    conditions=report['conditions'];assert len(conditions)==24
    assert {r['requested_threads'] for r in conditions}=={1,2,4}
    assert {r['mkldnn_enabled'] for r in conditions}=={True,False}
    assert {r['weight_layout'] for r in conditions}=={'contiguous','channels_last'}
    assert {r['feature_layout'] for r in conditions}=={'contiguous','time_contiguous'}
    for row in conditions:
        assert row['executed'] is (not fail)
        if fail:assert 'probe failure' in row['error']
        else:assert row['bit_differences']==0 and row['mismatched_elements']==0


def test_cli_retains_subsample_context_failure_as_failed_and_unvalidated(tmp_path,monkeypatch):
    def failure(*args,**kwargs):
        error=ValueError('native replay does not match pinned subsample final')
        error.context_diagnostics={'scope':'subsample_cpu_context_diagnosis_only','conditions':[{'executed':True,'bit_differences':0}]}
        raise error
    monkeypatch.setattr(audio_layer_replay,'prepare',failure)
    code=audio_layer_replay.main(['--snapshot',str(tmp_path),'--context',str(tmp_path),'--output',str(tmp_path/'out'),'--reference-source','a'*40,'--diagnostic-source','c'*40,'--report',str(tmp_path/'report.json')])
    import json
    report=json.loads((tmp_path/'report.json').read_text())
    assert code==1 and not report['success'] and not report['context_validated']
    assert report.get('subsample_context_diagnostics',{}).get('conditions'),'Failure-only diagnostic evidence is missing'
