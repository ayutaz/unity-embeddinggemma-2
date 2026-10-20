import json
import pytest
from test_image_cpu_reference import pipeline,cohort


@pytest.fixture
def selected(pipeline,monkeypatch):
    driver,snapshot,fixtures,refs,state=pipeline;original=driver._load_pipeline
    configuration=dict(policy='sixteen-lane-vision',selected_norms=1,replacements=[dict(name='vision.fixture',width=64,scaled=True,scale_borrowed=True)])
    def load(*args,statistics_reference=None):
        assert statistics_reference=='sixteen-lane'
        processor,encoder,backbone=original(*args)
        encoder.statistics_reference_configuration=configuration
        return processor,encoder,backbone
    monkeypatch.setattr(driver,'_load_pipeline',load)
    return driver,snapshot,fixtures,refs,state


def run(selected,output,mode='sixteen-lane'):
    driver,snapshot,fixtures,refs,_=selected
    return driver.measure_image_cpu_reference(snapshot,refs,'square',output,source='a'*40,fixture_root=fixtures,statistics_reference=mode)


def test_explicit_statistics_mode_keeps_original_four_carries_and_measures_release(selected,tmp_path):
    driver,_,_,refs,state=selected;output=tmp_path/'measurement';report=run(selected,output)
    assert report['success'] and report['statistics_reference']=='sixteen-lane' and len(report['rounds'])==4
    assert len(state['calls'])==12 and report['release_completed'] is True and report['release_wall_seconds']>=0
    assert driver.audit_image_cpu_reference(output,refs,source='a'*40)['all_four_chains_match_fixed_references']


@pytest.mark.parametrize('field',['statistics_reference','statistics_reference_configuration','release_completed'])
def test_saved_audit_rejects_changed_mode_or_missing_lifecycle(selected,tmp_path,field):
    driver,_,_,refs,_=selected;output=tmp_path/'measurement';run(selected,output)
    path=output/'image-cpu-reference.json';report=json.loads(path.read_bytes())
    report[field]='native' if field=='statistics_reference' else {} if field=='statistics_reference_configuration' else False
    path.write_text(json.dumps(report),encoding='utf-8')
    with pytest.raises(ValueError):driver.audit_image_cpu_reference(output,refs,source='a'*40)


def test_unknown_statistics_mode_is_rejected_before_model_load(selected,tmp_path):
    *_,state=selected
    with pytest.raises(ValueError):run(selected,tmp_path/'measurement',mode='automatic')
    assert state['loads']==0


def test_cli_forwards_explicit_reference_and_audit_cannot_override_it(selected,tmp_path,monkeypatch):
    driver,snapshot,fixtures,refs,_=selected;output=tmp_path/'measurement'
    monkeypatch.setattr(driver.torch,'set_num_interop_threads',lambda _:None)
    def measure(*args,**kwargs):
        assert kwargs['statistics_reference']=='sixteen-lane'
        return {}
    monkeypatch.setattr(driver,'measure_image_cpu_reference',measure)
    args=['measure','--references',str(refs),'--output',str(output),'--snapshot',str(snapshot),'--profile','square','--source','a'*40,'--statistics-reference','sixteen-lane']
    assert driver.main(args)==0
    monkeypatch.setattr(driver,'audit_image_cpu_reference',lambda *args,**kwargs:pytest.fail('audit accepted an arithmetic override'))
    assert driver.main(['audit','--references',str(refs),'--output',str(output),'--source','a'*40,'--statistics-reference','sixteen-lane'])==1
