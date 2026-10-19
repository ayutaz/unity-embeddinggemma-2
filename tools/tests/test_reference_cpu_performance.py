import hashlib,importlib,json
from dataclasses import replace

import pytest
import torch

FP32='40aefa5de06f575d31b7582c24314c1ab035da374116b3ce3036de0e919cd647'
IMAGE='90692ad38c6d7d844dc6b6e79c7d5cfa719ea59866946dbf94a40bca10ac9eac'


def implementation():
    try:module=importlib.import_module('embeddinggemma_tools.reference_cpu_performance')
    except ModuleNotFoundError:module=None
    assert module is not None,'The matched CPU reference measurement protocol is missing'
    return module


def fixture():
    module=implementation();raw=torch.tensor([1.]);encoder=torch.nn.Linear(1,1,bias=False).eval()
    with torch.no_grad():encoder.weight.fill_(2)
    expected=torch.zeros(768);expected[0]=1;expected[1]=2;expected/=expected.norm();calls=[]
    def preprocess():calls.append('preprocessing');return raw.clone()
    def encode(value):calls.append('encoder');assert torch.equal(value,torch.tensor([1.]));return encoder(value)
    def backbone(value):
        calls.append('backbone');assert torch.equal(value,torch.tensor([2.]))
        result=torch.zeros(768);result[0]=1;result[1]=value[0];return result/result.norm()
    case=module.CpuReferenceCase(module='image-square',precision='fp32',descriptor_sha256=FP32,reference_sha256=IMAGE,input_sha256='b'*64,reference_case_id='square-0',source='a'*40,expected=expected,input_signature=lambda:hashlib.sha256(raw.numpy().tobytes()).hexdigest(),preprocess=preprocess,encoder=encode,backbone=backbone,active_modules=(encoder,))
    return module,case,raw,calls


def test_every_round_executes_all_actual_stages_and_keeps_identical_inputs(tmp_path):
    module,case,raw,calls=fixture();result=module.measure_reference_pipeline(case,tmp_path/'result.json')
    assert calls==['preprocessing','encoder','backbone']*4 and torch.equal(raw,torch.tensor([1.]))
    assert result['success'] and result['actual_backend']=='CPU' and result['framework']=='PyTorch reference'
    assert [row['round'] for row in result['rounds']]==['first','warm','repeat-1','repeat-2']
    assert all([stage['stage'] for stage in row['stages']]==['preprocessing','encoder','backbone'] for row in result['rounds'])
    assert all(row['cosine']>=.999 and abs(row['norm_squared']-1)<=.002 and row['wall_seconds']>0 and row['process_cpu_seconds']>=0 for row in result['rounds'])
    assert all(stage['gpu_elapsed_ns'] is None for row in result['rounds'] for stage in row['stages'])
    assert result['GPU_execution_claimed'] is False and result['complete_performance_accepted'] is False
    assert json.loads((tmp_path/'result.json').read_text())==result
    for row in result['rounds']:
        capture=tmp_path/row['capture'];assert capture.stat().st_size==768*4
        assert hashlib.sha256(capture.read_bytes()).hexdigest()==row['capture_sha256']


@pytest.mark.parametrize('precision',[None,'','float16-weights','float16','auto'])
def test_reference_fp32_does_not_claim_unmatched_half_weight_representation(tmp_path,precision):
    module,case,_,calls=fixture()
    with pytest.raises(ValueError):module.measure_reference_pipeline(replace(case,precision=precision),tmp_path/'result.json')
    assert not calls and not (tmp_path/'result.json').exists()


@pytest.mark.parametrize('field,value',[('module','text'),('module','image-wide'),('descriptor_sha256','c'*64),('reference_sha256','c'*64),('input_sha256','unbound'),('source','main'),('reference_case_id','landscape-0')])
def test_wrong_module_source_descriptor_or_reference_fails_before_work(tmp_path,field,value):
    module,case,_,calls=fixture()
    with pytest.raises(ValueError):module.measure_reference_pipeline(replace(case,**{field:value}),tmp_path/'result.json')
    assert not calls


@pytest.mark.parametrize('bad',['zero','nonfinite','length','wrong','unnormalized','meta','half'])
def test_bad_actual_vector_never_becomes_a_timing_pass(tmp_path,bad):
    module,case,_,_=fixture()
    values={'zero':torch.zeros(768),'nonfinite':torch.full((768,),float('nan')),'length':torch.ones(767),'wrong':torch.nn.functional.one_hot(torch.tensor(5),768).float(),'unnormalized':case.expected*2,'meta':torch.empty(768,device='meta'),'half':case.expected.half()}
    with pytest.raises(ValueError):module.measure_reference_pipeline(replace(case,backbone=lambda value:values[bad]),tmp_path/'result.json')
    result=json.loads((tmp_path/'result.json').read_text());assert not result['success'] and result['phase']=='failed'


def test_input_mutation_is_detected_instead_of_reused_hash_labels(tmp_path):
    module,case,raw,_=fixture();original=case.encoder
    def mutate(value):
        actual=original(value);raw.add_(1);return actual
    with pytest.raises(ValueError):module.measure_reference_pipeline(replace(case,encoder=mutate),tmp_path/'result.json')
    assert json.loads((tmp_path/'result.json').read_text())['success'] is False


def test_wrong_encoder_output_is_really_passed_to_backbone_not_replaced_by_a_golden(tmp_path):
    module,case,_,_=fixture();seen=[]
    def backbone(value):seen.append(value.clone());result=torch.zeros(768);result[0]=1;result[1]=value[0];return result/result.norm()
    with pytest.raises(ValueError):module.measure_reference_pipeline(replace(case,encoder=lambda value:value,backbone=backbone),tmp_path/'result.json')
    assert len(seen)==1 and torch.equal(seen[0],torch.tensor([1.]))


def test_non_cpu_intermediate_is_rejected_before_the_next_stage(tmp_path):
    module,case,_,calls=fixture()
    with pytest.raises(ValueError):module.measure_reference_pipeline(replace(case,preprocess=lambda:torch.empty(1,device='meta')),tmp_path/'result.json')
    assert not calls


def test_non_fp32_active_model_weights_are_rejected_before_work(tmp_path):
    module,case,_,calls=fixture();case.active_modules[0].half()
    with pytest.raises(ValueError):module.measure_reference_pipeline(case,tmp_path/'result.json')
    assert not calls


def test_existing_report_and_capture_bytes_are_never_overwritten(tmp_path):
    module,case,_,calls=fixture();path=tmp_path/'result.json';path.write_bytes(b'original')
    with pytest.raises(FileExistsError):module.measure_reference_pipeline(case,path)
    assert path.read_bytes()==b'original' and not calls
    path.unlink();capture=tmp_path/'result-first.bin';capture.write_bytes(b'original capture')
    with pytest.raises(FileExistsError):module.measure_reference_pipeline(case,path)
    assert capture.read_bytes()==b'original capture' and not calls


def test_wrong_expected_reference_is_rejected_without_normalizing_it_away(tmp_path):
    module,case,_,calls=fixture()
    with pytest.raises(ValueError):module.measure_reference_pipeline(replace(case,expected=case.expected*2),tmp_path/'result.json')
    assert not calls


def test_stage_snapshot_observer_runs_before_consumption_without_replacing_actual_values(tmp_path):
    module, case, _, calls = fixture()
    observed = []
    def observer(name, value):
        observed.append((name, value.detach().clone()))
        return torch.tensor([999.])
    result = module.measure_reference_pipeline(replace(case, after_stage=observer), tmp_path / 'result.json')
    assert result['success'] and calls == ['preprocessing', 'encoder', 'backbone'] * 4
    assert [name for name, _ in observed] == ['preprocessing', 'encoder', 'backbone'] * 4
    assert observed[0][1].item() == 1 and observed[1][1].item() == 2


def test_stage_snapshot_time_is_separate_from_stage_and_chain_measurements(tmp_path, monkeypatch):
    module, case, _, _ = fixture()
    clock = {'wall': 0., 'cpu': 0.}
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock['wall'])
    monkeypatch.setattr(module.time, 'process_time', lambda: clock['cpu'])
    def timed(function):
        def run(*args):
            result = function(*args)
            clock['wall'] += .1
            clock['cpu'] += .05
            return result
        return run
    def observer(name, value):
        clock['wall'] += 1
        clock['cpu'] += .4
    actual = replace(case, preprocess=timed(case.preprocess), encoder=timed(case.encoder),
                     backbone=timed(case.backbone), after_stage=observer)
    result = module.measure_reference_pipeline(actual, tmp_path / 'result.json')
    for row in result['rounds']:
        assert row['wall_seconds'] == pytest.approx(.3)
        assert row['process_cpu_seconds'] == pytest.approx(.15)
        assert row['stage_observer_wall_seconds'] == pytest.approx(3)
        assert row['stage_observer_process_cpu_seconds'] == pytest.approx(1.2)
        assert all(stage['wall_seconds'] == pytest.approx(.1) for stage in row['stages'])
