import json
import pytest
from test_audio_cpu_reference import prepared,cohort,SOURCE,cli_arguments


@pytest.fixture
def native_prepared(prepared,monkeypatch):
    driver,state,output,_,snapshot=prepared
    from embeddinggemma_tools.cpu_fp32_reference import NativeFp32Reference
    built=[];sentinel=object()
    monkeypatch.setattr(NativeFp32Reference,'build',lambda path:(built.append(path),sentinel)[1])
    original=driver._load_modules
    def modules(*args,native_candidate=None,native=None):
        assert native_candidate==snapshot.parent/'candidate' and native is sentinel
        pre,owner,backbone=original(*args)
        owner.configuration={k:v for k,v in owner.configuration.items() if k in ('source','case_id','profile','references_used_as_inference_inputs','CPU_reference_comparison_qualified','GPU_validated','complete_performance_accepted')}
        from embeddinggemma_tools.native_regression_distribution import configuration,CONFIG_SHA
        from embeddinggemma_tools.audio_generation_candidate import candidate_contract
        transport,inventory=configuration()
        provenance=dict(executed_graph_sha256=inventory['files']['candidate-1119.pt2']['sha256'],graph_producer=candidate_contract()['producer_source'],
            case_id='silence',profile='frames-1119-valid-1119',graph_transport_contract_sha256=CONFIG_SHA,graph_distribution_inventory_sha256=transport['inventory_sha256'])
        owner.configuration.update(executor='native-fp32-reference',original_ONNX_bytes_executed=False,
            original_coefficients_verified_unchanged=False,closed=False,maximum_owned_parts=5,execution_schedule='chain-major',
            coefficient_graph_provenance=provenance,
            native_reference=dict(source_sha256='d'*64,compiler_binary_sha256='e'*64,native_library_sha256='f'*64,
                executor='strict-native-block192-fma-reference-only',native_threads=4,AVX2_FMA_and_OS_vector_state_verified=True,model_weights_changed=False,GPU_execution_validated=False),
            native_actual_calls=dict(linear=0,matmul=684,fma=4800,stem_convolution=8))
        close=owner.close
        def release():close();owner.configuration['closed']=True
        def verify():owner.configuration['original_coefficients_verified_unchanged']=True;state['coefficient_verifications']=state.get('coefficient_verifications',0)+1;return True
        owner.close=release;owner.verify_coefficients=verify
        return pre,owner,backbone
    monkeypatch.setattr(driver,'_load_modules',modules)
    def run():
        base=snapshot.parent
        return driver.measure_audio_cpu_reference(snapshot,base/'pre',base/'encoder',base/'document.json',None,output,
            source=SOURCE,case_id='silence',native_candidate=base/'candidate',native_build=base/'native-build')
    return driver,state,output,snapshot,run,built


def test_explicit_native_carries_preserve_gates_and_measure_load_release(native_prepared):
    driver,state,output,snapshot,run,built=native_prepared
    result=run()
    assert len(built)==1 and state['pre_calls']==state['encoder_calls']==state['backbone_calls']==4
    assert state['closes']==state['coefficient_verifications']==1
    assert result['encoder_mode']=='native-fp32-reference' and result['release_completed'] is True
    assert result['encoder_configuration']['closed'] is True
    assert result['encoder_configuration']['original_coefficients_verified_unchanged'] is True
    assert result['load_wall_seconds']>=0 and result['release_wall_seconds']>=0
    assert result['coefficient_verification_wall_seconds']>=0
    base=snapshot.parent
    audit=driver.audit_audio_cpu_reference(output,base/'pre',base/'encoder',base/'document.json',source=SOURCE)
    assert audit['all_four_chains_match_originals'] and audit['encoder_mode']=='native-fp32-reference'


@pytest.mark.parametrize('field',['executor','closed','original_coefficients_verified_unchanged','native_actual_calls','native_reference','graph_sha256','graph_producer'])
def test_native_saved_audit_rejects_changed_executor_proof(native_prepared,field):
    driver,_,output,snapshot,run,_=native_prepared;run()
    path=output/'audio-cpu-reference.json';report=json.loads(path.read_bytes())
    if field=='executor':report['encoder_configuration'][field]='onnx-cpu-reference'
    elif field in ('closed','original_coefficients_verified_unchanged'):report['encoder_configuration'][field]=False
    elif field=='graph_sha256':report['encoder_configuration']['coefficient_graph_provenance']['executed_graph_sha256']='0'*64
    elif field=='graph_producer':report['encoder_configuration']['coefficient_graph_provenance']['graph_producer']='0'*40
    else:report['encoder_configuration'][field]={}
    path.write_text(json.dumps(report),encoding='utf-8');base=snapshot.parent
    with pytest.raises(ValueError):driver.audit_audio_cpu_reference(output,base/'pre',base/'encoder',base/'document.json',source=SOURCE)


@pytest.mark.parametrize('fault',['missing_build','mixed_chain','retained_source'])
def test_native_options_fail_before_compilation_or_model_loading(native_prepared,fault):
    driver,state,output,snapshot,_,built=native_prepared;base=snapshot.parent
    kwargs=dict(native_candidate=base/'candidate',native_build=base/'build')
    chain=None
    if fault=='missing_build':kwargs.pop('native_build')
    elif fault=='mixed_chain':chain=base/'chain.json'
    else:kwargs['retained_producer_source']='b'*40
    with pytest.raises(ValueError):driver.measure_audio_cpu_reference(snapshot,base/'pre',base/'encoder',base/'document.json',chain,output,source=SOURCE,case_id='silence',**kwargs)
    assert not built and state['encoder_calls']==state['closes']==0


def test_native_failure_releases_resources_and_preserves_actual_intermediates(native_prepared):
    driver,state,output,_,run,_=native_prepared;state['backbone_error']=True
    with pytest.raises(ValueError):run()
    report=json.loads((output/'audio-cpu-reference.json').read_bytes())
    assert report['success'] is False and report['release_completed'] is True and state['closes']==1
    assert set(report['rounds'][0]['captures'])=={'features','frame_mask','encoder'}


def test_cli_explicit_native_measure_then_audit(native_prepared,monkeypatch):
    driver,_,output,snapshot,_,_=native_prepared;base=snapshot.parent
    arguments=cli_arguments(snapshot,output)
    index=arguments.index('--chain-report');del arguments[index:index+2]
    arguments+=['--native-candidate',str(base/'candidate'),'--native-build',str(base/'native-build')]
    monkeypatch.setattr(driver.torch,'set_num_interop_threads',lambda _:None)
    assert driver.main(arguments)==0
    arguments=cli_arguments(snapshot,output,'audit')
    assert driver.main(arguments)==0
    assert driver.main(arguments+['--native-candidate',str(base/'candidate')])==1
