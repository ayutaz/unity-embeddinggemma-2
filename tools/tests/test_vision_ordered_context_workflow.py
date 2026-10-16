from pathlib import Path
import yaml


def test_ordered_context_is_a_separate_registered_opt_in_and_preserves_native_probe_modes():
    root=Path(__file__).resolve().parents[2]/'.github/workflows'
    parent=yaml.safe_load((root/'reference-context.yml').read_text(encoding='utf-8'))
    inputs=parent.get('on',parent.get(True))['workflow_dispatch']['inputs']
    assert 'original_softmax16_replay' in inputs,'Registered ordered-context opt-in is missing'
    assert inputs['original_softmax16_replay']['default'] is False
    job=parent['jobs']['original_ordered_context_probe']
    assert job['uses']=='./.github/workflows/vision-ordered-context.yml'
    assert 'inputs.original_vision' in job['if'] and 'inputs.original_softmax16_replay' in job['if']
    for name in ('original_vision_probe','original_rotary_probe','original_static_matmul_probe'):
        assert '!inputs.original_softmax16_replay' in parent['jobs'][name]['if']


def test_ordered_context_guards_original_host_patch_and_first_layer_before_full_capture():
    path=Path(__file__).resolve().parents[2]/'.github/workflows/vision-ordered-context.yml'
    assert path.is_file(),'Ordered-context diagnostic workflow is missing'
    value=yaml.safe_load(path.read_text(encoding='utf-8'));assert 'workflow_call' in value.get('on',value.get(True))
    steps=value['jobs']['probe']['steps'];commands='\n'.join(step.get('run','') for step in steps)
    host=next(i for i,s in enumerate(steps) if s.get('name')=='Require AMD host before model download')
    fetch=next(i for i,s in enumerate(steps) if s.get('name')=='Fetch original saved graph and unchanged reference')
    assert host<fetch and 'AuthenticAMD' in commands
    assert '--artifact-id 11774238857' in commands and 'snapshot_download' not in commands
    for text in ('1e99fa140cb0084e624b5019d96ae424192eff77346ef2f7ffea892c1bea0607',
                 'fb33092e54e997da999b87246aefe2089ce2d0f376a89586b8c329db069bb7fa',
                 'VisionSoftmax16Diagnostic','capture_saved_program_nodes','cpu_diagnostic_candidate_not_native_dispatch'):
        assert text in commands
    assert commands.index("'add_15'")<commands.index('capture_saved_program_nodes(program')
    assert 'maximum_absolute_error' in commands and 'observed_final_matches_normal_bits' in commands
    assert 'ATEN_CPU_CAPABILITY=' not in commands and 'MKL_CBWR=' not in commands
    upload=next(s for s in steps if 'actions/upload-artifact@' in s.get('uses',''))
    assert upload['if']=='always()' and '.pt2' not in upload['with']['path']
