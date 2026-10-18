from pathlib import Path
import yaml
from workflow_privacy_assertions import assert_privacy_gated_upload


def test_static_matmul_workflow_checks_cpu_and_original_patch_before_partial_native_probe():
    path=Path(__file__).resolve().parents[2]/'.github/workflows/vision-matmul-context.yml'
    assert path.is_file(),'Original vision matmul-only workflow is missing'
    value=yaml.safe_load(path.read_text(encoding='utf-8'));job=value['jobs']['probe'];steps=job['steps']
    guard=next(i for i,s in enumerate(steps) if s.get('name')=='Require AMD host before original graph download')
    fetch=next(i for i,s in enumerate(steps) if s.get('name')=='Fetch original square graph and patch fixture')
    assert guard<fetch
    commands='\n'.join(s.get('run','') for s in steps)
    assert 'AuthenticAMD' in commands and '--artifact-id 11774238857' in commands
    assert '1e99fa140cb0084e624b5019d96ae424192eff77346ef2f7ffea892c1bea0607' in commands
    assert 'probe_saved_static_matmuls' in commands and "'mm_5', 'mm_6', 'mm_7'" in commands
    assert 'rows=2520' in commands and 'prefix_rows=16' in commands
    assert 'replay_saved_reference_context' not in commands and 'snapshot_download' not in commands
    upload=next(s for s in steps if 'actions/upload-artifact@' in s.get('uses',''))
    assert_privacy_gated_upload(upload)
    assert '.pt2' not in upload['with']['path']


def test_matmul_probe_is_reachable_through_existing_registered_dispatch_without_main_changes():
    root=Path(__file__).resolve().parents[2]/'.github/workflows'
    parent=yaml.safe_load((root/'reference-context.yml').read_text(encoding='utf-8'))
    trigger=parent.get('on',parent.get(True));inputs=trigger['workflow_dispatch']['inputs']
    assert 'original_matmul_only' in inputs,'Registered dispatcher matmul input is missing'
    assert inputs['original_matmul_only']['default'] is False
    job=parent['jobs']['original_static_matmul_probe']
    assert 'inputs.original_matmul_only' in job['if'] and job['uses']=='./.github/workflows/vision-matmul-context.yml'
    assert '!inputs.original_matmul_only' in parent['jobs']['original_vision_probe']['if']
    assert '!inputs.original_matmul_only' in parent['jobs']['original_rotary_probe']['if']
    child=yaml.safe_load((root/'vision-matmul-context.yml').read_text(encoding='utf-8'))
    assert 'workflow_call' in child.get('on',child.get(True))
