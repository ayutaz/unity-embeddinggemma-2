from pathlib import Path
import yaml


def test_saved_conv_workflow_is_opt_in_pinned_and_retains_numeric_failure():
    path=Path(__file__).resolve().parents[2]/'.github/workflows/conv-reference-context.yml'
    assert path.is_file(),'Separate saved convolution context workflow is missing'
    workflow=yaml.load(path.read_text(encoding='utf-8'),Loader=yaml.BaseLoader)
    assert workflow['permissions']=={'contents':'read','actions':'read'}
    assert 'workflow_dispatch' in workflow['on'] and 'pull_request_target' not in workflow['on']
    job=workflow['jobs']['probe']
    assert job['strategy']['matrix']['context']==['native','avx2']
    assert 'conv-reference-diagnostics' in job['if'] and 'head.repo.full_name == github.repository' in job['if']
    steps=job['steps'];commands='\n'.join(step.get('run','') for step in steps)
    assert 'snapshot_download' not in commands and '11814567883' in commands and '11881872190' in commands
    assert 'dc43d56b896b64077d8e86d56c3fd2b89cfe3d31170eabaa387165ea961438d7' in commands
    assert '664a8242d079f53bf98d2032f9398de6d2c8d6bcd2312d1baaa35bb498ce5fb8' in commands
    assert 'DNNL_MAX_CPU_ISA=AVX2' in commands and job['env']['ONEDNN_VERBOSE']=='1'
    probe=next(step for step in steps if 'embeddinggemma_tools.conv_reference_context' in step.get('run',''))
    assert not probe.get('continue-on-error') and 'set -o pipefail' in probe['run']
    upload=next(step for step in steps if 'actions/upload-artifact' in step.get('uses',''))
    assert upload['if']=='always()' and upload['with']['path']=='artifacts/conv-context/'


def test_native_host_is_checked_before_any_large_reference_download():
    path=Path(__file__).resolve().parents[2]/'.github/workflows/conv-reference-context.yml'
    workflow=yaml.load(path.read_text(encoding='utf-8'),Loader=yaml.BaseLoader)
    steps=workflow['jobs']['probe']['steps']
    guards=[(index,step) for index,step in enumerate(steps) if 'missing_required_cpu' in step.get('run','')]
    assert len(guards)==1,'Record unavailable AVX512 before fetching a native model'
    index,guard=guards[0]
    assert guard['if']=="matrix.context == 'native'" and 'AVX512' in guard['run']
    assert not guard.get('continue-on-error') and 'raise SystemExit' in guard['run']
    fetch=next(index for index,step in enumerate(steps) if 'embeddinggemma_tools.artifact_zip' in step.get('run',''))
    assert index<fetch
