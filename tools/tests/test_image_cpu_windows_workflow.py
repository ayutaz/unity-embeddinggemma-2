"""Windows image reference CI declares each arithmetic policy before execution."""
import fnmatch
import os
from pathlib import Path
import shutil
import subprocess
import sys
import json

import pytest
import yaml

from workflow_privacy_assertions import assert_privacy_gated_upload


def workflow():
    return yaml.safe_load((Path(__file__).parents[2]/'.github/workflows/image-cpu-windows.yml').read_text(encoding='utf-8'))


def image_job():
    return workflow()['jobs']['image_cpu_windows']


def test_ci_limits_all_cpu_dispatchers_before_importing_torch():
    selected={'ATEN_CPU_CAPABILITY':'avx2','MKL_ENABLE_INSTRUCTIONS':'AVX2','DNNL_MAX_CPU_ISA':'AVX2','MKL_CBWR':'COMPATIBLE'}
    environment=image_job()['env']
    assert {key:environment.get(key) for key in selected}==selected
    result=subprocess.run([sys.executable,'-P','-c',
        'import json; from embeddinggemma_tools.cpu_execution_context import cpu_execution_context; print(json.dumps(cpu_execution_context()))'],
        env={**os.environ,**selected,'PYTHONPATH':str(Path(__file__).parents[1])+os.pathsep+os.environ.get('PYTHONPATH','')},
        capture_output=True,text=True,check=True)
    context=json.loads(result.stdout)
    assert context['CPU_capability']=='AVX2'
    assert {key:context['math_environment'][key] for key in selected}==selected


def test_profiles_and_statistics_are_fixed_before_execution_without_fallback():
    job=image_job()
    assert job['runs-on']=='windows-latest' and 0<job['timeout-minutes']<=30
    assert job['strategy']=={'fail-fast':False,'matrix':{'include':[
        {'profile':'square','statistics':'sixteen-lane'},
        {'profile':'landscape','statistics':'native'},
        {'profile':'portrait','statistics':'native'}]}}
    assert job['defaults']['run']=={'shell':'bash','working-directory':'tools'}
    assert job['env']['OMP_NUM_THREADS']==job['env']['MKL_NUM_THREADS']=='4'
    assert job['env']['STATISTICS_REFERENCE']=='${{ matrix.statistics }}'
    assert not job.get('continue-on-error')
    runs=[step.get('run','') for step in job['steps']]
    assert not any(step.get('continue-on-error') for step in job['steps'])
    assert not any('|| true' in run or 'frozen_audio' in run or 'prepare-vision' in run for run in runs)
    measure=next(run for run in runs if 'image_cpu_reference measure' in run)
    assert '--statistics-reference "$STATISTICS_REFERENCE"' in measure
    assert 'env MKL_CBWR=COMPATIBLE' in measure and 'env -u MKL_CBWR' not in measure


@pytest.mark.parametrize('name',['tools/embeddinggemma_tools/image_cpu_reference.py',
    'tools/embeddinggemma_tools/vision_cpu_simd_reference.py','tools/embeddinggemma_tools/export.py',
    'tools/tests/test_image_cpu_windows_workflow.py'])
def test_relevant_execution_changes_trigger_the_check(name):
    data=workflow();events=data.get('on',data.get(True))
    assert set(events)=={'pull_request','workflow_dispatch'}
    assert events['pull_request']['branches']==['main']
    assert any(fnmatch.fnmatchcase(name,pattern) for pattern in events['pull_request']['paths'])
    assert data['permissions']=={'contents':'read'}


def test_actual_original_input_execution_precedes_independent_audit_and_exact_publication_guard():
    steps=image_job()['steps']
    positions=[next(i for i,step in enumerate(steps) if marker in step.get('run','')) for marker in (
        'embeddinggemma_tools.image_reference_distribution', 'image_cpu_reference measure','image_cpu_reference audit')]
    assert positions==sorted(set(positions))
    for index in positions[1:]:
        assert '--source "$GITHUB_SHA"' in steps[index]['run']
        assert '--references ../artifacts/original-image-references' in steps[index]['run']
    assert '--profile "$PROFILE"' in steps[positions[1]]['run']
    index,upload=next((i,step) for i,step in enumerate(steps) if 'actions/upload-artifact@' in step.get('uses',''))
    assert_privacy_gated_upload(upload)
    guard=steps[index-1]
    assert guard['if']=='always()' and guard['uses']=='./.github/actions/privacy-artifact-guard'
    assert guard['with']['paths']==upload['with']['path']
    assert set(upload['with']['path'].splitlines())=={
        'artifacts/image-cpu-${{ matrix.profile }}/*.json','artifacts/image-cpu-${{ matrix.profile }}/*.bin'}
    assert upload['with']['name']=='image-cpu-windows-${{ matrix.profile }}-${{ github.sha }}'


@pytest.mark.parametrize('marker',['image_reference_distribution','image_cpu_reference measure','image_cpu_reference audit'])
@pytest.mark.parametrize('status',[0,7])
def test_actual_shell_preserves_failure_and_keeps_diagnostics_private(tmp_path,marker,status):
    executable=shutil.which('git' if os.name=='nt' else 'bash')
    bash=Path(executable).resolve().parents[1]/'bin/bash.exe' if executable and os.name=='nt' else Path(executable) if executable else None
    if bash is None or not bash.is_file():
        pytest.skip('Bash unavailable; actual CI shell handling is not verified')
    command=next(step['run'] for step in image_job()['steps'] if marker in step.get('run',''))
    working=tmp_path/'tools';working.mkdir()
    logs=tmp_path/'artifacts/private-image-cpu-logs';logs.mkdir(parents=True)
    diagnostic='synthetic-private-image-diagnostic'
    # env executes the shim just as it executes uv in the runner's shell.
    shim=tmp_path/'bin';shim.mkdir()
    (shim/'uv').write_text('#!/usr/bin/env bash\nprintf "%s\\n" '+diagnostic+'\nexit '+str(status)+'\n',encoding='utf-8')
    (shim/'uv').chmod(0o755)
    result=subprocess.run([str(bash),'--noprofile','--norc','-e','-o','pipefail','-c',command],cwd=working,
        env={**os.environ,'PATH':str(shim)+os.pathsep+os.environ.get('PATH',''),'PROFILE':'square',
             'STATISTICS_REFERENCE':'sixteen-lane','GITHUB_SHA':'a'*40},capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    assert result.returncode==status
    assert diagnostic.encode() not in result.stdout+result.stderr
    assert any(diagnostic in log.read_text(encoding='utf-8') for log in logs.glob('*.log'))
