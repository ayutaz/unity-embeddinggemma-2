from pathlib import Path
import os
import shlex
import subprocess
import sys

import pytest
import yaml

from workflow_privacy_assertions import assert_privacy_gated_upload


def workflow():
    return yaml.safe_load((Path(__file__).parents[2]/'.github/workflows/reference-context.yml').read_text(encoding='utf-8'))


def audio_job():
    return workflow()['jobs']['audio_cpu_measurement']


def test_explicit_audio_mode_is_within_dispatch_limit_and_cannot_fall_through_to_other_models():
    value=workflow()
    inputs=value.get('on',value.get(True))['workflow_dispatch']['inputs']
    assert len(inputs)<=10 and inputs['audio_cpu_measurement']['type']=='boolean'
    assert inputs['audio_cpu_measurement']['default'] is False
    assert "github.event_name == 'workflow_dispatch'" in audio_job()['if']
    assert 'inputs.audio_cpu_measurement' in audio_job()['if']
    for name,job in value['jobs'].items():
        if name!='audio_cpu_measurement':assert '!inputs.audio_cpu_measurement' in job['if']


def test_both_actual_GPU_benchmark_input_cases_have_separate_bounded_CPU_jobs():
    job=audio_job()
    assert job['runs-on']=='ubuntu-latest' and 0<job['timeout-minutes']<=90
    assert job['strategy']['matrix']=={'case_id':['silence','padded_noise']}
    assert job['strategy']['fail-fast'] is False
    assert job['env']['OMP_NUM_THREADS']==job['env']['MKL_NUM_THREADS']==job['env']['OPENBLAS_NUM_THREADS']=='4'
    assert job['env']['MKL_CBWR']=='COMPATIBLE' and job['env']['MKL_DYNAMIC']=='FALSE'
    assert not job.get('continue-on-error') and not any(v.get('continue-on-error') for v in job['steps'])


def test_conflicting_selections_are_rejected_before_dependencies_download_or_conversion():
    steps=audio_job()['steps']
    index,guard=next((i,v) for i,v in enumerate(steps) if v.get('name')=='Reject incompatible audio CPU selections')
    env=guard['env']['LEGACY_SELECTED']
    for name in ('image_cpu_measurement','image_cpu_host_comparison','original_vision','require_avx512_native',
        'original_rotary_only','original_matmul_only','original_softmax16_replay','original_layer_four_detail','original_blas_context'):
        assert 'inputs.'+name in env
    assert index<next(i for i,v in enumerate(steps) if v.get('run')=='uv sync --locked --group cpu-comparison')


@pytest.mark.parametrize('selection,expected',[('false',0),('true',1),('',1),('False',1)])
def test_actual_admission_code_accepts_only_explicit_nonconflicting_selection(selection,expected):
    guard=next(v for v in audio_job()['steps'] if v.get('name')=='Reject incompatible audio CPU selections')
    args=shlex.split(guard['run'])
    code=args[args.index('-c')+1]
    result=subprocess.run([sys.executable,'-c',code],env={**os.environ,'LEGACY_SELECTED':selection},capture_output=True)
    assert result.returncode==expected


def test_existing_original_graphs_are_generated_once_before_actual_PCM_chain_and_separate_audit():
    steps=audio_job()['steps']
    def command_index(text):return next(i for i,v in enumerate(steps) if text in v.get('run',''))
    fetch=command_index('selected_audio_generation fetch-distribution')
    materialize=command_index('embeddinggemma_tools.audio_cpu_inputs')
    graph=command_index('embeddinggemma_tools.frozen_audio_chained_onnx')
    measure=command_index('embeddinggemma_tools.audio_cpu_reference measure')
    audit=command_index('embeddinggemma_tools.audio_cpu_reference audit')
    assert fetch<materialize<graph<measure<audit
    assert '--fetch-missing-inputs' in steps[materialize]['run']
    assert '--raw-input --variant ordered_original_midpoint_bmm' in steps[graph]['run']
    for index in (materialize,measure,audit):assert '--source "$GITHUB_SHA"' in steps[index]['run']
    assert '--case "$CASE_ID"' in steps[graph]['run'] and '--case "$CASE_ID"' in steps[measure]['run']
    assert not any('embeddinggemma_tools.onnx_cpu_comparison' in v.get('run','') or 'artifact_zip' in v.get('run','') or '|| true' in v.get('run','') for v in steps)


def test_complete_or_partial_numerical_evidence_has_exact_guard_without_PCM_or_weights():
    steps=audio_job()['steps']
    index,upload=next((i,v) for i,v in enumerate(steps) if 'actions/upload-artifact@' in v.get('uses',''))
    assert_privacy_gated_upload(upload)
    guard=steps[index-1]
    assert guard['uses']=='./.github/actions/privacy-artifact-guard' and guard['if']=='always()'
    paths=upload['with']['path']
    assert paths==guard['with']['paths']
    assert 'artifacts/audio-cpu-pinned/*.bin' in paths and 'artifacts/audio-cpu-full/*.bin' in paths
    assert 'artifacts/audio-cpu-inputs/audio-cpu-inputs.json' in paths
    assert 'artifacts/audio-cpu-inputs/*' not in paths
    assert not any(v in paths for v in ('snapshot','*.onnx','*.pt2','*.safetensors','*.log'))
