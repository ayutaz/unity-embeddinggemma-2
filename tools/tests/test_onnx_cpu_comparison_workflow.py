from pathlib import Path
import tomllib

import yaml

ROOT = Path(__file__).resolve().parents[2]


def workflow(name):
    return yaml.load((ROOT / '.github/workflows' / name).read_text(encoding='utf-8'),
                     Loader=yaml.BaseLoader)


def test_python_ci_installs_and_uses_the_optional_cpu_group_in_every_environment():
    steps = workflow('python-tests.yml')['jobs']['unit']['steps']
    sync = next(s['run'] for s in steps if s.get('run', '').startswith('uv sync'))
    tests = next(s['run'] for s in steps if 'python -m pytest' in s.get('run', ''))
    assert '--group cpu-comparison' in sync and '--group cpu-comparison' in tests
    project = tomllib.loads((ROOT / 'tools/pyproject.toml').read_text(encoding='utf-8'))
    assert any(d.startswith('onnxruntime') for d in project['dependency-groups']['cpu-comparison'])
    assert not any(d.startswith('onnxruntime') for d in project['project']['dependencies'])


def test_cpu_all_six_replays_use_audited_actual_chains_and_are_hard_gates():
    job = workflow('pinned-audio-acceptance.yml')['jobs']['execute']
    steps = job['steps']
    cpu = next(s for s in steps if 'embeddinggemma_tools.onnx_cpu_comparison' in s.get('run', ''))
    assert cpu.get('if', 'success()') == 'success()'
    assert cpu.get('continue-on-error', 'false') == 'false'
    assert '--group cpu-comparison' in cpu['run']
    assert '--graph-optimizations disabled' in cpu['run']
    assert '--execution-schedule chain-major' in cpu['run']
    assert '--node-order priority' in cpu['run']
    assert 40 <= int(job['timeout-minutes']) <= 55
    assert int(cpu['timeout-minutes']) == 12
    assert '--chain-report ../artifacts/pinned-audio/frozen-audio-chained-onnx.json' in cpu['run']
    assert '--original ../artifacts/original' in cpu['run']
    assert '--source "$GITHUB_SHA"' in cpu['run']
    assert '--output ../artifacts/pinned-cpu-onnx/cpu-onnx-replay.json' in cpu['run']
    numerical = next(s for s in steps if 'embeddinggemma_tools.frozen_audio_chained_onnx' in s.get('run', ''))
    guard = next(s for s in steps if s.get('uses') == './.github/actions/privacy-artifact-guard')
    original_upload = next(s for s in steps if s.get('with', {}).get('name', '').startswith('pinned-audio-'))
    assert steps.index(numerical) < steps.index(guard) < steps.index(original_upload) < steps.index(cpu)
    assert 'artifacts/pinned-audio/*.json' in guard['with']['paths']
    assert 'artifacts/pinned-audio/*.bin' in guard['with']['paths']
    assert job['strategy']['matrix']['case_id'] == ['silence', 'impulse', 'sine', 'nyquist', 'noise', 'padded_noise']


def test_cpu_diagnostics_have_their_own_guarded_upload_after_failure():
    steps = workflow('pinned-audio-acceptance.yml')['jobs']['execute']['steps']
    guard = next(s for s in steps if s.get('id') == 'privacy_cpu_upload')
    assert 'always()' in guard['if'] and 'matrix.case_id' not in guard['if']
    assert 'artifacts/pinned-cpu-onnx/*.json' in guard['with']['paths']
    assert 'artifacts/pinned-cpu-onnx/*.bin' in guard['with']['paths']
    upload = next(s for s in steps if s.get('with', {}).get('name', '').startswith('cpu-onnx-replay-'))
    assert "steps.privacy_cpu_upload.outcome == 'success'" in upload['if']
    assert 'always()' in upload['if']
    assert 'matrix.case_id' not in upload['if']
    assert upload['with']['name'] == 'cpu-onnx-replay-${{ github.sha }}-${{ matrix.case_id }}'
    assert steps.index(guard) < steps.index(upload)
