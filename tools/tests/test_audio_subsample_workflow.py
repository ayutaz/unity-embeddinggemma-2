from pathlib import Path

import yaml


def test_subsample_replay_workflow_pins_exact_features_and_audits_before_download():
    workflow = yaml.safe_load((Path(__file__).resolve().parents[2] / '.github/workflows/audio-layer-replay.yml').read_text())
    steps = workflow['jobs']['replay']['steps']
    fetch = next(step['run'] for step in steps if step.get('name') == 'Fetch only frozen stage input and output')
    member = 'audio-runtime-reference/frames-1119-valid-1119/silence-features-fp32.bin'
    assert '--member "audio-runtime-reference/frames-1119-valid-1119/${REPLAY_SIGNAL}-features-fp32.bin"' in fetch
    triggers=workflow.get('on',workflow.get(True))
    assert triggers['workflow_dispatch']['inputs']['signal']['default']=='silence'
    binding=workflow['jobs']['replay']['env']['REPLAY_SIGNAL']
    assert "inputs.scope == 'attention' || inputs.scope == 'subsample'" in binding
    assert "|| 'silence'" in binding  # Full/default scope still fetches the original silence input.
    audit = next(step['run'] for step in steps if step.get('name') == 'Verify pinned context manifests before model download')
    assert '366de71cb4e45e4feda58a605babe4ea9d8693a7e37e84eac0735a7c42d0ad24' in audit
    producer = next(step['run'] for step in steps if step.get('name') == 'Replay official layer 0 against unchanged native final')
    assert '--subsample-input ../artifacts/frozen-audio-layer/' + member in producer
    assert '--subsample-input-sha256 366de71cb4e45e4feda58a605babe4ea9d8693a7e37e84eac0735a7c42d0ad24' in producer
