"""Actual CLI/HTTPS contracts with opaque tiny objects; not model/GPU tests."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from media_setup_https_harness import LocalBundleServer, create_certificate
from test_media_setup import fixture, digest

TOOLS = Path(__file__).resolve().parents[1]


def cli(server, output, pin, *, trust=True):
    env = dict(os.environ, NO_PROXY='127.0.0.1,localhost')
    if trust:
        env['SSL_CERT_FILE'] = str(server.certificate)
    else:
        env.pop('SSL_CERT_FILE', None)
    return subprocess.run([sys.executable, '-m', 'embeddinggemma_tools.media_setup',
        '--descriptor-url', server.url, '--descriptor-sha256', pin, '--output', str(output)],
        cwd=TOOLS, env=env, capture_output=True, text=True, timeout=30)


@pytest.fixture
def https(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    data, payloads = fixture(); raw = json.dumps(data).encode()
    (source/'bundle.json').write_bytes(raw)
    for name, payload in payloads.items():
        path = source/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
    certificate, private_key = create_certificate(tmp_path/'tls')
    with LocalBundleServer(source, certificate, private_key) as server:
        yield server, raw, data, payloads


def test_cli_requires_verified_certificate_and_acquires_over_actual_https(https, tmp_path):
    server, raw, data, payloads = https
    bad = cli(server, tmp_path/'untrusted', digest(raw), trust=False)
    assert bad.returncode == 1 and 'CERTIFICATE_VERIFY_FAILED' in bad.stderr
    assert server.object_requests == [] and not (tmp_path/'untrusted').exists()
    result = cli(server, tmp_path/'output', digest(raw))
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt['success'] and receipt['downloaded_files'] == len(payloads)
    assert not receipt['inference_executed'] and not receipt['gpu_validated']
    assert json.loads((tmp_path/'output/bundle.json').read_text()) == data


def test_cli_retries_real_truncated_transfer_and_repairs_same_size_corruption(https, tmp_path):
    server, raw, _, payloads = https; output = tmp_path/'output'
    server.reset(truncate_ordinal=2)
    failed = cli(server, output, digest(raw))
    assert failed.returncode == 1 and not (output/'bundle.json').exists()
    assert not json.loads((output/'preparation.json').read_text())['success']
    assert not list(output.rglob('*.pending'))
    server.reset()
    recovered = cli(server, output, digest(raw))
    assert recovered.returncode == 0, recovered.stderr
    assert json.loads(recovered.stdout)['reused_files'] == 1
    server.reset()
    warm = cli(server, output, digest(raw))
    assert warm.returncode == 0 and server.object_requests == []
    assert json.loads(warm.stdout)['reused_files'] == len(payloads)
    name = max(payloads, key=lambda key: len(payloads[key]))
    path = output/name; payload = path.read_bytes()
    path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    assert path.stat().st_size == len(payload) and hashlib.sha256(path.read_bytes()).hexdigest() != digest(payload)
    server.reset()
    repaired = cli(server, output, digest(raw))
    assert repaired.returncode == 0 and server.object_requests == [name]
    assert json.loads(repaired.stdout)['downloaded_files'] == 1 and path.read_bytes() == payload


def test_cli_missing_https_object_retains_old_descriptor_until_repair(https, tmp_path):
    server, raw, _, payloads = https; output = tmp_path/'output'
    assert cli(server, output, digest(raw)).returncode == 0
    previous = (output/'bundle.json').read_bytes()
    name = max(payloads, key=lambda key: len(payloads[key])); path = output/name
    payload = path.read_bytes(); path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    server.reset(missing_object=name)
    failed = cli(server, output, digest(raw))
    assert failed.returncode == 1 and '404' in failed.stderr
    assert (output/'bundle.json').read_bytes() == previous
    assert not json.loads((output/'preparation.json').read_text())['success']
    assert not list(output.rglob('*.pending'))
    server.reset()
    assert cli(server, output, digest(raw)).returncode == 0
    assert path.read_bytes() == payload


@pytest.mark.parametrize('mode', ['missing', 'expired', 'wrong-revision', 'wrong-pin'])
def test_cli_rejects_https_source_failure_before_object_requests(https, tmp_path, mode):
    server, raw, data, _ = https; output = tmp_path/'output'; pin = digest(raw)
    if mode == 'missing': server.reset(descriptor_status=404)
    elif mode == 'expired': server.reset(descriptor_status=410)
    elif mode == 'wrong-revision':
        data['model_revision'] = '0'*40
        wrong = json.dumps(data).encode(); server.reset(descriptor_body=wrong); pin = digest(wrong)
    elif mode == 'wrong-pin': pin = '0'*64
    result = cli(server, output, pin)
    assert result.returncode == 1 and not output.exists()
    assert server.object_requests == []


def test_retry_after_forced_cli_termination_releases_lease_and_removes_partial(https, tmp_path):
    server, _, data, payloads = https
    old_model = data['modules']['text']['model']
    payload = b'synthetic transfer fixture, not a real model\n' * 100000
    assert len(payload) > 4*1024*1024
    new_model = 'objects/'+digest(payload)+'.sentis'
    def replace(value):
        if isinstance(value, dict): return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list): return [replace(item) for item in value]
        return new_model if value == old_model else value
    data = replace(data); data['files'].pop(old_model)
    data['files'][new_model] = dict(bytes=len(payload), sha256=digest(payload))
    (server.root/new_model).write_bytes(payload)
    descriptor = json.dumps(data).encode(); output = tmp_path/'output'
    resume = threading.Event()
    server.reset(descriptor_body=descriptor, pause_object=new_model, resume_event=resume)
    env = dict(os.environ, NO_PROXY='127.0.0.1,localhost', SSL_CERT_FILE=str(server.certificate))
    command = [sys.executable, '-m', 'embeddinggemma_tools.media_setup', '--descriptor-url', server.url,
               '--descriptor-sha256', digest(descriptor), '--output', str(output)]
    process = subprocess.Popen(command, cwd=TOOLS, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    try:
        deadline = time.monotonic()+10
        pending = []
        while time.monotonic() < deadline and process.poll() is None:
            pending = [path for path in (output/'objects').glob('*.pending') if path.stat().st_size >= 1024*1024]
            if pending: break
            time.sleep(.02)
        assert process.poll() is None and pending, 'The child must be alive with an actual received partial file before kill.'
        before_requests = server.object_requests
        before_receipt = (output/'preparation.json').read_bytes() if (output/'preparation.json').exists() else None
        blocked = cli(server, output, digest(descriptor))
        assert blocked.returncode == 1 and 'already active' in blocked.stderr
        assert server.object_requests == before_requests and process.poll() is None
        assert all(path.exists() and path.stat().st_size >= 1024*1024 for path in pending)
        assert ((output/'preparation.json').read_bytes() if (output/'preparation.json').exists() else None) == before_receipt
        process.kill(); process.communicate(timeout=10)
        assert process.returncode != 0 and any(path.exists() for path in pending)
    finally:
        if process.poll() is None: process.kill(); process.communicate(timeout=10)
        resume.set()
    deadline = time.monotonic()+5
    while not any(row.get('paused') for row in server.requests) and time.monotonic() < deadline:
        time.sleep(.02)
    assert any(row.get('paused') for row in server.requests)
    server.reset(descriptor_body=descriptor)
    retried = cli(server, output, digest(descriptor))
    assert retried.returncode == 0, retried.stderr
    assert not list(output.rglob('*.pending'))
    receipt = json.loads(retried.stdout)
    assert receipt['stale_pending_files_removed'] >= 1
    assert (output/new_model).read_bytes() == payload
