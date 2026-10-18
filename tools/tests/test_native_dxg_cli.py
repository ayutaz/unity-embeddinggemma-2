"""Opt-in Windows native collector protocol checks; these do not prove model/GPU execution."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import pytest

@pytest.fixture
def observer():
    value=os.environ.get('EG_NATIVE_DXG_OBSERVER_EXE')
    if os.name!='nt' or not value:
        pytest.skip('Windows native diagnostic binary must be explicitly provided; not GPU evidence.')
    path=Path(value)
    assert path.is_file()
    return path

def run(exe,*args):
    return subprocess.run([str(exe),*map(str,args)],capture_output=True,text=True,timeout=15)

def test_help_describes_finite_capture_without_starting_a_session(observer):
    result=run(observer,'--help')
    assert result.returncode==0 and 'PID' in result.stdout and '900' in result.stdout

@pytest.mark.parametrize('pid,seconds',[('0','1'),('abc','1'),('42','0'),('42','901'),('42','1junk')])
def test_bad_identity_or_duration_is_rejected_before_creating_output(observer,tmp_path,pid,seconds):
    output=tmp_path/'capture.jsonl'
    result=run(observer,pid,seconds,output)
    assert result.returncode==2 and not output.exists()

def test_missing_target_is_not_a_successful_empty_capture(observer,tmp_path):
    output=tmp_path/'missing.jsonl'
    result=run(observer,4294967292,1,output)
    assert result.returncode==2 and not output.exists()

def test_existing_output_is_never_truncated(observer,tmp_path):
    output=tmp_path/'existing.jsonl'
    output.write_bytes(b'preserve-existing-observation')
    result=run(observer,os.getpid(),1,output)
    assert result.returncode==2 and output.read_bytes()==b'preserve-existing-observation'

def test_owned_empty_target_has_clock_identity_loss_and_closure_receipts(observer,tmp_path):
    output=tmp_path/'owned.jsonl'
    result=run(observer,os.getpid(),1,output)
    assert result.returncode==0, result.stdout+result.stderr
    rows=[json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
    header,footer=rows[0],rows[-1]
    assert header['kind']=='capture_begin' and footer['kind']=='capture_end'
    assert header['pid']==footer['pid']==os.getpid()
    assert header['target_creation_filetime']>0 and header['timestamp_unit']=='qpc_epoch_ns'
    assert header['qpc_frequency']==footer['qpc_frequency']>0
    for anchor in (header,footer):
        assert 0<anchor['qpc_before']<=anchor['qpc_after']
        assert anchor['filetime_utc_100ns']>116444736000000000
    assert header['qpc_before']<footer['qpc_before']
    assert footer['session_stopped'] and footer['events_lost']==footer['buffers_lost']==0
    assert not footer['real_model_validated'] and not footer['gpu_control_validated']
    assert footer['invalid_records']==footer['dropped_records']==0
    assert all(row.get('pid')==os.getpid() for row in rows[1:-1])
    # The session is named in its receipt so normal closure can be checked independently.
    sessions=subprocess.run(['logman','query','-ets'],capture_output=True,text=True,timeout=10)
    assert sessions.returncode==0 and footer['session_name'] not in sessions.stdout
