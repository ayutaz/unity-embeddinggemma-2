"""Mock provider errors and PID lifetime without launching Unity or a GPU."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

PWSH = shutil.which('pwsh')
pytestmark = pytest.mark.skipif(PWSH is None, reason='PowerShell is required for collector protocol tests')
COLLECTOR = Path(__file__).resolve().parents[1] / 'observe_owned_gpu_engine.ps1'


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def collect(tmp_path, scenario):
    receipt = tmp_path / 'process.json'
    output = tmp_path / 'counter.jsonl'
    receipt.write_text('{"pid":12,"terminal":false}', encoding='utf-8')
    script = tmp_path / 'provider.ps1'
    script.write_text('''
$fixtureReceipt = RECEIPT
$fixtureOutput = OUTPUT
$scenario = SCENARIO
$created = [DateTime]::UtcNow
$script:queryFinished = $false
function Get-CimInstance {
    param($ClassName, $Filter)
    if($script:queryFinished -and $scenario -eq 'identity'){return $null}
    [pscustomobject]@{Name='Unity.exe';CommandLine='Unity.exe -projectPath C:/unity-embeddinggemma-2/test';CreationDate=$created}
}
function Get-Counter {
    [CmdletBinding()]param([string[]]$Counter)
    $script:queryFinished = $true
    [IO.File]::WriteAllText($fixtureReceipt,'{"pid":12,"terminal":true}')
    if($scenario -eq 'fatal'){throw 'provider unavailable'}
    $moment=[DateTime]::Now
    $ownedStatus=if($scenario -eq 'owned-invalid'){0xc0000bc6L}else{0L}
    if($scenario -in @('foreign-invalid','owned-invalid')){Write-Error 'One provider sample is invalid'}
    $samples=@(
        [pscustomobject]@{InstanceName='pid_12_luid_0x0_0x1_phys_0_eng_0_engtype_3D';CounterType='Timer100Ns';TimeBase=10000000L;RawValue=20L;Timestamp=$moment;Timestamp100NSec=[DateTime]::SpecifyKind($moment,[DateTimeKind]::Utc).ToFileTimeUtc();Status=$ownedStatus},
        [pscustomobject]@{InstanceName='pid_123_luid_0x0_0x1_phys_0_eng_0_engtype_3D';CounterType='Timer100Ns';TimeBase=10000000L;RawValue=99L;Timestamp=$moment;Timestamp100NSec=0L;Status=0xc0000bc6L})
    [pscustomobject]@{Timestamp=$moment;CounterSamples=$samples}
}
. COLLECTOR -Receipt $fixtureReceipt -Output $fixtureOutput
'''.replace('RECEIPT', literal(receipt)).replace('OUTPUT', literal(output))
        .replace('SCENARIO', literal(scenario)).replace('COLLECTOR', literal(COLLECTOR)), encoding='utf-8')
    subprocess.run([PWSH, '-NoProfile', '-File', str(script)], capture_output=True, text=True, check=True, timeout=25)
    rows = [json.loads(line) for line in output.read_text(encoding='utf-8-sig').splitlines()]
    assert len(rows) == 1
    return rows[0]


def test_foreign_invalid_status_does_not_discard_valid_owned_counter(tmp_path):
    row = collect(tmp_path, 'foreign-invalid')
    assert row['timestamp_source'] == 'counter_sample_set'
    assert row['error'] is None and row['provider_errors']
    assert len(row['rows']) == 1 and row['rows'][0]['status'] == 0
    assert row['rows'][0]['timestamp_100ns'] == row['timestamp_100ns']


def test_owned_invalid_status_is_retained_for_parser_rejection(tmp_path):
    row = collect(tmp_path, 'owned-invalid')
    assert row['timestamp_source'] == 'counter_sample_set'
    assert len(row['rows']) == 1 and row['rows'][0]['status'] == 0xc0000bc6


def test_counter_clock_stays_identified_when_owned_process_exits_after_collection(tmp_path):
    row = collect(tmp_path, 'identity')
    assert row['timestamp_source'] == 'counter_sample_set'
    assert row['timestamp_100ns'] == row['rows'][0]['timestamp_100ns']
    assert row['error'] and 'identity' in row['error']


def test_whole_query_failure_has_no_counter_timestamp_or_fake_zero(tmp_path):
    row = collect(tmp_path, 'fatal')
    assert row['timestamp_source'] == 'failed_query_end' and row['error']
    assert row['timestamp_100ns'] is None and row['rows'] == []
