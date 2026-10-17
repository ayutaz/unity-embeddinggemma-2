param([Parameter(Mandatory=$true)][string]$Receipt,[Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference='Stop'
$receiptFile=[IO.Path]::GetFullPath($Receipt)
$outputFile=[IO.Path]::GetFullPath($Output)
if(Test-Path -LiteralPath $outputFile){throw 'Preserve existing GPU engine observations.'}
$deadline=[DateTime]::UtcNow.AddMinutes(25)
while([DateTime]::UtcNow -lt $deadline){
    if(-not(Test-Path -LiteralPath $receiptFile)){Start-Sleep -Seconds 1;continue}
    try{$receiptData=Get-Content -LiteralPath $receiptFile -Raw|ConvertFrom-Json}catch{Start-Sleep -Seconds 1;continue}
    if($receiptData.terminal){break}
    if(-not $receiptData.pid){Start-Sleep -Seconds 1;continue}
    $ownedPid=[int]$receiptData.pid
    $process=Get-CimInstance Win32_Process -Filter "ProcessId = $ownedPid"
    if(-not $process -or $process.Name -ne 'Unity.exe' -or $process.CommandLine -notlike '*unity-embeddinggemma-2*'){break}
    $created=$process.CreationDate.ToUniversalTime().ToString('o')
    $observation=[ordered]@{pid=$ownedPid;process_created_utc=$created;request_begin_utc=[DateTime]::UtcNow.ToString('o');
        timestamp_source=$null;timestamp_100ns=$null;utc=$null;rows=@();error=$null;
        scope='Raw Timer100Ns numerator per exact process/adapter/engine. Not exclusive SDK time, not a sum of overlapping engines.'}
    try{
        $set=Get-Counter -Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction Stop
        $observation.timestamp_100ns=$set.Timestamp.ToUniversalTime().ToFileTimeUtc()
        $observation.utc=$set.Timestamp.ToUniversalTime().ToString('o')
        $observation.timestamp_source='counter_sample_set'
        $samples=@($set.CounterSamples|Where-Object{$_.InstanceName -like "pid_${ownedPid}_luid_*"})
        $observation.rows=@($samples|ForEach-Object{[ordered]@{name=$_.InstanceName;counter='utilization percentage';
            counter_type=$_.CounterType.ToString();time_base=$_.TimeBase;raw_value=$_.RawValue;
            timestamp_100ns=$_.Timestamp100NSec;status=$_.Status}})
        $after=Get-CimInstance Win32_Process -Filter "ProcessId = $ownedPid"
        if(-not $after -or $after.CreationDate.ToUniversalTime().ToString('o') -ne $created){
            $observation.timestamp_source='invalid_process_identity';$observation.error='Process exited or identity changed during counter query; do not attribute samples.'
        }
    }catch{
        $observation.error=$_.Exception.Message
        $observation.timestamp_source='failed_query_end'
        $observation.utc=[DateTime]::UtcNow.ToString('o')
    }
    $observation|ConvertTo-Json -Depth 5 -Compress|Add-Content -LiteralPath $outputFile -Encoding utf8
    Start-Sleep -Milliseconds 250
}
