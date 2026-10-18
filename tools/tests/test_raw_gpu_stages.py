import copy
import pytest
from embeddinggemma_tools.raw_gpu_stages import summarize_raw_gpu_stages

def fixture():
    header=dict(kind='capture_begin',pid=42,qpc_before=1,qpc_after=2,qpc_frequency=10000000,target_creation_filetime=100)
    footer=dict(kind='capture_end',pid=42,qpc_before=1000,qpc_after=1001,qpc_frequency=10000000,
        capture_valid=True,session_stopped=True,target_exited=True,record_count=4,
        events_lost=0,buffers_lost=0,decode_failures=0,invalid_records=0,dropped_records=0,diagnostics=0,
        open_owned_executions=0,start_status=0,d3d_enable_status=0,dxg_enable_status=0,
        d3d_rundown_status=0,dxg_rundown_status=0,stop_status=0,process_trace_status=0,close_trace_status=0,stop_had_failure=False)
    work=lambda queue,begin,end:dict(kind='gpu_work',pid=42,hardware_queue=queue,begin_ns=begin,end_ns=end)
    records=[header,work(7,10000,15000),work(7,12000,18000),work(9,17000,25000),
        dict(kind='packet_envelope',pid=42,hardware_queue=7,begin_ns=9000,end_ns=26000),footer]
    stages=[dict(id='load',timestamp_unit='qpc_ticks',clock_source='QueryPerformanceCounter',frequency_hz=10000000,begin_ticks=100,end_ticks=200)]
    bridge=lambda start:dict(validated=True,high_resolution=True,native_qpc_before=start,counter_ticks=start+1,
        native_qpc_after=start+2,frequency_hz=10000000,native_frequency_hz=10000000,counter_frequency_hz=10000000,clock_source='QueryPerformanceCounter')
    return records,stages,[bridge(10),bridge(900)]

def test_merges_overlapping_work_per_queue_and_retains_crossing_scope_without_engine_sum():
    records,stages,bridges=fixture();result=summarize_raw_gpu_stages(42,records,stages,bridges)
    assert result['raw_stage_correlation_accepted'] is True
    assert result['exclusive_sdk_gpu_time_measured'] is False
    row=result['stages'][0]
    assert row['id']=='load' and row['begin_ns']==10000 and row['end_ns']==20000
    assert row['gpu_work_records']==3 and row['boundary_crossing_records']==1
    assert row['queues']==[dict(hardware_queue=7,observed_union_ns=8000),dict(hardware_queue=9,observed_union_ns=3000)]
    assert 'summed_engine_ns' not in row and 'pure_kernel_ns' not in row

@pytest.mark.parametrize('field',['events_lost','buffers_lost','decode_failures','invalid_records','dropped_records','diagnostics','open_owned_executions','stop_status'])
def test_incomplete_observation_is_rejected_instead_of_zero_or_partial_time(field):
    records,stages,bridges=fixture();records[-1][field]=1
    with pytest.raises(ValueError):summarize_raw_gpu_stages(42,records,stages,bridges)

@pytest.mark.parametrize('mutation',['missing_footer','wrong_pid','missing_status','false_valid','false_stop','live_target','wrong_count'])
def test_missing_identity_or_lifetime_evidence_is_rejected(mutation):
    records,stages,bridges=fixture()
    if mutation=='missing_footer':records.pop()
    elif mutation=='wrong_pid':records[1]['pid']=99
    elif mutation=='missing_status':del records[-1]['events_lost']
    elif mutation=='false_valid':records[-1]['capture_valid']=False
    elif mutation=='false_stop':records[-1]['session_stopped']=False
    elif mutation=='live_target':records[-1]['target_exited']=False
    else:records[-1]['record_count']=3
    with pytest.raises(ValueError):summarize_raw_gpu_stages(42,records,stages,bridges)

@pytest.mark.parametrize('mutation',['frequency','coarse','epoch','missing_bridge','after_begin','before_end','wrong_clock'])
def test_runtime_native_clock_bridge_must_cover_the_stage(mutation):
    records,stages,bridges=fixture()
    if mutation=='frequency':bridges[0]['counter_frequency_hz']=1000
    elif mutation=='coarse':bridges[0]['high_resolution']=False
    elif mutation=='epoch':bridges[0]['counter_ticks']=999
    elif mutation=='missing_bridge':bridges.pop()
    elif mutation=='after_begin':bridges[0].update(native_qpc_before=110,counter_ticks=111,native_qpc_after=112)
    elif mutation=='wrong_clock':bridges[0]['clock_source']='Stopwatch'
    else:bridges[-1].update(native_qpc_before=180,counter_ticks=181,native_qpc_after=182)
    with pytest.raises(ValueError):summarize_raw_gpu_stages(42,records,stages,bridges)

@pytest.mark.parametrize('mutation',['wrong_frequency','backwards','outside_capture','float_ticks','duplicate_id','overlap','wrong_clock'])
def test_stage_endpoints_and_ids_are_strict(mutation):
    records,stages,bridges=fixture()
    if mutation=='wrong_frequency':stages[0]['frequency_hz']=1000
    elif mutation=='backwards':stages[0]['end_ticks']=99
    elif mutation=='outside_capture':stages[0]['end_ticks']=2000
    elif mutation=='float_ticks':stages[0]['begin_ticks']=100.0
    elif mutation=='duplicate_id':stages.append(copy.deepcopy(stages[0]))
    elif mutation=='wrong_clock':stages[0]['clock_source']='Stopwatch'
    else:stages.append(dict(id='query',timestamp_unit='qpc_ticks',frequency_hz=10000000,begin_ticks=199,end_ticks=300))
    with pytest.raises(ValueError):summarize_raw_gpu_stages(42,records,stages,bridges)

def test_no_recorded_work_is_explicit_observation_not_zero_sdk_gpu_time():
    records,stages,bridges=fixture();records=[records[0],records[-1]];records[-1]['record_count']=0
    result=summarize_raw_gpu_stages(42,records,stages,bridges)
    assert result['stages'][0]['queues']==[] and result['stages'][0]['gpu_work_records']==0
    assert result['exclusive_sdk_gpu_time_measured'] is False

def test_integer_epoch_is_not_rounded_through_float():
    records,stages,bridges=fixture();offset=9007199254740993
    for r in (records[0],records[-1]):
        r['qpc_before']+=offset;r['qpc_after']+=offset
    for r in records[1:-1]:r['begin_ns']+=offset*100;r['end_ns']+=offset*100
    for s in stages:s['begin_ticks']+=offset;s['end_ticks']+=offset
    for b in bridges:
        for key in ('native_qpc_before','counter_ticks','native_qpc_after'):b[key]+=offset
    row=summarize_raw_gpu_stages(42,records,stages,bridges)['stages'][0]
    assert row['begin_ns']==(offset+100)*100 and row['queues'][0]['observed_union_ns']==8000

def test_backward_gpu_work_cannot_be_silently_discarded():
    records,stages,bridges=fixture();records[1]['end_ns']=9999
    with pytest.raises(ValueError):summarize_raw_gpu_stages(42,records,stages,bridges)
