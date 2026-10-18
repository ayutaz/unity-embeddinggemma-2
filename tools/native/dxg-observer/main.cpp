#include <windows.h>
#include <evntrace.h>
#include <evntcons.h>
#include <DxTimingCaptureLibrary/DxTimingCaptureEventHandler.h>
#include <DxTimingCaptureLibrary/QpcTimestampConverter.h>
#include <DxTimingCaptureLibrary/EtwProviders.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include "NativeCallbacks.h"
#include "OwnedTraceLease.h"
#include "CaptureKeywords.h"
#include "CaptureConfiguration.h"

using namespace DirectX::Etw;
using namespace embeddinggemma::diagnostics;
namespace {
constexpr wchar_t SessionName[]=L"EmbeddingGemmaNativeDxgCapture";
struct Handle {
    HANDLE value=INVALID_HANDLE_VALUE;
    explicit Handle(HANDLE h):value(h) {}
    ~Handle(){if(value && value!=INVALID_HANDLE_VALUE) CloseHandle(value);}
    Handle(const Handle&)=delete;
};
struct TraceProperties {
    EVENT_TRACE_PROPERTIES value{};
    wchar_t name[256]{};
    TraceProperties() {
        value.Wnode.BufferSize=sizeof(*this); value.Wnode.Flags=WNODE_FLAG_TRACED_GUID; value.Wnode.ClientContext=1;
        value.LogFileMode=EVENT_TRACE_REAL_TIME_MODE|EVENT_TRACE_NO_PER_PROCESSOR_BUFFERING;
        value.BufferSize=CaptureConfiguration::BufferSizeKiB;
        value.MinimumBuffers=CaptureConfiguration::MinimumBuffers;
        value.MaximumBuffers=CaptureConfiguration::MaximumBuffers;
        value.LoggerNameOffset=offsetof(TraceProperties,name);
    }
};
uint64_t FileTimeValue(FILETIME t){return (uint64_t(t.dwHighDateTime)<<32)|t.dwLowDateTime;}
std::string Anchor() {
    LARGE_INTEGER before{},after{},frequency{}; FILETIME utc{};
    if(!QueryPerformanceFrequency(&frequency) || !QueryPerformanceCounter(&before)) throw std::runtime_error("QPC query failed");
    GetSystemTimePreciseAsFileTime(&utc);
    if(!QueryPerformanceCounter(&after)) throw std::runtime_error("QPC query failed");
    std::ostringstream out;
    out<<"\"qpc_before\":"<<before.QuadPart<<",\"qpc_after\":"<<after.QuadPart
       <<",\"filetime_utc_100ns\":"<<FileTimeValue(utc)<<",\"qpc_frequency\":"<<frequency.QuadPart;
    return out.str();
}
bool Number(const wchar_t* text,uint64_t maximum,uint64_t& value) {
    value=0;if(!text || !*text) return false;
    for(auto p=text;*p;++p){if(*p<L'0'||*p>L'9')return false;auto digit=uint64_t(*p-L'0');if(value>(maximum-digit)/10)return false;value=value*10+digit;}
    return value>0 && value<=maximum;
}
void Write(HANDLE output,const std::string& text) {
    size_t offset=0;
    while(offset<text.size()) {
        DWORD written=0; DWORD size=static_cast<DWORD>((std::min)(size_t(1048576),text.size()-offset));
        if(!WriteFile(output,text.data()+offset,size,&written,nullptr) || !written) throw std::runtime_error("Output write failed");
        offset+=written;
    }
}
struct Consumer {
    DxTimingCaptureEventHandler* handler;
    OwnedGpuRecords* records;
    std::atomic<bool> opened=false;
    std::atomic<ULONG> openStatus=ERROR_NOT_READY;
    ULONG processStatus=ERROR_NOT_READY,closeStatus=ERROR_NOT_READY;
    size_t decodeFailures=0;
    static void WINAPI Event(EVENT_RECORD* event) {
        auto& c=*static_cast<Consumer*>(event->UserContext);
        try { c.handler->HandleEventRecord(event); } catch(...) { ++c.decodeFailures;c.records->Diagnostic(); }
    }
    static ULONG WINAPI Buffer(EVENT_TRACE_LOGFILEW* trace) {
        auto& c=*static_cast<Consumer*>(trace->Context);
        try {c.handler->ReportTraceStatistics(*trace);}catch(...){++c.decodeFailures;c.records->Diagnostic();}
        return TRUE;
    }
    void Run() {
        EVENT_TRACE_LOGFILEW trace{}; trace.LoggerName=const_cast<wchar_t*>(SessionName);trace.Context=this;
        trace.ProcessTraceMode=PROCESS_TRACE_MODE_REAL_TIME|PROCESS_TRACE_MODE_EVENT_RECORD|PROCESS_TRACE_MODE_RAW_TIMESTAMP;
        trace.EventRecordCallback=Event;trace.BufferCallback=Buffer;
        auto handle=OpenTraceW(&trace);
        openStatus=handle==INVALID_PROCESSTRACE_HANDLE ? GetLastError() : ERROR_SUCCESS;
        opened=true;
        if(openStatus!=ERROR_SUCCESS)return;
        processStatus=ProcessTrace(&handle,1,nullptr,nullptr);
        try {handler->OnDataComplete();handler->ReportTraceStatistics(trace);}catch(...){++decodeFailures;records->Diagnostic();}
        closeStatus=CloseTrace(handle);
    }
};
ULONG Enable(TRACEHANDLE handle,const GUID& provider,ULONGLONG keywords,UCHAR level) {
    ENABLE_TRACE_PARAMETERS parameters{};parameters.Version=ENABLE_TRACE_PARAMETERS_VERSION_2;
    return EnableTraceEx2(handle,&provider,EVENT_CONTROL_CODE_ENABLE_PROVIDER,level,keywords,0,1000,&parameters);
}
ULONG Rundown(TRACEHANDLE handle,const GUID& provider) {
    return EnableTraceEx2(handle,&provider,EVENT_CONTROL_CODE_CAPTURE_STATE,TRACE_LEVEL_VERBOSE,0,0,1000,nullptr);
}
}

int wmain(int argc,wchar_t** argv) {
    if(argc==2 && std::wstring(argv[1])==L"--help") {
        std::cout<<"NativeDxgObserver PID seconds output.jsonl\nPID must be alive. Duration1..900 seconds. Output must not exist.\n"
            <<"At most200000 rows,4096 active executions/resources,64MiB requested ETW buffers. No model/GPU acceptance from exit alone.\n";
        return 0;
    }
    uint64_t pid=0,seconds=0;
    if(argc!=4 || !Number(argv[1],MAXDWORD,pid) || !Number(argv[2],900,seconds)) {
        std::cerr<<"Invalid PID/duration/arguments; use --help.\n";return 2;
    }
    Handle target(OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION|SYNCHRONIZE,FALSE,static_cast<DWORD>(pid)));
    FILETIME creation{},exit{},kernel{},user{};
    if(!target.value || !GetProcessTimes(target.value,&creation,&exit,&kernel,&user) || WaitForSingleObject(target.value,0)!=WAIT_TIMEOUT) {
        std::cerr<<"Target PID missing, inaccessible, or no longer alive.\n";return 2;
    }
    Handle output(CreateFileW(argv[3],GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr));
    if(output.value==INVALID_HANDLE_VALUE){std::cerr<<"Cannot create exclusive output; existing observations are preserved.\n";return 2;}
    OwnedGpuRecords records(static_cast<uint32_t>(pid));
    GpuCallbacks gpu(records);PacketCallbacks packets(records);ResourceCallbacks resources(records);CaptureDiagnostics diagnostics(records);
    TraceProperties properties;TRACEHANDLE traceHandle=0;
    OwnedTraceLease lease([&](uint64_t h){return ControlTraceW(h,SessionName,&properties.value,EVENT_TRACE_CONTROL_STOP);});
    std::unique_ptr<DxTimingCaptureEventHandler> handler;
    std::unique_ptr<Consumer> consumer;
    std::thread thread;
    ULONG startStatus=ERROR_NOT_READY,d3dStatus=ERROR_NOT_READY,dxgStatus=ERROR_NOT_READY,d3dRundown=ERROR_NOT_READY,dxgRundown=ERROR_NOT_READY;
    bool targetExited=false,operationalError=false;
    try {
        Write(output.value,"{\"kind\":\"capture_begin\",\"schema\":1,\"pid\":"+std::to_string(pid)
            +",\"target_creation_filetime\":"+std::to_string(FileTimeValue(creation))
            +",\"timestamp_unit\":\"qpc_epoch_ns\",\"session_name\":\"EmbeddingGemmaNativeDxgCapture\","+Anchor()+"}\n");
        DxTimingCaptureLibraryOptions options{};options.TrackApiObjects=true;options.TrackGpuTiming=true;options.TrackGpuEngineActivity=true;
        DxTimingCaptureEventCallbacks callbacks{};callbacks.ApiObjectCallbacks=&resources;callbacks.GpuTimingsCallbacks=&gpu;
        callbacks.GpuEngineActivityCallbacks=&packets;callbacks.DiagnosticsSink=&diagnostics;
        handler=DxTimingCaptureEventHandler::Create(static_cast<DWORD>(pid),options,callbacks);
        startStatus=StartTraceW(&traceHandle,SessionName,&properties.value);lease.Started(startStatus,traceHandle);
        if(startStatus!=ERROR_SUCCESS)throw std::runtime_error("StartTraceW failed");
        const ULONGLONG d3dKeywords=D3D12CaptureKeywords();
        const ULONGLONG dxgKeywords=DXGK_KEYWORD_LOG_FLAGS_LONG_HAUL|DXGK_KEYWORD_LOG_FLAGS_BASE|
            DXGK_KEYWORD_LOG_FLAGS_PROFILER|DXGK_KEYWORD_LOG_FLAGS_HISTORY_BUFFER|DXGK_KEYWORD_LOG_FLAGS_RESOURCE|
            DXGK_KEYWORD_LOG_FLAGS_ALLOCATIONS_REFERENCES;
        d3dStatus=Enable(traceHandle,Direct3D12EtwProviderGuid,d3dKeywords,TRACE_LEVEL_RESERVED6);
        dxgStatus=Enable(traceHandle,DxgkControlGuid,dxgKeywords,TRACE_LEVEL_VERBOSE);
        if(d3dStatus!=ERROR_SUCCESS || dxgStatus!=ERROR_SUCCESS)throw std::runtime_error("EnableTraceEx2 failed");
        consumer=std::make_unique<Consumer>();consumer->handler=handler.get();consumer->records=&records;
        thread=std::thread([&]{consumer->Run();});
        auto until=std::chrono::steady_clock::now()+std::chrono::seconds(5);
        while(!consumer->opened && std::chrono::steady_clock::now()<until)Sleep(10);
        if(!consumer->opened || consumer->openStatus!=ERROR_SUCCESS)throw std::runtime_error("OpenTrace did not open");
        d3dRundown=Rundown(traceHandle,Direct3D12EtwProviderGuid);dxgRundown=Rundown(traceHandle,DxgkControlGuid);
        if(d3dRundown!=ERROR_SUCCESS || dxgRundown!=ERROR_SUCCESS)throw std::runtime_error("Provider rundown failed");
        std::cout<<"{\"capture_ready\":true,\"pid\":"<<pid<<",\"session_name\":\"EmbeddingGemmaNativeDxgCapture\"}"<<std::endl;
        auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(seconds);
        while(std::chrono::steady_clock::now()<deadline) {
            auto wait=WaitForSingleObject(target.value,20);
            if(wait==WAIT_OBJECT_0){targetExited=true;Sleep(500);break;}
            if(wait==WAIT_FAILED)throw std::runtime_error("Target handle wait failed");
        }
    }catch(const std::exception& error){operationalError=true;std::cerr<<error.what()<<"\n";}
    catch(...){operationalError=true;std::cerr<<"Capture setup/processing exception\n";}
    lease.Close();if(lease.owned)lease.Close();
    if(thread.joinable())thread.join();
    bool valid=!operationalError && lease.stopped && !lease.hadStopFailure && records.Valid()
        && properties.value.EventsLost==0 && properties.value.LogBuffersLost==0 && properties.value.RealTimeBuffersLost==0
        && consumer && consumer->openStatus==ERROR_SUCCESS && consumer->processStatus==ERROR_SUCCESS
        && consumer->closeStatus==ERROR_SUCCESS && consumer->decodeFailures==0;
    try {
        std::ostringstream rows;records.WriteJsonl(rows);Write(output.value,rows.str());
        std::ostringstream footer;
        footer<<"{\"kind\":\"capture_end\",\"schema\":1,\"pid\":"<<pid
            <<",\"timestamp_unit\":\"qpc_epoch_ns\",\"session_name\":\"EmbeddingGemmaNativeDxgCapture\","
            <<Anchor()<<",\"session_stopped\":"<<(lease.stopped?"true":"false")
            <<",\"start_status\":"<<startStatus<<",\"d3d_enable_status\":"<<d3dStatus<<",\"dxg_enable_status\":"<<dxgStatus
            <<",\"d3d_keywords\":"<<D3D12CaptureKeywords()<<",\"shader_compilation_statistics_collected\":false"
            <<",\"requested_buffer_size_kib\":"<<CaptureConfiguration::BufferSizeKiB
            <<",\"requested_minimum_buffers\":"<<CaptureConfiguration::MinimumBuffers
            <<",\"requested_maximum_buffers\":"<<CaptureConfiguration::MaximumBuffers
            <<",\"d3d_rundown_status\":"<<d3dRundown<<",\"dxg_rundown_status\":"<<dxgRundown<<",\"stop_status\":"<<lease.lastStopStatus
            <<",\"stop_had_failure\":"<<(lease.hadStopFailure?"true":"false")
            <<",\"process_trace_status\":"<<(consumer?consumer->processStatus:ERROR_NOT_READY)
            <<",\"close_trace_status\":"<<(consumer?consumer->closeStatus:ERROR_NOT_READY)
            <<",\"decode_failures\":"<<(consumer?consumer->decodeFailures:0)
            <<",\"events_lost\":"<<properties.value.EventsLost<<",\"buffers_lost\":"<<(properties.value.LogBuffersLost+properties.value.RealTimeBuffersLost)
            <<",\"invalid_records\":"<<records.invalidRecords<<",\"dropped_records\":"<<records.droppedRecords
            <<",\"diagnostics\":"<<records.diagnostics<<",\"open_owned_executions\":"<<records.OpenOwnedExecutions()
            <<",\"open_resources\":"<<records.TrackedResources()<<",\"record_count\":"<<records.rows.size()
            <<",\"target_exited\":"<<(targetExited?"true":"false")<<",\"capture_valid\":"<<(valid?"true":"false")
            <<",\"real_model_validated\":false,\"gpu_control_validated\":false,\"diagnostic_codes\":[";
        for(size_t i=0;i<diagnostics.codes.size();++i){if(i)footer<<",";footer<<"["<<diagnostics.codes[i].first<<","<<diagnostics.codes[i].second<<"]";}
        footer<<"]}\n";Write(output.value,footer.str());
        if(!FlushFileBuffers(output.value))valid=false;
    }catch(...){valid=false;std::cerr<<"Cannot finish observation receipt\n";}
    return valid?0:2;
}
