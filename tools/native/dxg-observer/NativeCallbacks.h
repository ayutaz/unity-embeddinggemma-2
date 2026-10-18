#pragma once
#include <windows.h>
#include <DxTimingCaptureLibrary/NoOpCallbacks.h>
#include <DxTimingCaptureLibrary/EtwDiagnostics.h>
#include "OwnedGpuRecords.h"
namespace embeddinggemma::diagnostics {
class GpuCallbacks : public DirectX::Etw::NoOpGpuTimingsCallbacks {
    OwnedGpuRecords& records;
public:
    explicit GpuCallbacks(OwnedGpuRecords& r):records(r) {}
    HRESULT OnGpuExecutionBegin(UINT32 pid,UINT32,UINT64 queue,INT64 ts,UINT64,UINT64* id) override {
        if(!id) { records.Diagnostic(); return E_POINTER; }
        *id=records.Begin(pid,queue,ts); return S_OK;
    }
    HRESULT OnGpuExecutionComplete(UINT64 id) override { records.Complete(id); return S_OK; }
    HRESULT OnGpuWork(UINT64 queue,UINT64 execution,UINT32 list,UINT64 marker,INT64 begin,INT64 end) override {
        records.Work(queue,execution,begin,end,list,marker); return S_OK;
    }
};
class PacketCallbacks : public DirectX::Etw::NoOpGpuEngineActivityCallbacks {
    OwnedGpuRecords& records;
public:
    explicit PacketCallbacks(OwnedGpuRecords& r):records(r) {}
    HRESULT OnGpuEngineActivity(UINT32 pid,UINT64 queue,UINT32 affinity,INT64 begin,INT64 end,bool scheduled) override {
        records.Packet(pid,queue,begin,end,affinity,scheduled); return S_OK;
    }
};
class ResourceCallbacks : public DirectX::Etw::NoOpApiObjectCallbacks {
    OwnedGpuRecords& records;
    void Remember(UINT32 pid,UINT64 id,INT64 ts,const D3D12_RESOURCE_DESC* desc) {
        std::optional<uint64_t> bytes;
        // Width is a logical byte count for buffers; texture Width is not allocation size.
        if(desc && desc->Dimension==D3D12_RESOURCE_DIMENSION_BUFFER) bytes=desc->Width;
        records.Resource(pid,id,ts,bytes);
    }
public:
    explicit ResourceCallbacks(OwnedGpuRecords& r):records(r) {}
    HRESULT OnCommittedResourceCreation(INT64 ts,UINT64 device,UINT32 pid,UINT32 tid,const D3D12_RESOURCE_DESC* desc,
        const DirectX::Etw::ObjectPlacementInfo* placement,const D3D12_HEAP_PROPERTIES* heap,D3D12_HEAP_FLAGS flags,UINT64* id) override {
        UINT64 local=0; auto output=id?id:&local;
        auto hr=NoOpApiObjectCallbacks::OnCommittedResourceCreation(ts,device,pid,tid,desc,placement,heap,flags,output);
        if(SUCCEEDED(hr)) Remember(pid,*output,ts,desc); return hr;
    }
    HRESULT OnPlacedResourceCreation(INT64 ts,UINT64 device,UINT32 pid,UINT32 tid,const D3D12_RESOURCE_DESC* desc,
        const DirectX::Etw::ObjectPlacementInfo* placement,UINT64* id) override {
        UINT64 local=0; auto output=id?id:&local;
        auto hr=NoOpApiObjectCallbacks::OnPlacedResourceCreation(ts,device,pid,tid,desc,placement,output);
        if(SUCCEEDED(hr)) Remember(pid,*output,ts,desc); return hr;
    }
    HRESULT OnReservedResourceCreation(INT64 ts,UINT64 device,UINT32 pid,UINT32 tid,const D3D12_RESOURCE_DESC* desc,
        const DirectX::Etw::ReservedResourceInfo* reserved,UINT64* id) override {
        UINT64 local=0; auto output=id?id:&local;
        auto hr=NoOpApiObjectCallbacks::OnReservedResourceCreation(ts,device,pid,tid,desc,reserved,output);
        if(SUCCEEDED(hr)) Remember(pid,*output,ts,desc); return hr;
    }
    HRESULT OnObjectDestruction(INT64 ts,DirectX::Etw::ApiObjectType type,UINT64 id) override {
        if(type==DirectX::Etw::ApiObjectType::Resource) records.Destroy(id,ts); return S_OK;
    }
};
class CaptureDiagnostics : public DirectX::Etw::DiagnosticsSink {
    OwnedGpuRecords& records;
public:
    std::vector<std::pair<int,int>> codes;
    explicit CaptureDiagnostics(OwnedGpuRecords& r):records(r) {}
    void OnDiagnostic(DirectX::Etw::DiagnosticSeverity severity,DirectX::Etw::DiagnosticCode code,std::wstring_view) override {
        records.Diagnostic();
        if(codes.size()<64) codes.emplace_back(static_cast<int>(severity),static_cast<int>(code));
    }
};
}
