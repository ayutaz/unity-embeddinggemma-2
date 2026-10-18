#pragma once
#include <cstdint>
#include <cstddef>
#include <ostream>
#include <optional>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

namespace embeddinggemma::diagnostics {
enum class RecordKind { ExecutionBegin, GpuWork, ExecutionComplete, PacketEnvelope, ResourceCreated, ResourceDestroyed };
struct Record {
    RecordKind kind{};
    uint32_t pid{};
    uint64_t execution{}, apiQueue{}, hardwareQueue{}, object{}, marker{};
    uint32_t commandList{};
    uint32_t engineAffinity{};
    std::optional<bool> hardwareScheduled;
    std::optional<int64_t> beginNs, endNs;
    std::optional<uint64_t> bytes;
};
class OwnedGpuRecords {
    struct Execution { bool owned; uint64_t apiQueue; };
    struct ResourceInfo { int64_t created; std::optional<uint64_t> bytes; };
    uint32_t target;
    size_t maximumRows, maximumActive;
    uint64_t nextExecution=0;
    std::unordered_map<uint64_t,Execution> executions;
    std::unordered_map<uint64_t,ResourceInfo> resources;
    void Append(Record row) {
        if(rows.size()>=maximumRows) { ++droppedRecords; return; }
        row.pid=target;
        rows.push_back(row);
    }
    bool Interval(int64_t begin,int64_t end) {
        if(begin<0 || end<begin) { ++invalidRecords; return false; }
        return true;
    }
public:
    std::vector<Record> rows;
    size_t invalidRecords=0, droppedRecords=0, diagnostics=0;
    OwnedGpuRecords(uint32_t pid,size_t rowCap=200000,size_t activeCap=4096)
        :target(pid),maximumRows(rowCap),maximumActive(activeCap) {
        if(!pid || !rowCap || !activeCap) throw std::invalid_argument("Nonzero target PID and finite positive caps required.");
    }
    uint64_t Begin(uint32_t pid,uint64_t queue,int64_t timestamp) {
        auto id=++nextExecution;
        if(executions.size()>=maximumActive) { ++droppedRecords; return id; }
        bool owned=pid==target;
        executions.emplace(id,Execution{owned,owned?queue:0});
        if(owned && Interval(timestamp,timestamp)) {
            Record row{RecordKind::ExecutionBegin}; row.execution=id; row.apiQueue=queue; row.beginNs=timestamp;
            Append(row);
        }
        return id;
    }
    void Work(uint64_t hardwareQueue,uint64_t execution,int64_t begin,int64_t end,uint32_t commandList=0,uint64_t marker=0) {
        auto it=executions.find(execution);
        if(it==executions.end()) { ++invalidRecords; return; }
        if(!it->second.owned || !Interval(begin,end)) return;
        Record row{RecordKind::GpuWork}; row.execution=execution; row.apiQueue=it->second.apiQueue;
        row.hardwareQueue=hardwareQueue; row.beginNs=begin; row.endNs=end;
        row.commandList=commandList; row.marker=marker; Append(row);
    }
    void Complete(uint64_t execution) {
        auto it=executions.find(execution);
        if(it==executions.end()) { ++invalidRecords; return; }
        if(it->second.owned) {
            Record row{RecordKind::ExecutionComplete}; row.execution=execution; row.apiQueue=it->second.apiQueue;
            Append(row);
        }
        executions.erase(it);
    }
    void Packet(uint32_t pid,uint64_t queue,int64_t begin,int64_t end,uint32_t affinity=0,std::optional<bool> scheduled=std::nullopt) {
        if(pid!=target || !Interval(begin,end)) return;
        Record row{RecordKind::PacketEnvelope}; row.hardwareQueue=queue; row.beginNs=begin; row.endNs=end;
        row.engineAffinity=affinity;row.hardwareScheduled=scheduled;Append(row);
    }
    void Resource(uint32_t pid,uint64_t object,int64_t timestamp,std::optional<uint64_t> bytes) {
        if(pid!=target) return;
        if(!object || resources.contains(object)) { ++invalidRecords; return; }
        if(!Interval(timestamp,timestamp)) return;
        if(resources.size()>=maximumActive) { ++droppedRecords; return; }
        resources.emplace(object,ResourceInfo{timestamp,bytes});
        Record row{RecordKind::ResourceCreated}; row.object=object; row.beginNs=timestamp; row.bytes=bytes; Append(row);
    }
    void Destroy(uint64_t object,int64_t timestamp) {
        auto it=resources.find(object);
        // The library reports other object types too; an untracked destruction is not owned resource data.
        if(it==resources.end()) return;
        if(!Interval(it->second.created,timestamp)) return;
        Record row{RecordKind::ResourceDestroyed}; row.object=object; row.beginNs=timestamp; row.bytes=it->second.bytes;
        Append(row); resources.erase(it);
    }
    void Diagnostic() { ++diagnostics; }
    bool Valid() const { return !invalidRecords && !droppedRecords && !diagnostics && OpenOwnedExecutions()==0; }
    size_t OpenOwnedExecutions() const {
        size_t count=0; for(auto& [id,execution]:executions) if(execution.owned) ++count; return count;
    }
    size_t TrackedExecutions() const { return executions.size(); }
    size_t TrackedResources() const { return resources.size(); }
    void WriteJsonl(std::ostream& out) const {
        static constexpr const char* names[]={"execution_begin","gpu_work","execution_complete","packet_envelope","resource_created","resource_destroyed"};
        for(auto& row:rows) {
            out<<"{\"kind\":\""<<names[static_cast<size_t>(row.kind)]<<"\",\"pid\":"<<row.pid
                <<",\"execution\":"<<row.execution<<",\"api_queue\":"<<row.apiQueue<<",\"hardware_queue\":"<<row.hardwareQueue
                <<",\"object\":"<<row.object<<",\"command_list_index\":"<<row.commandList<<",\"api_marker\":"<<row.marker<<",\"begin_ns\":";
            if(row.beginNs) out<<*row.beginNs; else out<<"null";
            out<<",\"end_ns\":"; if(row.endNs) out<<*row.endNs; else out<<"null";
            out<<",\"bytes\":"; if(row.bytes) out<<*row.bytes; else out<<"null";
            out<<",\"engine_affinity\":"<<row.engineAffinity<<",\"hardware_scheduled\":";
            if(row.hardwareScheduled) out<<(*row.hardwareScheduled?"true":"false"); else out<<"null";
            out<<"}\n";
        }
    }
};
}
