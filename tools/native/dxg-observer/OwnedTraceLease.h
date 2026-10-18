#pragma once
#include <cstdint>
#include <functional>
#include <stdexcept>
namespace embeddinggemma::diagnostics {
class OwnedTraceLease {
    std::function<uint32_t(uint64_t)> stop;
    uint64_t handle=0;
public:
    bool owned=false,stopped=false,hadStopFailure=false;
    uint32_t lastStopStatus=0;
    explicit OwnedTraceLease(std::function<uint32_t(uint64_t)> callback):stop(std::move(callback)) {}
    OwnedTraceLease(const OwnedTraceLease&)=delete;
    OwnedTraceLease(OwnedTraceLease&&)=delete;
    OwnedTraceLease& operator=(const OwnedTraceLease&)=delete;
    OwnedTraceLease& operator=(OwnedTraceLease&&)=delete;
    ~OwnedTraceLease() { if(owned) Close(); }
    void Started(uint32_t status,uint64_t value) {
        if(owned) throw std::logic_error("Cannot replace an owned trace.");
        if(status==0) { handle=value; owned=true; stopped=false; }
    }
    uint32_t Close() noexcept {
        if(!owned) return lastStopStatus;
        try { lastStopStatus=stop(handle); } catch(...) { lastStopStatus=UINT32_MAX; }
        if(lastStopStatus==0) { owned=false; stopped=true; } else hadStopFailure=true;
        return lastStopStatus;
    }
};
}
