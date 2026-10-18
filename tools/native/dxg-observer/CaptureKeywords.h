#pragma once
#include <windows.h>
#include <D3D12Events.h>
#include <cstdint>
namespace embeddinggemma::diagnostics {
constexpr uint64_t D3D12CaptureKeywords() {
    return D3D12_ETW_LOG_FLAGS_RENDER_OPERATION_MARKERS | D3D12_ETW_LOG_FLAGS_DRIVER_CUSTOM_MARKERS |
        D3D12_ETW_LOG_FLAGS_BUNDLE_MARKERS | D3D12_ETW_LOG_FLAGS_NAMES | D3D12_ETW_LOG_FLAGS_DEVICES |
        D3D12_ETW_LOG_FLAGS_OBJECT_LIFETIME | D3D12_ETW_LOG_FLAGS_RESOURCES;
    // This observer consumes GPU work and resource lifetimes, not PSO compilation
    // cache statistics. LOG_APIS enables that additional decoder path; it is not
    // needed by the submission/history/resource events verified in the tests.
}
}
