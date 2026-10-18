#pragma once
#include <cstdint>
namespace embeddinggemma::diagnostics {
struct CaptureConfiguration {
    static constexpr uint32_t BufferSizeKiB=256;
    static constexpr uint32_t MinimumBuffers=16;
    static constexpr uint32_t MaximumBuffers=256;
};
}
