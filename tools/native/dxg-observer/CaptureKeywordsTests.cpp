#include "CaptureKeywords.h"
#include "CaptureConfiguration.h"
#include <gtest/gtest.h>
using namespace embeddinggemma::diagnostics;
namespace {
bool Enabled(const EVENT_DESCRIPTOR& event) {
    // Bits48..63 are ETW channel keywords, not this provider's enable categories.
    return (event.Keyword & 0x0000ffffffffffffULL & D3D12CaptureKeywords()) != 0;
}
}
TEST(CaptureKeywords, DoesNotRequestUnusedPsoCompilationStatistics) {
    EXPECT_EQ(D3D12CaptureKeywords() & D3D12_ETW_LOG_APIS, 0u);
}
TEST(CaptureKeywords, PreservesGpuSubmissionExecutionAndHistoryEvents) {
    EXPECT_TRUE(Enabled(EventD3D12CommandBufferSubmission));
    EXPECT_TRUE(Enabled(EventD3D12StartExecuteCommandList));
    EXPECT_TRUE(Enabled(EventD3D12StopExecuteCommandList));
    EXPECT_TRUE(Enabled(EventD3D12StartExecuteCommandLists));
    EXPECT_TRUE(Enabled(EventD3D12StopExecuteCommandLists));
    EXPECT_TRUE(Enabled(EventD3D12HistoryBufferCompletion));
}
TEST(CaptureKeywords, PreservesResourceCreationDestructionAndRundown) {
    EXPECT_TRUE(Enabled(EventD3D12CreateResource));
    EXPECT_TRUE(Enabled(EventD3D12DestroyResource));
    EXPECT_TRUE(Enabled(EventD3D12ReportResource));
    EXPECT_TRUE(Enabled(EventD3D12CreateHeap));
    EXPECT_TRUE(Enabled(EventD3D12DestroyHeap));
    EXPECT_TRUE(Enabled(EventD3D12ReportHeap));
}
TEST(CaptureKeywords, KeepsAllThreeCompilationFamiliesOutsideDeclaredCaptureScope) {
    for(auto event : {EventD3D12BeginCreatePipelineStateObject, EventD3D12EndCreatePipelineStateObject,
        EventD3D12BeginCreateStateObject, EventD3D12EndCreateStateObject,
        EventD3D12BeginAddToStateObject, EventD3D12EndAddToStateObject,
        EventD3D12CreatePipelineStateObjectCacheStatistics, EventD3D12CreateStateObjectCacheStatistics,
        EventD3D12AddToStateObjectCacheStatistics}) EXPECT_FALSE(Enabled(event)) << event.Id;
}
TEST(CaptureKeywords, PreservesObjectIdentityAndLifetimeCategories) {
    EXPECT_EQ(D3D12CaptureKeywords() & D3D12_ETW_LOG_FLAGS_NAMES, D3D12_ETW_LOG_FLAGS_NAMES);
    EXPECT_EQ(D3D12CaptureKeywords() & D3D12_ETW_LOG_FLAGS_DEVICES, D3D12_ETW_LOG_FLAGS_DEVICES);
    EXPECT_EQ(D3D12CaptureKeywords() & D3D12_ETW_LOG_FLAGS_OBJECT_LIFETIME, D3D12_ETW_LOG_FLAGS_OBJECT_LIFETIME);
    EXPECT_EQ(D3D12CaptureKeywords() & D3D12_ETW_LOG_FLAGS_RESOURCES, D3D12_ETW_LOG_FLAGS_RESOURCES);
}
TEST(CaptureConfiguration, BufferCanHoldLargeHistoryEventsPlusTraceHeaders) {
    EXPECT_GE(CaptureConfiguration::BufferSizeKiB,256u);
}
TEST(CaptureConfiguration, PreallocatesAtLeastFourMiBForStartupBursts) {
    EXPECT_GE(uint64_t(CaptureConfiguration::MinimumBuffers)*CaptureConfiguration::BufferSizeKiB*1024,4u*1024*1024);
}
TEST(CaptureConfiguration, RequestedBuffersRemainFiniteAndAtMostSixtyFourMiB) {
    EXPECT_GT(CaptureConfiguration::MinimumBuffers,0u);
    EXPECT_GE(CaptureConfiguration::MaximumBuffers,CaptureConfiguration::MinimumBuffers);
    EXPECT_LE(uint64_t(CaptureConfiguration::MaximumBuffers)*CaptureConfiguration::BufferSizeKiB*1024,64u*1024*1024);
}
