#include "NativeCallbacks.h"
#include <gtest/gtest.h>
using namespace embeddinggemma::diagnostics;
using namespace DirectX::Etw;
TEST(NativeCallbacks, NativeGpuWorkPreservesProcessQueueMarkerAndRawEpoch) {
    OwnedGpuRecords r(42); GpuCallbacks g(r); UINT64 id=0; constexpr INT64 t=9007199254740993LL;
    ASSERT_EQ(g.OnGpuExecutionBegin(42,13,7,t,0,&id),S_OK);
    ASSERT_EQ(g.OnGpuWork(9,id,5,83,t+5,t+19),S_OK); ASSERT_EQ(g.OnGpuExecutionComplete(id),S_OK);
    ASSERT_EQ(r.rows.size(),3u); EXPECT_NE(id,0u); EXPECT_EQ(r.rows[1].beginNs,t+5); EXPECT_EQ(r.rows[1].endNs,t+19);
    EXPECT_EQ(r.rows[1].pid,42u); EXPECT_EQ(r.rows[1].apiQueue,7u); EXPECT_EQ(r.rows[1].hardwareQueue,9u);
    EXPECT_EQ(r.rows[1].commandList,5u); EXPECT_EQ(r.rows[1].marker,83u); EXPECT_TRUE(r.Valid());
}
TEST(NativeCallbacks, NativeForeignWorkDoesNotAcquireTargetIdentity) {
    OwnedGpuRecords r(42); GpuCallbacks g(r); UINT64 id=0;
    g.OnGpuExecutionBegin(999,13,7,100,0,&id); g.OnGpuWork(9,id,0,0,101,105); g.OnGpuExecutionComplete(id);
    EXPECT_TRUE(r.rows.empty()); EXPECT_TRUE(r.Valid());
}
TEST(NativeCallbacks, PacketCallbackRemainsAnEnvelopeWithTargetFiltering) {
    OwnedGpuRecords r(42); PacketCallbacks g(r);
    g.OnGpuEngineActivity(999,9,0,100,200,true); g.OnGpuEngineActivity(42,9,0,110,190,true);
    ASSERT_EQ(r.rows.size(),1u); EXPECT_EQ(r.rows[0].kind,RecordKind::PacketEnvelope);
    EXPECT_EQ(r.rows[0].beginNs,110); EXPECT_EQ(r.rows[0].endNs,190);
}
TEST(NativeCallbacks, ResourceCallbackIdsAreStableAlongsideOtherObjectTypes) {
    OwnedGpuRecords r(42); ResourceCallbacks g(r); UINT64 device=0,resource=0;
    D3D12_RESOURCE_DESC desc{}; desc.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; desc.Width=8388608;
    g.OnDeviceCreation(90,nullptr,&device);
    g.OnCommittedResourceCreation(100,device,42,1,&desc,nullptr,nullptr,D3D12_HEAP_FLAG_NONE,&resource);
    g.OnObjectDestruction(200,ApiObjectType::Resource,resource);
    EXPECT_NE(device,0u); EXPECT_NE(resource,0u); EXPECT_NE(device,resource);
    ASSERT_EQ(r.rows.size(),2u); EXPECT_EQ(r.rows[0].bytes,8388608u);
    EXPECT_EQ(r.rows[1].object,resource); EXPECT_EQ(r.rows[1].beginNs,200); EXPECT_TRUE(r.Valid());
}
TEST(NativeCallbacks, TextureWidthIsNotMisreportedAsAllocatedBytes) {
    OwnedGpuRecords r(42); ResourceCallbacks g(r); UINT64 id=0;
    D3D12_RESOURCE_DESC desc{}; desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D; desc.Width=160;
    g.OnPlacedResourceCreation(100,1,42,1,&desc,nullptr,&id);
    ASSERT_EQ(r.rows.size(),1u); EXPECT_FALSE(r.rows[0].bytes);
}
TEST(NativeCallbacks, ForeignResourceCallbacksNeverExportTheirPayload) {
    OwnedGpuRecords r(42); ResourceCallbacks g(r); UINT64 id=0;
    D3D12_RESOURCE_DESC desc{}; desc.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; desc.Width=123;
    g.OnReservedResourceCreation(100,1,999,1,&desc,nullptr,&id); g.OnObjectDestruction(200,ApiObjectType::Resource,id);
    EXPECT_TRUE(r.rows.empty()); EXPECT_TRUE(r.Valid());
}
TEST(NativeCallbacks, ProviderWarningsAreRetainedAsIncompleteData) {
    OwnedGpuRecords r(42); CaptureDiagnostics d(r);
    d.OnDiagnostic(DiagnosticSeverity::Warning,DiagnosticCode::CommandListExecutionsUnfinishedAtEndOfTrace,L"Unfinished");
    EXPECT_EQ(r.diagnostics,1u); EXPECT_FALSE(r.Valid());
}
TEST(NativeCallbacks, MissingExecutionIdOutputFailsWithoutDereference) {
    OwnedGpuRecords r(42); GpuCallbacks g(r);
    EXPECT_EQ(g.OnGpuExecutionBegin(42,1,7,100,0,nullptr),E_POINTER); EXPECT_FALSE(r.Valid());
}
TEST(NativeCallbacks, DiagnosticSeverityAndCodeSurviveWithoutMessagePayload) {
    OwnedGpuRecords r(42); CaptureDiagnostics d(r);
    d.OnDiagnostic(DiagnosticSeverity::Error,DiagnosticCode::EventsLost,L"Do not export this payload");
    ASSERT_EQ(d.codes.size(),1u); EXPECT_EQ(d.codes[0].first,static_cast<int>(DiagnosticSeverity::Error));
    EXPECT_EQ(d.codes[0].second,static_cast<int>(DiagnosticCode::EventsLost));
}
TEST(NativeCallbacks, DiagnosticCodeStorageIsBoundedButTotalLossIsCounted) {
    OwnedGpuRecords r(42); CaptureDiagnostics d(r);
    for(int i=0;i<100;++i) d.OnDiagnostic(DiagnosticSeverity::Error,DiagnosticCode::EventsLost,L"loss");
    EXPECT_EQ(d.codes.size(),64u); EXPECT_EQ(r.diagnostics,100u); EXPECT_FALSE(r.Valid());
}
TEST(NativeCallbacks, PacketSchedulingMetadataIsNotLost) {
    OwnedGpuRecords r(42); PacketCallbacks p(r); p.OnGpuEngineActivity(42,9,4,100,200,true);
    ASSERT_EQ(r.rows.size(),1u); EXPECT_EQ(r.rows[0].engineAffinity,4u);
    ASSERT_TRUE(r.rows[0].hardwareScheduled.has_value()); EXPECT_TRUE(*r.rows[0].hardwareScheduled);
}
