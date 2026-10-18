#include "OwnedGpuRecords.h"
#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>
using namespace embeddinggemma::diagnostics;
int main(int argc,char** argv) { testing::InitGoogleTest(&argc,argv); return RUN_ALL_TESTS(); }

TEST(OwnedGpuRecords, RejectsUnboundedOrAnonymousConfiguration) {
    EXPECT_THROW(OwnedGpuRecords(0),std::invalid_argument);
    EXPECT_THROW(OwnedGpuRecords(42,0,2),std::invalid_argument);
    EXPECT_THROW(OwnedGpuRecords(42,2,0),std::invalid_argument);
}
TEST(OwnedGpuRecords, PreservesQpcEpochNanosecondsWithoutFloatOrRebase) {
    OwnedGpuRecords r(42); constexpr int64_t t=9007199254740993LL;
    auto id=r.Begin(42,7,t); r.Work(9,id,t+13,t+97); r.Complete(id);
    ASSERT_EQ(r.rows.size(),3u);
    EXPECT_EQ(r.rows[0].beginNs,t); EXPECT_EQ(r.rows[1].beginNs,t+13); EXPECT_EQ(r.rows[1].endNs,t+97);
    EXPECT_EQ(r.rows[1].pid,42u); EXPECT_EQ(r.rows[1].apiQueue,7u); EXPECT_EQ(r.rows[1].hardwareQueue,9u);
    EXPECT_TRUE(r.Valid());
}
TEST(OwnedGpuRecords, OtherProcessesNeverBecomeOwnedRows) {
    OwnedGpuRecords r(42); auto other=r.Begin(999,7,1); r.Work(9,other,2,8); r.Complete(other);
    r.Packet(999,10,3,9); r.Resource(999,71,2,64); r.Destroy(71,20);
    EXPECT_TRUE(r.rows.empty()); EXPECT_TRUE(r.Valid());
}
TEST(OwnedGpuRecords, ExecutionIdsStayUniqueAndNonzeroAcrossProcesses) {
    OwnedGpuRecords r(42); auto a=r.Begin(42,7,1), b=r.Begin(999,7,2), c=r.Begin(42,8,3);
    EXPECT_NE(a,0u); EXPECT_NE(b,0u); EXPECT_NE(c,0u); EXPECT_NE(a,b); EXPECT_NE(b,c); EXPECT_NE(a,c);
}
TEST(OwnedGpuRecords, CompletionHasNoInventedGpuTimestamp) {
    OwnedGpuRecords r(42); auto id=r.Begin(42,7,1); r.Work(9,id,2,8); r.Complete(id);
    ASSERT_FALSE(r.rows.empty()); auto end=r.rows.back();
    EXPECT_EQ(end.kind,RecordKind::ExecutionComplete); EXPECT_FALSE(end.beginNs); EXPECT_FALSE(end.endNs);
    EXPECT_EQ(r.OpenOwnedExecutions(),0u); EXPECT_EQ(r.TrackedExecutions(),0u);
}
TEST(OwnedGpuRecords, UnknownExecutionMakesMissingAttributionVisible) {
    OwnedGpuRecords r(42); r.Work(9,999,2,8);
    EXPECT_EQ(r.invalidRecords,1u); EXPECT_FALSE(r.Valid()); EXPECT_TRUE(r.rows.empty());
}
TEST(OwnedGpuRecords, BackwardOrNegativeTimestampCannotPass) {
    OwnedGpuRecords r(42); auto id=r.Begin(42,7,1); r.Work(9,id,8,2); r.Work(9,id,-1,2); r.Complete(id);
    EXPECT_EQ(r.invalidRecords,2u); EXPECT_FALSE(r.Valid());
    for(auto& row:r.rows) EXPECT_NE(row.kind,RecordKind::GpuWork);
}
TEST(OwnedGpuRecords, ZeroDurationIsRetainedWithoutCallingItPositiveGpuWork) {
    OwnedGpuRecords r(42); auto id=r.Begin(42,7,1); r.Work(9,id,2,2); r.Complete(id);
    ASSERT_EQ(r.rows.size(),3u); EXPECT_EQ(r.rows[1].beginNs,r.rows[1].endNs);
}
TEST(OwnedGpuRecords, DriverPacketEnvelopeRemainsDistinctFromGpuWork) {
    OwnedGpuRecords r(42); r.Packet(42,9,100,200);
    ASSERT_EQ(r.rows.size(),1u); EXPECT_EQ(r.rows[0].kind,RecordKind::PacketEnvelope);
    EXPECT_EQ(r.rows[0].beginNs,100); EXPECT_EQ(r.rows[0].endNs,200);
}
TEST(OwnedGpuRecords, ResourceLifetimeKeepsRawEpochAndKnownBytes) {
    OwnedGpuRecords r(42); r.Resource(42,99,100,8388608); r.Destroy(99,300);
    ASSERT_EQ(r.rows.size(),2u); EXPECT_EQ(r.rows[0].kind,RecordKind::ResourceCreated);
    EXPECT_EQ(r.rows[0].beginNs,100); EXPECT_EQ(r.rows[0].bytes,8388608u);
    EXPECT_EQ(r.rows[1].kind,RecordKind::ResourceDestroyed); EXPECT_EQ(r.rows[1].beginNs,300);
    EXPECT_EQ(r.rows[1].object,99u); EXPECT_TRUE(r.Valid());
}
TEST(OwnedGpuRecords, UnknownResourceSizeIsNotZeroBytes) {
    OwnedGpuRecords r(42); r.Resource(42,99,100,std::nullopt);
    ASSERT_EQ(r.rows.size(),1u); EXPECT_FALSE(r.rows[0].bytes);
}
TEST(OwnedGpuRecords, DestructionBeforeCreationIsRejected) {
    OwnedGpuRecords r(42); r.Resource(42,99,100,64); r.Destroy(99,50);
    EXPECT_EQ(r.invalidRecords,1u); EXPECT_FALSE(r.Valid()); EXPECT_EQ(r.rows.size(),1u);
}
TEST(OwnedGpuRecords, EventCapIsABoundedFailureRatherThanASilentPass) {
    OwnedGpuRecords r(42,2,8); r.Packet(42,9,1,2); r.Packet(42,9,3,4); r.Packet(42,9,5,6);
    EXPECT_EQ(r.rows.size(),2u); EXPECT_EQ(r.droppedRecords,1u); EXPECT_FALSE(r.Valid());
}
TEST(OwnedGpuRecords, ActiveExecutionCapIncludesForeignTraffic) {
    OwnedGpuRecords r(42,50,2); auto a=r.Begin(999,9,1); auto b=r.Begin(999,10,2); r.Begin(42,11,3);
    EXPECT_EQ(r.TrackedExecutions(),2u); EXPECT_GT(r.droppedRecords,0u); EXPECT_FALSE(r.Valid());
    r.Complete(a); r.Complete(b); EXPECT_EQ(r.TrackedExecutions(),0u);
}
TEST(OwnedGpuRecords, UnfinishedOwnedExecutionCannotPass) {
    OwnedGpuRecords r(42); r.Begin(42,9,1); EXPECT_EQ(r.OpenOwnedExecutions(),1u); EXPECT_FALSE(r.Valid());
}
TEST(OwnedGpuRecords, ForeignExecutionStillOpenDoesNotInvalidateOwnedWork) {
    OwnedGpuRecords r(42); r.Begin(999,9,1); auto id=r.Begin(42,7,2); r.Work(9,id,3,8); r.Complete(id);
    EXPECT_EQ(r.OpenOwnedExecutions(),0u); EXPECT_TRUE(r.Valid()); ASSERT_EQ(r.rows.size(),3u);
}
TEST(OwnedGpuRecords, DiagnosticsMakeIncompleteDataExplicit) {
    OwnedGpuRecords r(42); r.Diagnostic(); EXPECT_EQ(r.diagnostics,1u); EXPECT_FALSE(r.Valid());
}
TEST(OwnedGpuRecords, JsonlKeepsInt64AndNullWithoutExportingOtherPids) {
    OwnedGpuRecords r(42); auto other=r.Begin(999,1,1); r.Complete(other);
    auto id=r.Begin(42,7,9007199254740993LL); r.Work(9,id,9007199254741000LL,9007199254741010LL); r.Complete(id);
    r.Resource(42,5,9007199254741020LL,std::nullopt);
    std::ostringstream out; r.WriteJsonl(out); auto text=out.str();
    EXPECT_NE(text.find("9007199254740993"),std::string::npos);
    EXPECT_NE(text.find("\"pid\":42"),std::string::npos); EXPECT_EQ(text.find("999"),std::string::npos);
    EXPECT_NE(text.find("\"bytes\":null"),std::string::npos); EXPECT_NE(text.find("\"end_ns\":null"),std::string::npos);
}
TEST(OwnedGpuRecords, ResourceTrackingAlsoHasAHardCap) {
    OwnedGpuRecords r(42,50,2); r.Resource(42,1,1,64); r.Resource(42,2,2,64); r.Resource(42,3,3,64);
    EXPECT_EQ(r.TrackedResources(),2u); EXPECT_GT(r.droppedRecords,0u); EXPECT_FALSE(r.Valid());
}
TEST(OwnedGpuRecords, DuplicateOrZeroResourceIdsAreRejected) {
    OwnedGpuRecords r(42); r.Resource(42,99,1,64); r.Resource(42,99,2,64); r.Resource(42,0,3,64);
    EXPECT_EQ(r.invalidRecords,2u); EXPECT_EQ(r.rows.size(),1u); EXPECT_FALSE(r.Valid());
}
TEST(OwnedGpuRecords, NegativeOwnedSubmissionCannotPass) {
    OwnedGpuRecords r(42); r.Begin(42,7,-1); EXPECT_EQ(r.invalidRecords,1u); EXPECT_FALSE(r.Valid());
}
TEST(OwnedGpuRecords, UnknownCompletionIsNotInventedAsAnOwnedEvent) {
    OwnedGpuRecords r(42); r.Complete(999); EXPECT_EQ(r.invalidRecords,1u); EXPECT_TRUE(r.rows.empty()); EXPECT_FALSE(r.Valid());
}
