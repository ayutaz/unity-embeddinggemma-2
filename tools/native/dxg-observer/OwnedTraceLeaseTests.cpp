#include "OwnedTraceLease.h"
#include <gtest/gtest.h>
#include <vector>
#include <type_traits>
using namespace embeddinggemma::diagnostics;
TEST(OwnedTraceLease, ClosesOnlyTheSuccessfullyStartedHandleOnce) {
    std::vector<uint64_t> handles;
    { OwnedTraceLease lease([&](uint64_t h){handles.push_back(h);return 0;});
      lease.Started(0,123); EXPECT_TRUE(lease.owned); EXPECT_EQ(lease.Close(),0u); EXPECT_TRUE(lease.stopped); lease.Close(); }
    EXPECT_EQ(handles,(std::vector<uint64_t>{123}));
}
TEST(OwnedTraceLease, ExistingForeignSessionIsNeverStopped) {
    int calls=0; { OwnedTraceLease lease([&](uint64_t){++calls;return 0;}); lease.Started(183,999); lease.Close(); }
    EXPECT_EQ(calls,0);
}
TEST(OwnedTraceLease, ExceptionDuringProviderSetupStillClosesOwnedSession) {
    int calls=0;
    try { OwnedTraceLease lease([&](uint64_t h){EXPECT_EQ(h,123u);++calls;return 0;});lease.Started(0,123);throw std::runtime_error("provider setup failed"); }
    catch(const std::runtime_error&) {}
    EXPECT_EQ(calls,1);
}
TEST(OwnedTraceLease, FailedStopRetainsOwnershipAndCannotBecomeAnUnqualifiedPass) {
    int calls=0; OwnedTraceLease lease([&](uint64_t){return ++calls==1 ? 5u : 0u;});
    lease.Started(0,123); EXPECT_EQ(lease.Close(),5u); EXPECT_TRUE(lease.owned); EXPECT_TRUE(lease.hadStopFailure); EXPECT_FALSE(lease.stopped);
    EXPECT_EQ(lease.Close(),0u); EXPECT_FALSE(lease.owned); EXPECT_TRUE(lease.stopped); EXPECT_TRUE(lease.hadStopFailure); EXPECT_EQ(calls,2);
}
TEST(OwnedTraceLease, CannotReplaceAStillOwnedHandle) {
    OwnedTraceLease lease([](uint64_t){return 0u;}); lease.Started(0,123);
    EXPECT_THROW(lease.Started(0,999),std::logic_error);
}
TEST(OwnedTraceLease, OwnershipCannotBeCopiedOrMovedToAnotherDestructor) {
    EXPECT_FALSE(std::is_copy_constructible_v<OwnedTraceLease>); EXPECT_FALSE(std::is_move_constructible_v<OwnedTraceLease>);
}
TEST(OwnedTraceLease, StopExceptionsRemainFailedAndDoNotEscapeCleanup) {
    OwnedTraceLease lease([](uint64_t)->uint32_t{throw std::runtime_error("stop error");}); lease.Started(0,123);
    EXPECT_EQ(lease.Close(),UINT32_MAX); EXPECT_TRUE(lease.hadStopFailure); EXPECT_TRUE(lease.owned);
}
