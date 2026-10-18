using System;
using System.Collections.Generic;
using System.Diagnostics;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class RawQpcStageClockTests
    {
        [Test] public void OrderedMatchingNativeBridgeIsAccepted()
        { Assert.DoesNotThrow(()=>RawQpcStageClock.ValidateBridge(10,11,12,10000000,10000000,true)); }

        [TestCase(-1,0,1)] [TestCase(11,10,12)] [TestCase(10,13,12)] [TestCase(12,12,11)]
        public void InvalidNativeBracketIsRejected(long before,long sample,long after)
        { Assert.Throws<ArgumentException>(()=>RawQpcStageClock.ValidateBridge(before,sample,after,10000000,10000000,true)); }

        [TestCase(0,0)] [TestCase(-1,-1)] [TestCase(10000000,1000000)]
        public void NonpositiveOrDifferentClockFrequencyIsRejected(long native,long runtime)
        { Assert.Throws<ArgumentException>(()=>RawQpcStageClock.ValidateBridge(10,11,12,native,runtime,true)); }

        [Test] public void ACoarseRuntimeClockCannotCertifyRawQpcEpoch()
        { Assert.Throws<ArgumentException>(()=>RawQpcStageClock.ValidateBridge(10,11,12,1000,1000,false)); }

        [Test] public void WindowPreservesIntegerEpochBeyondDoublePrecision()
        {
            var values=new Queue<long>(new[] {9007199254740993L,9007199254741123L});
            var window=new RawQpcStageWindow(()=>values.Dequeue(),10000000);
            window.Finish(); var row=window.ToJson();
            Assert.That(row.Value<long>("begin_ticks"),Is.EqualTo(9007199254740993L));
            Assert.That(row.Value<long>("end_ticks"),Is.EqualTo(9007199254741123L));
            Assert.That(row.Value<long>("frequency_hz"),Is.EqualTo(10000000));
            Assert.That(row.Value<string>("timestamp_unit"),Is.EqualTo("qpc_ticks"));
            Assert.That(row["begin_ticks"].Type,Is.EqualTo(JTokenType.Integer));
            Assert.That(values.Count,Is.Zero);
        }

        [Test] public void UnfinishedWindowCannotBeExportedAsComplete()
        { var window=new RawQpcStageWindow(()=>10,10000000); Assert.Throws<InvalidOperationException>(()=>window.ToJson()); }

        [Test] public void FinishingTwiceCannotChangeAnAlreadyRecordedEnd()
        {
            var values=new Queue<long>(new[] {10L,20L,30L});var window=new RawQpcStageWindow(()=>values.Dequeue(),10000000);
            window.Finish();Assert.Throws<InvalidOperationException>(()=>window.Finish());
            Assert.That(window.ToJson().Value<long>("end_ticks"),Is.EqualTo(20));Assert.That(values.Count,Is.EqualTo(1));
        }

        [Test] public void BackwardsClockFaultCannotBeRetriedIntoAPass()
        {
            var values=new Queue<long>(new[] {20L,10L,30L});var window=new RawQpcStageWindow(()=>values.Dequeue(),10000000);
            Assert.Throws<InvalidOperationException>(()=>window.Finish());
            Assert.Throws<InvalidOperationException>(()=>window.Finish());
            Assert.Throws<InvalidOperationException>(()=>window.ToJson());
        }

        [Test] public void CounterFailureCannotBeRetriedIntoAPass()
        {
            var count=0;var window=new RawQpcStageWindow(()=>++count==1?10:throw new ApplicationException("counter failed"),10000000);
            Assert.Throws<ApplicationException>(()=>window.Finish());
            Assert.Throws<InvalidOperationException>(()=>window.Finish());
            Assert.Throws<InvalidOperationException>(()=>window.ToJson());
        }

        [Test] public void MissingCounterIsRejected()
        { Assert.Throws<ArgumentNullException>(()=>new RawQpcStageWindow(null,10000000)); }

        [TestCase(0)] [TestCase(-1)] public void InvalidWindowFrequencyIsRejected(long frequency)
        { Assert.Throws<ArgumentOutOfRangeException>(()=>new RawQpcStageWindow(()=>10,frequency)); }

        [Test] public void NegativeWindowStartIsRejected()
        { Assert.Throws<ArgumentOutOfRangeException>(()=>new RawQpcStageWindow(()=>-1,10000000)); }

        [Test] public void SameTickEndpointsRemainZeroDurationWithoutInventingPositiveWork()
        {
            var window=new RawQpcStageWindow(()=>10,10000000);window.Finish();var row=window.ToJson();
            Assert.That(row.Value<long>("begin_ticks"),Is.EqualTo(10));Assert.That(row.Value<long>("end_ticks"),Is.EqualTo(10));
        }

        [Test] public void ActualWindowsStageCounterFitsTheNativeQpcBracket()
        {
            if(Environment.OSVersion.Platform!=PlatformID.Win32NT)
            { Assert.Throws<PlatformNotSupportedException>(()=>RawQpcStageClock.CaptureBridge());return; }
            var bridge=RawQpcStageClock.CaptureBridge();
            Assert.That(bridge.Value<bool>("validated"),Is.True);
            Assert.That(bridge.Value<long>("native_qpc_before"),Is.LessThanOrEqualTo(bridge.Value<long>("counter_ticks")));
            Assert.That(bridge.Value<long>("native_qpc_after"),Is.GreaterThanOrEqualTo(bridge.Value<long>("counter_ticks")));
            Assert.That(bridge.Value<string>("clock_source"),Is.EqualTo("QueryPerformanceCounter"));
            Assert.That(bridge.Value<long>("frequency_hz"),Is.GreaterThan(0));
            var window=RawQpcStageWindow.Start();window.Finish();
            Assert.That(window.ToJson().Value<long>("frequency_hz"),Is.EqualTo(bridge.Value<long>("frequency_hz")));
        }
    }
}
