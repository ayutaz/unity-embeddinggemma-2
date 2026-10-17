using System;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class FrameAllocationWindowTests
    {
        [Test] public void IncludesTheFirstCompletedCounterSampleAfterOperationCompletion()
        {
            var result=FrameAllocationWindow.ThroughFirstSampleAfterCompletion(new long[]{10,20,30},2,8,false);
            Assert.That(result.ManagedBytes,Is.EqualTo(60));Assert.That(result.IncludedSamples,Is.EqualTo(3));Assert.That(result.DiscardedTailSamples,Is.Zero);
        }
        [Test] public void ActualZeroIsAcceptedOnlyWhenASampleExists()
        {
            Assert.That(FrameAllocationWindow.ThroughFirstSampleAfterCompletion(new long[]{0},0,8,false).ManagedBytes,Is.Zero);
            Assert.Throws<InvalidOperationException>(()=>FrameAllocationWindow.ThroughFirstSampleAfterCompletion(Array.Empty<long>(),0,8,false));
        }
        [Test] public void LaterSamplesAreExcludedWithoutMutatingTheCapture()
        {
            var values=new long[]{10,20,999};var result=FrameAllocationWindow.ThroughFirstSampleAfterCompletion(values,1,8,false);
            Assert.That(result.ManagedBytes,Is.EqualTo(30));Assert.That(result.IncludedSamples,Is.EqualTo(2));Assert.That(result.DiscardedTailSamples,Is.EqualTo(1));Assert.That(values[2],Is.EqualTo(999));
        }
        [Test] public void NoSampleAfterCompletionIsUnavailableRatherThanZero()
        {
            Assert.Throws<InvalidOperationException>(()=>FrameAllocationWindow.ThroughFirstSampleAfterCompletion(new long[]{10,20},2,8,false));
        }
        [Test] public void WrappedOrFullStorageCannotProveCompleteCoverage()
        {
            Assert.Throws<InvalidOperationException>(()=>FrameAllocationWindow.ThroughFirstSampleAfterCompletion(new long[]{10},0,8,true));
            Assert.Throws<InvalidOperationException>(()=>FrameAllocationWindow.ThroughFirstSampleAfterCompletion(new long[]{10,20},0,2,false));
        }
        [Test] public void InvalidCounterValuesAnywhereRejectTheCapture()
        {
            Assert.Throws<InvalidOperationException>(()=>FrameAllocationWindow.ThroughFirstSampleAfterCompletion(new long[]{-1},0,8,false));
            Assert.Throws<InvalidOperationException>(()=>FrameAllocationWindow.ThroughFirstSampleAfterCompletion(new long[]{10,-1},0,8,false));
        }
        [Test] public void TotalOverflowCannotWrapIntoAValidByteCount()
        {
            Assert.Throws<OverflowException>(()=>FrameAllocationWindow.ThroughFirstSampleAfterCompletion(new long[]{long.MaxValue,1},1,8,false));
        }
        [Test] public void InvalidMetadataIsRejectedBeforeAggregation()
        {
            Assert.Throws<ArgumentNullException>(()=>FrameAllocationWindow.ThroughFirstSampleAfterCompletion(null,0,8,false));
            Assert.Throws<ArgumentOutOfRangeException>(()=>FrameAllocationWindow.ThroughFirstSampleAfterCompletion(new long[]{0},-1,8,false));
            Assert.Throws<ArgumentOutOfRangeException>(()=>FrameAllocationWindow.ThroughFirstSampleAfterCompletion(new long[]{0},0,0,false));
            Assert.Throws<ArgumentOutOfRangeException>(()=>FrameAllocationWindow.ThroughFirstSampleAfterCompletion(new long[]{0,0},0,1,false));
        }
        [Test] public void EqualValuesOnDifferentFramesAreNotDeduplicated()
        {
            Assert.That(FrameAllocationWindow.ThroughFirstSampleAfterCompletion(new long[]{8,8,8},2,8,false).ManagedBytes,Is.EqualTo(24));
        }
    }
}
