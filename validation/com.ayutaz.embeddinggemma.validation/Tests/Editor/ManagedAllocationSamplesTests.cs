using System;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class ManagedAllocationSamplesTests
    {
        static ManagedAllocationSample Row(string marker,double bytes,string thread="Main Thread",int frame=1)
            => new ManagedAllocationSample(marker,thread,frame,bytes);

        [Test] public void ObservedZeroAndKnownPositiveAllocationsRemainDistinct()
        {
            var result=ManagedAllocationSamples.RequireSingleMainThreadSamples(new[]{"noop","small"},new[]{Row("noop",0),Row("small",65568)});
            Assert.That(result["noop"],Is.Zero);Assert.That(result["small"],Is.EqualTo(65568));Assert.That(result.Count,Is.EqualTo(2));
        }
        [Test] public void UnrelatedMarkersAndOtherThreadsNeverInflateTheRequestedMarker()
        {
            var result=ManagedAllocationSamples.RequireSingleMainThreadSamples(new[]{"work"},new[]{Row("work",999,"Render Thread"),Row("work.Child",888),Row("private unrelated",777),Row("work",24)});
            Assert.That(result.Count,Is.EqualTo(1));Assert.That(result["work"],Is.EqualTo(24));
        }
        [Test] public void MissingMarkerIsUnavailableRatherThanZero()
        {
            Assert.Throws<InvalidOperationException>(()=>ManagedAllocationSamples.RequireSingleMainThreadSamples(new[]{"work"},Array.Empty<ManagedAllocationSample>()));
            Assert.Throws<InvalidOperationException>(()=>ManagedAllocationSamples.RequireSingleMainThreadSamples(new[]{"work"},new[]{Row("work",0,"Render Thread")}));
        }
        [Test] public void RepeatedMarkerAcrossFramesIsAmbiguousRatherThanAnAllocationSum()
        {
            Assert.Throws<InvalidOperationException>(()=>ManagedAllocationSamples.RequireSingleMainThreadSamples(new[]{"work"},new[]{Row("work",8,frame:1),Row("work",8,frame:2)}));
        }
        [Test] public void NonIntegralNegativeNonFiniteAndInexactByteCountsAreRejected()
        {
            foreach(var bytes in new[]{-1.0,.5,double.NaN,double.PositiveInfinity,double.NegativeInfinity,9007199254740994.0})
                Assert.Throws<InvalidOperationException>(()=>ManagedAllocationSamples.RequireSingleMainThreadSamples(new[]{"work"},new[]{Row("work",bytes)}));
        }
        [Test] public void InvalidFramesCannotProvideAllocationEvidence()
        {
            Assert.Throws<InvalidOperationException>(()=>ManagedAllocationSamples.RequireSingleMainThreadSamples(new[]{"work"},new[]{Row("work",0,frame:-1)}));
        }
        [Test] public void ExpectedMarkersMustBeNonemptyDistinctAndExplicit()
        {
            Assert.Throws<ArgumentNullException>(()=>ManagedAllocationSamples.RequireSingleMainThreadSamples(null,Array.Empty<ManagedAllocationSample>()));
            foreach(var markers in new[]{Array.Empty<string>(),new[]{"work","work"},new[]{" "},new string[]{null}})
                Assert.Throws<ArgumentException>(()=>ManagedAllocationSamples.RequireSingleMainThreadSamples(markers,Array.Empty<ManagedAllocationSample>()));
        }
        [Test] public void ACaseVariantDoesNotSatisfyTheExactMarker()
        {
            Assert.Throws<InvalidOperationException>(()=>ManagedAllocationSamples.RequireSingleMainThreadSamples(new[]{"Work"},new[]{Row("work",0)}));
        }
        [Test] public void NullSampleCollectionIsNotAnEmptySuccessfulMeasurement()
        {
            Assert.Throws<ArgumentNullException>(()=>ManagedAllocationSamples.RequireSingleMainThreadSamples(new[]{"work"},null));
        }
    }
}
