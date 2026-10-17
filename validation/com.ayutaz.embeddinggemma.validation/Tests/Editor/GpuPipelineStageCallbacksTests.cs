using System;
using System.Collections.Generic;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    // Fake callbacks prove diagnostic ordering/ownership only, never GPU acceptance.
    public sealed class GpuPipelineStageCallbacksTests
    {
        [Test] public void EndsModelMarkerBeforeCacheReleaseAndStartsNextOnlyAfterCompletedCleanup()
        {
            var events = new List<string>();
            var before = new Action[] { () => events.Add("cache0"), () => events.Add("cache1") };
            var completed = new Action[] { () => events.Add("completed0"), () => events.Add("completed1") };
            var originalBefore = (Action[])before.Clone(); var originalCompleted = (Action[])completed.Clone();
            using (var trace = new GpuPipelineStageCallbacks(before, completed,
                i => events.Add("begin" + i), (i, aborted) => events.Add("end" + i + ":" + aborted)))
            {
                trace.BeginCall(); before[0](); completed[0](); before[1](); completed[1](); trace.EndCall();
                Assert.That(events, Is.EqualTo(new[] { "begin0", "end0:False", "cache0", "completed0", "begin1", "end1:False", "cache1", "completed1" }));
            }
            for (var i = 0; i < 2; i++) { Assert.That(before[i], Is.SameAs(originalBefore[i])); Assert.That(completed[i], Is.SameAs(originalCompleted[i])); }
        }
        [Test] public void CancellationBetweenStagesClosesUnsubmittedMarkerAsAborted()
        {
            var events = new List<string>(); var before = new Action[] { () => {}, () => {} }; var completed = (Action[])before.Clone();
            using var trace = new GpuPipelineStageCallbacks(before, completed, i => events.Add("begin" + i), (i, aborted) => events.Add("end" + i + ":" + aborted));
            trace.BeginCall(); before[0](); completed[0](); trace.EndCall(); trace.EndCall();
            Assert.That(events, Is.EqualTo(new[] { "begin0", "end0:False", "begin1", "end1:True" }));
        }
        [Test] public void OriginalCleanupExceptionPropagatesWithoutStartingAnotherMarker()
        {
            var failure = new InvalidOperationException("controlled cache failure"); var begins = new List<int>();
            var before = new Action[] { () => {}, () => {} }; var completed = new Action[] { () => throw failure, () => {} };
            using var trace = new GpuPipelineStageCallbacks(before, completed, begins.Add, (_, __) => {});
            trace.BeginCall(); before[0](); Assert.That(Assert.Throws<InvalidOperationException>(() => completed[0]()), Is.SameAs(failure));
            trace.EndCall(); Assert.That(begins, Is.EqualTo(new[] { 0 }));
        }
        [Test] public void DisposeDoesNotOverwriteAnotherOwnersReplacementAndIsIdempotent()
        {
            Action original = () => {}; Action replacement = () => {};
            var before = new[] { original }; var completed = new[] { original }; var ends = 0;
            var trace = new GpuPipelineStageCallbacks(before, completed, _ => {}, (_, aborted) => { Assert.That(aborted, Is.True); ends++; });
            trace.BeginCall(); before[0] = replacement; trace.Dispose(); trace.Dispose();
            Assert.That(before[0], Is.SameAs(replacement)); Assert.That(completed[0], Is.SameAs(original)); Assert.That(ends, Is.EqualTo(1));
            Assert.Throws<ObjectDisposedException>(() => trace.BeginCall());
        }
        [Test] public void CallsMustBeSerialAndStageOrderMustMatch()
        {
            var actions = new Action[] { () => {}, () => {} }; var before = (Action[])actions.Clone(); var completed = (Action[])actions.Clone();
            using var trace = new GpuPipelineStageCallbacks(before, completed, _ => {}, (_, __) => {});
            Assert.Throws<InvalidOperationException>(() => before[0]()); trace.BeginCall();
            Assert.Throws<InvalidOperationException>(() => trace.BeginCall()); Assert.Throws<InvalidOperationException>(() => before[1]());
            trace.EndCall(); trace.BeginCall(); before[0](); completed[0](); before[1](); completed[1](); trace.EndCall();
        }
        [Test] public void InvalidArgumentsDoNotPartiallyReplaceCallbacks()
        {
            Action original = () => {}; var before = new[] { original };
            Assert.Throws<ArgumentException>(() => new GpuPipelineStageCallbacks(before, new Action[0], _ => {}, (_, __) => {}));
            Assert.Throws<ArgumentNullException>(() => new GpuPipelineStageCallbacks(before, new Action[] { null }, _ => {}, (_, __) => {}));
            Assert.That(before[0], Is.SameAs(original));
        }
    }
}
