using System;
using System.Collections.Generic;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public interface ITimingFixture
    {
        string Backend { get; }
        object Execute(object input);
        Task<object> ExecuteAsync(object input, CancellationToken token);
    }
    public sealed class MediaStageTimingTests
    {
        sealed class Stage : ITimingFixture
        {
            public string Backend => "unchanged fixture backend";
            public object Input; public CancellationToken Token; public Exception Error;
            public TaskCompletionSource<object> Pending = new TaskCompletionSource<object>();
            public object Execute(object input) { Input = input; if (Error != null) throw Error; return input; }
            public Task<object> ExecuteAsync(object input, CancellationToken token)
            { Input = input; Token = token; if (Error != null) throw Error; return Pending.Task; }
        }
        [Test] public void SynchronousCallPreservesBorrowedIdentityAndReportsSeparateMetrics()
        {
            var rows = new List<JObject>(); var original = new Stage();
            var wrapped = MediaStageTiming.Wrap<ITimingFixture>(original, "preprocessing", rows.Add);
            Assert.That(wrapped.Backend, Is.EqualTo(original.Backend)); Assert.That(rows, Is.Empty);
            var input = new object(); Assert.That(wrapped.Execute(input), Is.SameAs(input));
            Assert.That(original.Input, Is.SameAs(input)); Assert.That(rows, Has.Count.EqualTo(2));
            Assert.That((string)rows[0]["event"], Is.EqualTo("begin"));
            var end = rows[1]; Assert.That((string)end["event"], Is.EqualTo("completed"));
            Assert.That((string)end["stage"], Is.EqualTo("preprocessing"));
            Assert.That((double)end["wall_seconds"], Is.GreaterThanOrEqualTo(0));
            Assert.That((double)end["process_cpu_seconds"], Is.GreaterThanOrEqualTo(0));
            Assert.That((long)end["host_working_set_bytes"], Is.GreaterThan(0));
            Assert.That(end["gpu_elapsed_ns"].Type, Is.EqualTo(JTokenType.Null));
            Assert.That((string)end["cpu_scope"], Does.Contain("whole process"));
        }
        [Test] public async Task PendingAsyncCallDoesNotReportCompletionEarlyOrReplaceResultOrToken()
        {
            var rows = new List<JObject>(); var original = new Stage();
            var wrapped = MediaStageTiming.Wrap<ITimingFixture>(original, "encoder", rows.Add);
            using var cancellation = new CancellationTokenSource(); var input = new object();
            var task = wrapped.ExecuteAsync(input, cancellation.Token);
            Assert.That(task.IsCompleted, Is.False); Assert.That(rows, Has.Count.EqualTo(1));
            Assert.That(original.Input, Is.SameAs(input)); Assert.That(original.Token, Is.EqualTo(cancellation.Token));
            original.Pending.SetResult(input); Assert.That(await task, Is.SameAs(input));
            Assert.That(rows, Has.Count.EqualTo(2)); Assert.That((bool)rows[1]["asynchronous"], Is.True);
            Assert.That(rows[1]["gpu_elapsed_ns"].Type, Is.EqualTo(JTokenType.Null));
        }
        [Test] public void SynchronousFailureKeepsOriginalExceptionAndRecordsFailure()
        {
            var rows = new List<JObject>(); var error = new InvalidOperationException("controlled fixture failure");
            var wrapped = MediaStageTiming.Wrap<ITimingFixture>(new Stage { Error = error }, "backbone", rows.Add);
            Assert.That(Assert.Throws<InvalidOperationException>(() => wrapped.Execute(null)), Is.SameAs(error));
            Assert.That(rows, Has.Count.EqualTo(2)); Assert.That((string)rows[1]["event"], Is.EqualTo("failed"));
            Assert.That((string)rows[1]["error_type"], Is.EqualTo(typeof(InvalidOperationException).FullName));
        }
        [Test] public async Task AsynchronousFailureKeepsOriginalExceptionAndRecordsFailure()
        {
            var rows = new List<JObject>(); var original = new Stage();
            var wrapped = MediaStageTiming.Wrap<ITimingFixture>(original, "encoder", rows.Add);
            var task = wrapped.ExecuteAsync(null, default); var error = new InvalidOperationException("controlled async failure");
            original.Pending.SetException(error);
            try { await task; Assert.Fail("Expected original error"); } catch (InvalidOperationException actual) { Assert.That(actual, Is.SameAs(error)); }
            Assert.That(rows, Has.Count.EqualTo(2)); Assert.That((string)rows[1]["event"], Is.EqualTo("failed"));
        }
        [Test] public async Task CancellationIsRecordedWithoutChangingCancellationIntoSuccess()
        {
            var rows = new List<JObject>(); var original = new Stage();
            var wrapped = MediaStageTiming.Wrap<ITimingFixture>(original, "encoder", rows.Add);
            var task = wrapped.ExecuteAsync(null, default); original.Pending.SetCanceled();
            try { await task; Assert.Fail("Expected cancellation"); } catch (OperationCanceledException) { }
            Assert.That(task.IsCanceled, Is.True); Assert.That(rows, Has.Count.EqualTo(2));
            Assert.That((string)rows[1]["event"], Is.EqualTo("cancelled"));
        }
        [Test] public void ConsecutiveCallsRemainSeparateAndPropertyInspectionAddsNoSamples()
        {
            var rows = new List<JObject>(); var wrapped = MediaStageTiming.Wrap<ITimingFixture>(new Stage(), "backbone", rows.Add);
            wrapped.Execute(null); var ignored = wrapped.Backend; wrapped.Execute(null);
            Assert.That(rows, Has.Count.EqualTo(4));
            Assert.That((long)rows[0]["call_id"], Is.Not.EqualTo((long)rows[2]["call_id"]));
        }
        [TestCase(typeof(EmbeddingGemma.Tests.AsyncImageEmbedderTests))]
        [TestCase(typeof(EmbeddingGemma.Tests.AsyncAudioEmbedderTests))]
        public async Task ActualFacadeFieldsAreDecoratedAndRestoredWithoutOpeningADeviceOrChangingOwnership(Type fixture)
        {
            const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
            using var parts = (IDisposable)Activator.CreateInstance(fixture.GetNestedType("Parts", BindingFlags.NonPublic), true);
            using var facade = (IDisposable)parts.GetType().GetMethod("Create").Invoke(parts, null);
            var pre = facade.GetType().GetField("preprocessor", flags); var encoder = facade.GetType().GetField("encoder", flags);
            var backbone = facade.GetType().GetField("backbone", flags);
            var originalPre = pre.GetValue(facade); var originalEncoder = encoder.GetValue(facade); var originalBack = backbone.GetValue(facade);
            var rows = new List<JObject>();
            using (MediaStageTiming.Attach(facade, rows.Add))
            {
                Assert.That(pre.GetValue(facade), Is.Not.SameAs(originalPre));
                Assert.That(encoder.GetValue(facade), Is.Not.SameAs(originalEncoder));
                Assert.That(backbone.GetValue(facade), Is.Not.SameAs(originalBack));
                using var cancellation = new CancellationTokenSource(); Task pending;
                if (facade is ImageEmbedder)
                    pending = ((IAsyncImageFeatureEncoder)encoder.GetValue(facade)).EncodeAsync(null, cancellation.Token);
                else pending = ((IAsyncAudioFeatureEncoder)encoder.GetValue(facade)).EncodeAsync(null, cancellation.Token);
                Assert.That(pending.IsCompleted, Is.False); Assert.That(rows, Has.Count.EqualTo(1));
                cancellation.Cancel(); try { await pending; Assert.Fail("Expected cancellation"); } catch (OperationCanceledException) { }
                Assert.That(rows, Has.Count.EqualTo(2)); Assert.That((string)rows[1]["stage"], Is.EqualTo("encoder"));
                Assert.That((string)rows[1]["event"], Is.EqualTo("cancelled"));
            }
            Assert.That(pre.GetValue(facade), Is.SameAs(originalPre));
            Assert.That(encoder.GetValue(facade), Is.SameAs(originalEncoder));
            Assert.That(backbone.GetValue(facade), Is.SameAs(originalBack));
        }
    }
}
