using System;
using System.Collections;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Tests
{
    public sealed class AudioPreprocessorAsyncTests
    {
        const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;
        static Model Model() => (Model)typeof(AudioPreprocessorTests).GetMethod("SmallModel", BindingFlags.Static | BindingFlags.NonPublic).Invoke(null, new object[] { null });
        static AudioPcmInput Input(float value = .25f) => new AudioPcmInput(Enumerable.Repeat(value, 321).ToArray());
        static Task<AudioPreprocessingResult> Call(AudioPreprocessor stage, AudioPcmInput input, CancellationToken token = default)
        {
            var method = typeof(AudioPreprocessor).GetMethod("ProcessAsync", new[] { typeof(AudioPcmInput), typeof(CancellationToken) });
            Assert.That(method, Is.Not.Null, "Repeated/cancelled PCM processing must retire completed GPU caches at a fence.");
            return (Task<AudioPreprocessingResult>)method.Invoke(stage, new object[] { input, token });
        }
        static object Context(AudioPreprocessor stage)
        {
            var worker = typeof(AudioPreprocessor).GetField("worker", Private).GetValue(stage);
            var backend = typeof(Worker).GetField("m_Backend", Private).GetValue(worker);
            return typeof(Worker).Assembly.GetType("Unity.InferenceEngine.GfxDeviceBackend").GetField("m_GfxContext", Private).GetValue(backend);
        }
        static IEnumerator Wait(Task task) { while (!task.IsCompleted) { UnityEditor.EditorApplication.QueuePlayerLoopUpdate(); yield return null; } }
        [UnityTest] public IEnumerator OwnedGpuOutputsSurviveCompletedCacheRetirementRepeatAndDisposal()
        {
            using var stage = new AudioPreprocessor(Model(), BackendType.GPUCompute); using var baseline = stage.Process(Input());
            var context = Context(stage); var first = Call(stage, Input()); Assert.That(first.IsCompleted, Is.False); yield return Wait(first);
            using var result = first.GetAwaiter().GetResult(); Assert.That(Context(stage), Is.Not.SameAs(context));
            var again = Call(stage, Input(.5f)); yield return Wait(again); using var second = again.GetAwaiter().GetResult(); stage.Dispose();
            Assert.That(result.Features.backendType, Is.EqualTo(BackendType.GPUCompute)); Assert.That(result.FrameMask.backendType, Is.EqualTo(BackendType.GPUCompute));
            Assert.That(result.Features.DownloadToArray(), Is.EqualTo(baseline.Features.DownloadToArray()));
            Assert.That(second.Features.DownloadToArray()[0], Is.EqualTo(result.Features.DownloadToArray()[0] * 2));
            Assert.That(result.FrameMask.DownloadToArray().Take(3), Is.EqualTo(new[] { 1, 1, 0 }));
        }
        [UnityTest] public IEnumerator CancellationAtPreprocessingFenceRetiresContextAndAllowsReuse()
        {
            using var stage = new AudioPreprocessor(Model(), BackendType.GPUCompute); using var cancellation = new CancellationTokenSource();
            var context = Context(stage); var pending = Call(stage, Input(), cancellation.Token); Assert.That(pending.IsCompleted, Is.False); cancellation.Cancel();
            yield return Wait(pending); Assert.That(pending.IsCanceled, Is.True); Assert.That(Context(stage), Is.Not.SameAs(context));
            var retry = Call(stage, Input()); yield return Wait(retry); using var result = retry.GetAwaiter().GetResult(); Assert.That(result.Features.backendType, Is.EqualTo(BackendType.GPUCompute));
        }
        [UnityTest] public IEnumerator PendingCallRejectsOverlapAndDisposalWaitsForItsFence()
        {
            using var stage = new AudioPreprocessor(Model(), BackendType.GPUCompute); var pending = Call(stage, Input());
            Assert.Throws<InvalidOperationException>(() => stage.Process(Input())); stage.Dispose();
            Assert.That(typeof(AudioPreprocessor).GetField("worker", Private).GetValue(stage), Is.Not.Null);
            yield return Wait(pending); Assert.Throws<ObjectDisposedException>(() => pending.GetAwaiter().GetResult());
            Assert.That(typeof(AudioPreprocessor).GetField("worker", Private).GetValue(stage), Is.Null);
        }
        [UnityTest] public IEnumerator PreCancellationDoesNotScheduleOrReplaceTheContext()
        {
            using var stage = new AudioPreprocessor(Model(), BackendType.GPUCompute); using var cancellation = new CancellationTokenSource(); cancellation.Cancel();
            var context = Context(stage); var pending = Call(stage, Input(), cancellation.Token); yield return Wait(pending);
            Assert.That(pending.IsCanceled, Is.True); Assert.That(Context(stage), Is.SameAs(context));
            var retry = Call(stage, Input()); yield return Wait(retry); using var result = retry.GetAwaiter().GetResult();
        }
    }
}
