using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Tests
{
    public sealed class GpuMediaSessionTests
    {
        sealed class Resource : IDisposable
        {
            readonly List<int> releases; readonly int id;
            public Resource(List<int> releases, int id) { this.releases = releases; this.id = id; }
            public void Dispose() { releases.Add(id); }
        }
        sealed class Image : IAsyncImageEmbedder
        {
            public BackendType Backend { get; set; } = BackendType.GPUCompute;
            public TaskCompletionSource<float[]> Completion = new TaskCompletionSource<float[]>();
            public CancellationToken Token; public TextRole Role; public string Title;
            public Task<float[]> EmbedAsync(Texture input, TextRole role = TextRole.Raw, string title = null, CancellationToken cancellationToken = default)
            { Token = cancellationToken; Role = role; Title = title; return Completion.Task; }
            public float[] Embed(Texture input, TextRole role = TextRole.Raw, string title = null) => throw new AssertionException("Use the staged async API.");
            public void Dispose() { }
        }
        sealed class Audio : IAsyncAudioEmbedder
        {
            public BackendType Backend => BackendType.GPUCompute;
            public readonly TaskCompletionSource<float[]> Completion = new TaskCompletionSource<float[]>();
            public AudioPcmInput Input; public CancellationToken Token;
            public Task<float[]> EmbedAsync(AudioPcmInput input, TextRole role = TextRole.Raw, string title = null, CancellationToken cancellationToken = default)
            { Input = input; Token = cancellationToken; return Completion.Task; }
            public float[] Embed(AudioPcmInput input, TextRole role = TextRole.Raw, string title = null) => throw new AssertionException("Use the staged async API.");
            public void Dispose() { }
        }
        sealed class Query : ITextEmbedder
        {
            public BackendType Backend => BackendType.GPUCompute;
            public float[] EmbedQuery(string text) => new[] { (float)text.Length };
            public float[] EmbedDocument(string text, string title = null) => throw new AssertionException("Media paths are never text documents.");
            public void Dispose() { }
        }
        static Type Required()
        {
            var type = typeof(TextEmbedder).Assembly.GetType("EmbeddingGemma.GpuMediaSession");
            Assert.That(type, Is.Not.Null, "An owning GPU session must await active calls before releasing its stages."); return type;
        }
        static object Create(IAsyncImageEmbedder image = null, IAsyncAudioEmbedder audio = null, ITextEmbedder query = null, params IDisposable[] resources)
        {
            var constructor = Required().GetConstructor(BindingFlags.NonPublic | BindingFlags.Instance, null,
                new[] { typeof(string), typeof(IAsyncImageEmbedder), typeof(IAsyncAudioEmbedder), typeof(ITextEmbedder), typeof(IDisposable[]) }, null);
            Assert.That(constructor, Is.Not.Null);
            return constructor.Invoke(new object[] { "test-module", image, audio, query, resources });
        }
        static object Call(object session, string method, params object[] args) => Required().GetMethod(method).Invoke(session, args);
        static Task<float[]> Embed(object session, CancellationToken token = default)
            => (Task<float[]>)Call(session, "EmbedImageAsync", null, TextRole.Document, null, token);
        static bool Closed(object session) => (bool)Required().GetProperty("IsClosed").GetValue(session);
        static Exception Error(Action action) => Assert.Throws<TargetInvocationException>(() => action()).InnerException;

        [Test] public void ClosingIdleSessionReleasesInReverseOrderExactlyOnce()
        {
            var releases = new List<int>(); var session = Create(new Image(), resources: new IDisposable[] { new Resource(releases, 1), new Resource(releases, 2) });
            ((IDisposable)session).Dispose(); ((IDisposable)session).Dispose();
            CollectionAssert.AreEqual(new[] { 2, 1 }, releases); Assert.That(Closed(session), Is.True);
            Assert.That(Error(() => Embed(session)), Is.InstanceOf<ObjectDisposedException>());
        }
        [UnityTest] public IEnumerator DisposeCancelsButDoesNotReleaseBorrowedStagesBeforePendingGpuCallEnds()
        {
            var releases = new List<int>(); var image = new Image(); var session = Create(image, resources: new[] { new Resource(releases, 1) });
            var task = Embed(session); ((IDisposable)session).Dispose(); var close = (Task)Call(session, "CloseAsync");
            Assert.That(image.Token.IsCancellationRequested, Is.True); Assert.That(close.IsCompleted, Is.False);
            Assert.That(releases, Is.Empty); Assert.That(Closed(session), Is.False);
            image.Completion.SetCanceled(); while (!task.IsCompleted || !close.IsCompleted) yield return null;
            Assert.That(task.IsCanceled, Is.True); Assert.That(close.Status, Is.EqualTo(TaskStatus.RanToCompletion));
            CollectionAssert.AreEqual(new[] { 1 }, releases); Assert.That(Closed(session), Is.True);
        }
        [UnityTest] public IEnumerator CancellationAlsoRejectsLateSuccessAndDefersReleaseUntilCallReturns()
        {
            var releases = new List<int>(); var image = new Image(); var session = Create(image, resources: new[] { new Resource(releases, 1) });
            var task = Embed(session); ((IDisposable)session).Dispose(); image.Completion.SetResult(new[] { 1f });
            while (!task.IsCompleted) yield return null;
            Assert.That(task.IsCanceled, Is.True); CollectionAssert.AreEqual(new[] { 1 }, releases);
        }
        [UnityTest] public IEnumerator OnlyOneCallMayOwnTheSessionAndFailureLeavesItReusable()
        {
            var image = new Image(); var session = Create(image); var first = Embed(session);
            Assert.That(Error(() => Embed(session)), Is.InstanceOf<InvalidOperationException>());
            image.Completion.SetException(new InvalidOperationException("retained GPU failure")); while (!first.IsCompleted) yield return null;
            Assert.That(first.Exception.InnerException.Message, Is.EqualTo("retained GPU failure"));
            image.Completion = new TaskCompletionSource<float[]>(); var second = Embed(session);
            image.Completion.SetResult(new[] { .25f }); while (!second.IsCompleted) yield return null;
            Assert.That(second.Result[0], Is.EqualTo(.25f));
            ((IDisposable)session).Dispose(); Assert.That(Closed(session), Is.True);
        }
        [UnityTest] public IEnumerator ExternalCancellationDoesNotCloseTheSession()
        {
            var image = new Image(); var session = Create(image); using var cancellation = new CancellationTokenSource();
            var task = Embed(session, cancellation.Token); cancellation.Cancel(); Assert.That(image.Token.IsCancellationRequested, Is.True);
            image.Completion.SetCanceled(); while (!task.IsCompleted) yield return null;
            Assert.That(Closed(session), Is.False); ((IDisposable)session).Dispose();
        }
        [UnityTest] public IEnumerator AudioUsesActualPcmAndStagedApi()
        {
            var audio = new Audio(); var session = Create(audio: audio); var input = new AudioPcmInput(new float[179200]);
            var task = (Task<float[]>)Call(session, "EmbedAudioAsync", input, TextRole.Document, null, CancellationToken.None);
            Assert.That(audio.Input, Is.SameAs(input)); audio.Completion.SetResult(new[] { .5f });
            while (!task.IsCompleted) yield return null;
            Assert.That(task.Result[0], Is.EqualTo(.5f)); ((IDisposable)session).Dispose();
        }
        [Test] public void TextQueriesUseTheQueryProviderAndCannotUseImageMethods()
        {
            var session = Create(query: new Query()); Assert.That(((float[])Call(session, "EmbedQuery", "猫"))[0], Is.EqualTo(1));
            Assert.That(Error(() => Embed(session)), Is.InstanceOf<InvalidOperationException>()); ((IDisposable)session).Dispose();
            Assert.That(Error(() => Call(session, "EmbedQuery", "猫")), Is.InstanceOf<ObjectDisposedException>());
        }
        [Test] public void CpuProvidersAreRejectedByTheOwningSession()
        {
            Assert.That(Error(() => Create(new Image { Backend = BackendType.CPU })), Is.InstanceOf<ArgumentException>());
        }
        [Test] public void PreparedBundleExposesProgressiveCancellableLoadingAndModuleFileSelection()
        {
            var type = typeof(TextEmbedder).Assembly.GetType("EmbeddingGemma.GpuMediaBundle");
            Assert.That(type.GetMethod("LoadAsync", new[] { typeof(string), typeof(CancellationToken) }), Is.Not.Null);
            Assert.That(type.GetMethod("FilesForModule", new[] { typeof(string) }), Is.Not.Null);
        }
    }
}
