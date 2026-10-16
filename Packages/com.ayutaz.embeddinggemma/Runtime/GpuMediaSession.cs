using System;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma
{
    /// <summary>
    /// Owns one prepared module. All calls and disposal stay on the creating Unity
    /// thread. Closing cancels an active call and retires its stages only after
    /// that call has returned across the staged encoder's GPU fences.
    /// </summary>
    public sealed class GpuMediaSession : IDisposable, ITextEmbedder
    {
        readonly IAsyncImageEmbedder image;
        readonly IAsyncAudioEmbedder audio;
        readonly ITextEmbedder query;
        readonly IDisposable[] resources;
        readonly CancellationTokenSource closing = new CancellationTokenSource();
        readonly int ownerThread = Thread.CurrentThread.ManagedThreadId;
        Task pending;
        bool executing, closeRequested;
        public string ModuleName { get; }
        public BackendType Backend => BackendType.GPUCompute;
        public bool IsClosed { get; private set; }

        internal GpuMediaSession(string name, IAsyncImageEmbedder image, IAsyncAudioEmbedder audio,
            ITextEmbedder query, IDisposable[] resources)
        {
            if ((image == null ? 0 : 1) + (audio == null ? 0 : 1) + (query == null ? 0 : 1) != 1 ||
                image != null && image.Backend != BackendType.GPUCompute ||
                audio != null && audio.Backend != BackendType.GPUCompute ||
                query != null && query.Backend != BackendType.GPUCompute)
                throw new ArgumentException("One GPUCompute module provider is required.");
            ModuleName = name; this.image = image; this.audio = audio; this.query = query;
            this.resources = resources ?? throw new ArgumentNullException(nameof(resources));
        }
        void ThreadCheck()
        {
            if (Thread.CurrentThread.ManagedThreadId != ownerThread)
                throw new InvalidOperationException("GPU module calls and disposal require the creating Unity thread.");
        }
        void Check()
        {
            ThreadCheck();
            if (closeRequested) throw new ObjectDisposedException(nameof(GpuMediaSession));
            if (executing) throw new InvalidOperationException("Only one module call may be active.");
        }
        public Task<float[]> EmbedImageAsync(Texture input, TextRole role = TextRole.Document, string title = null, CancellationToken cancellationToken = default)
        {
            Check(); if (image == null) throw new InvalidOperationException("This is not an image module.");
            return Start(token => image.EmbedAsync(input, role, title, token), cancellationToken);
        }
        public Task<float[]> EmbedAudioAsync(AudioPcmInput input, TextRole role = TextRole.Document, string title = null, CancellationToken cancellationToken = default)
        {
            Check(); if (audio == null) throw new InvalidOperationException("This is not an audio module.");
            return Start(token => audio.EmbedAsync(input, role, title, token), cancellationToken);
        }
        Task<float[]> Start(Func<CancellationToken, Task<float[]>> call, CancellationToken token)
        {
            token.ThrowIfCancellationRequested(); executing = true;
            var task = Execute(call, token); pending = task; return task;
        }
        async Task<float[]> Execute(Func<CancellationToken, Task<float[]>> call, CancellationToken token)
        {
            using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(token, closing.Token);
            try
            {
                var result = await call(cancellation.Token);
                cancellation.Token.ThrowIfCancellationRequested(); return result;
            }
            finally { executing = false; if (closeRequested) Release(); }
        }
        public float[] EmbedQuery(string text)
        {
            Check(); if (query == null) throw new InvalidOperationException("Load the text module for queries.");
            executing = true; try { return query.EmbedQuery(text); } finally { executing = false; }
        }
        float[] ITextEmbedder.EmbedDocument(string text, string title)
            => throw new NotSupportedException("Media documents must use Texture or PCM encoding.");
        public void Dispose()
        {
            ThreadCheck(); if (closeRequested) return;
            closeRequested = true; closing.Cancel(); if (!executing) Release();
        }
        public async Task CloseAsync()
        {
            Dispose(); var call = pending;
            if (call != null) try { await call; } catch (OperationCanceledException) { }
        }
        void Release()
        {
            if (IsClosed) return; IsClosed = true;
            try
            {
                // Attempt every release even if a resource fails to dispose.
                Exception failure = null;
                for (var index = resources.Length - 1; index >= 0; index--)
                    try { resources[index]?.Dispose(); } catch (Exception error) { failure ??= error; }
                if (failure != null) throw failure;
            }
            finally { closing.Dispose(); }
        }
    }
}
