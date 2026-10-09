using System;
using System.Collections.Generic;

namespace EmbeddingGemma
{
    /// <summary>Owns one embedding provider. Failed preparation releases all inference resources.</summary>
    public sealed class TextSearchSession : IDisposable
    {
        ITextEmbedder embedder;
        TextSearchIndex index;
        bool disposed;
        public bool IsReady => index != null;
        public int DocumentCount => index?.DocumentCount ?? 0;

        public void Prepare(IEnumerable<SearchDocument> documents, Func<ITextEmbedder> factory)
        {
            CheckDisposed();
            if (factory == null) throw new ArgumentNullException(nameof(factory));
            Release();
            try
            {
                embedder = factory() ?? throw new InvalidOperationException("No embedding provider was created.");
                index = new TextSearchIndex(documents, embedder);
            }
            catch { Release(); throw; }
        }

        public SearchHit[] Search(string query, int limit = int.MaxValue)
        {
            CheckDisposed();
            if (index == null) throw new InvalidOperationException("Prepare the model and documents first.");
            return index.Search(query, limit);
        }

        void CheckDisposed()
        {
            if (disposed) throw new ObjectDisposedException(nameof(TextSearchSession));
        }

        void Release()
        {
            index = null;
            var previous = embedder;
            embedder = null;
            previous?.Dispose();
        }

        public void Dispose()
        {
            if (disposed) return;
            disposed = true;
            Release();
        }
    }
}
