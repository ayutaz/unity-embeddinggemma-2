using System;
using System.Collections.Generic;
using System.Linq;
using Unity.InferenceEngine;

namespace EmbeddingGemma
{
    /// <summary>
    /// Searches owned copies of final image/audio/text vectors with a borrowed GPU text
    /// query provider. The caller encodes media with the appropriate GPU encoder first;
    /// file paths and display metadata are never passed to text document inference.
    /// Providers and encoder resources remain owned by the caller.
    /// </summary>
    public sealed class MultimodalSearchIndex
    {
        readonly SearchDocument[] documents;
        readonly float[][] vectors;
        readonly ITextEmbedder queries;

        public MultimodalSearchIndex(IEnumerable<SearchDocument> documents, IEnumerable<float[]> vectors, ITextEmbedder queries)
        {
            if (documents == null) throw new ArgumentNullException(nameof(documents));
            if (vectors == null) throw new ArgumentNullException(nameof(vectors));
            this.queries = queries ?? throw new ArgumentNullException(nameof(queries));
            if (queries.Backend != BackendType.GPUCompute)
                throw new ArgumentException("Multimodal search queries require GPUCompute.", nameof(queries));
            this.documents = documents.ToArray();
            var rows = vectors.ToArray();
            if (this.documents.Length == 0 || rows.Length != this.documents.Length || this.documents.Any(d => d == null)
                || this.documents.Select(d => d.Id).Distinct(StringComparer.Ordinal).Count() != this.documents.Length)
                throw new ArgumentException("Use nonempty unique documents and exactly one vector per document.", nameof(documents));
            this.vectors = new float[rows.Length][];
            for (var i = 0; i < rows.Length; i++)
            {
                Validate(rows[i]);
                this.vectors[i] = (float[])rows[i].Clone();
            }
        }

        public SearchHit[] Search(string query, int limit = int.MaxValue)
        {
            if (string.IsNullOrWhiteSpace(query)) throw new ArgumentException("Query text is required.", nameof(query));
            if (limit < 1) throw new ArgumentOutOfRangeException(nameof(limit));
            var vector = queries.EmbedQuery(query);
            Validate(vector);
            return documents.Select((document, i) => new SearchHit(document, TextSearchIndex.Cosine(vector, vectors[i])))
                .OrderByDescending(hit => hit.Score).ThenBy(hit => hit.Document.Id, StringComparer.Ordinal).Take(limit).ToArray();
        }

        static void Validate(float[] vector)
        {
            if (vector == null || vector.Length != TextEmbedder.EmbeddingDimension
                || vector.Any(v => float.IsNaN(v) || float.IsInfinity(v))
                || Math.Abs(vector.Sum(v => (double)v * v) - 1) > 0.002)
                throw new ArgumentException("Expected a finite L2-normalized 768-dimensional final embedding.");
        }
    }
}
