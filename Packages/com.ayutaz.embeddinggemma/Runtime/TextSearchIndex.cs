using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace EmbeddingGemma
{
    public sealed class SearchDocument
    {
        public string Id { get; }
        public string Text { get; }
        public string Title { get; }
        public SearchDocument(string id, string text, string title = null)
        {
            if (id == null || !Regex.IsMatch(id, "^[a-z0-9][a-z0-9_-]*$"))
                throw new ArgumentException("Use a lowercase ASCII document ID.", nameof(id));
            if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("Document text is required.", nameof(text));
            Id = id; Text = text; Title = title;
        }
    }

    public sealed class SearchHit
    {
        public SearchDocument Document { get; }
        public double Score { get; }
        internal SearchHit(SearchDocument document, double score) { Document = document; Score = score; }
    }

    /// <summary>Precomputes a small corpus once. The caller owns the embedding provider.</summary>
    public sealed class TextSearchIndex
    {
        readonly SearchDocument[] documents;
        readonly float[][] embeddings;
        readonly ITextEmbedder embedder;
        public int DocumentCount => documents.Length;

        public TextSearchIndex(IEnumerable<SearchDocument> documents, ITextEmbedder embedder)
        {
            if (documents == null) throw new ArgumentNullException(nameof(documents));
            this.embedder = embedder ?? throw new ArgumentNullException(nameof(embedder));
            this.documents = documents.ToArray();
            if (this.documents.Length == 0 || this.documents.Any(document => document == null)
                || this.documents.Select(document => document.Id).Distinct(StringComparer.Ordinal).Count() != this.documents.Length)
                throw new ArgumentException("Use a nonempty corpus with unique document IDs.", nameof(documents));
            embeddings = new float[this.documents.Length][];
            for (var i = 0; i < this.documents.Length; i++)
            {
                var document = this.documents[i];
                var vector = embedder.EmbedDocument(document.Text, document.Title);
                ValidateEmbedding(vector);
                embeddings[i] = (float[])vector.Clone();
            }
        }

        public SearchHit[] Search(string query, int limit = int.MaxValue)
        {
            if (string.IsNullOrWhiteSpace(query)) throw new ArgumentException("Query text is required.", nameof(query));
            if (limit < 1) throw new ArgumentOutOfRangeException(nameof(limit));
            var vector = embedder.EmbedQuery(query);
            ValidateEmbedding(vector);
            return documents.Select((document, i) => new SearchHit(document, Cosine(vector, embeddings[i])))
                .OrderByDescending(hit => hit.Score).ThenBy(hit => hit.Document.Id, StringComparer.Ordinal).Take(limit).ToArray();
        }

        static void ValidateEmbedding(float[] vector)
        {
            if (vector == null || vector.Length != TextEmbedder.EmbeddingDimension)
                throw new ArgumentException("Expected a 768-dimensional embedding.");
            Norm(vector);
        }

        static double Norm(float[] vector)
        {
            if (vector == null || vector.Length == 0 || vector.Any(value => float.IsNaN(value) || float.IsInfinity(value)))
                throw new ArgumentException("Embedding must be a nonempty finite vector.");
            var norm = Math.Sqrt(vector.Sum(value => (double)value * value));
            if (norm == 0) throw new ArgumentException("Embedding norm must be positive.");
            return norm;
        }

        public static double Cosine(float[] first, float[] second)
        {
            var firstNorm = Norm(first); var secondNorm = Norm(second);
            if (first.Length != second.Length) throw new ArgumentException("Embedding dimensions must match.");
            var dot = first.Zip(second, (a, b) => (double)a * b).Sum();
            return Math.Max(-1.0, Math.Min(1.0, dot / (firstNorm * secondNorm)));
        }
    }
}
