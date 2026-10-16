using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class MultimodalSearchIndexTests
    {
        sealed class Queries : ITextEmbedder
        {
            public BackendType Backend { get; set; } = BackendType.GPUCompute;
            public float[] Vector = Unit(0);
            public int Calls;
            public float[] EmbedQuery(string text) { Calls++; return Vector; }
            public float[] EmbedDocument(string text, string title = null) => throw new InvalidOperationException("Media vectors must come from their own encoders.");
            public void Dispose() { }
        }
        static float[] Unit(int axis) { var v = new float[768]; v[axis] = 1; return v; }
        static Type Required()
        {
            var type = typeof(TextSearchIndex).Assembly.GetType("EmbeddingGemma.MultimodalSearchIndex");
            Assert.That(type, Is.Not.Null, "A GPU-query index for actual image/audio embeddings is required.");
            return type;
        }
        static object Create(SearchDocument[] documents, float[][] vectors, ITextEmbedder queries)
            => Activator.CreateInstance(Required(), new object[] { documents, vectors, queries });
        static SearchHit[] Search(object index, string query = "猫の写真と鳴き声", int limit = 10)
            => (SearchHit[])Required().GetMethod("Search").Invoke(index, new object[] { query, limit });
        static SearchDocument[] Documents() => new[] { new SearchDocument("image-cat", "cat.png", "猫の写真"), new SearchDocument("audio-bird", "bird.wav", "鳥の声") };

        [Test]
        public void ImageAndAudioVectorsAreRankedByTextQueryWithoutReembeddingTheirPaths()
        {
            var queries = new Queries(); var index = Create(Documents(), new[] { Unit(0), Unit(1) }, queries);
            var hits = Search(index);
            Assert.That(hits.Select(h => h.Document.Id), Is.EqualTo(new[] { "image-cat", "audio-bird" }));
            Assert.That(hits.Select(h => h.Score), Is.EqualTo(new[] { 1d, 0d }));
            Assert.That(hits[0].Document.Text, Is.EqualTo("cat.png"));
            Assert.That(queries.Calls, Is.EqualTo(1));
        }
        [Test]
        public void StoredVectorsAreOwnedCopiesAndTiesUseOrdinalDocumentIds()
        {
            var vectors = new[] { Unit(0), Unit(0) }; var queries = new Queries();
            var index = Create(Documents(), vectors, queries); vectors[0][0] = -1;
            Assert.That(Search(index).Select(h => h.Document.Id), Is.EqualTo(new[] { "audio-bird", "image-cat" }));
            Assert.That(Search(index, limit: 1).Length, Is.EqualTo(1));
        }
        [TestCase("dimension")] [TestCase("nan")] [TestCase("zero")] [TestCase("not_normalized")]
        public void InvalidMediaVectorsCannotBecomeSearchable(string problem)
        {
            var bad = Unit(0);
            if (problem == "dimension") bad = new float[767];
            if (problem == "nan") bad[1] = float.NaN;
            if (problem == "zero") bad[0] = 0;
            if (problem == "not_normalized") bad[0] = 2;
            Assert.That(Assert.Throws<TargetInvocationException>(() => Create(Documents(), new[] { Unit(0), bad }, new Queries())).InnerException, Is.InstanceOf<ArgumentException>());
        }
        [TestCase("count")] [TestCase("duplicate")] [TestCase("empty")]
        public void CorpusAndVectorRowsMustCorrespondExactly(string problem)
        {
            var documents = Documents(); var vectors = new[] { Unit(0), Unit(1) };
            if (problem == "count") vectors = new[] { Unit(0) };
            if (problem == "duplicate") documents[1] = documents[0];
            if (problem == "empty") { documents = Array.Empty<SearchDocument>(); vectors = Array.Empty<float[]>(); }
            Assert.That(Assert.Throws<TargetInvocationException>(() => Create(documents, vectors, new Queries())).InnerException, Is.InstanceOf<ArgumentException>());
        }
        [Test]
        public void CpuQueryProviderIsRejectedRatherThanSilentlyUsed()
            => Assert.That(Assert.Throws<TargetInvocationException>(() => Create(Documents(), new[] { Unit(0), Unit(1) }, new Queries { Backend = BackendType.CPU })).InnerException, Is.InstanceOf<ArgumentException>());
        [TestCase("")] [TestCase(" ")]
        public void BlankQueryIsRejectedBeforeInference(string query)
        {
            var queries = new Queries(); var index = Create(Documents(), new[] { Unit(0), Unit(1) }, queries);
            Assert.That(Assert.Throws<TargetInvocationException>(() => Search(index, query)).InnerException, Is.InstanceOf<ArgumentException>());
            Assert.That(queries.Calls, Is.EqualTo(0));
        }
        [Test]
        public void InvalidQueryVectorAndLimitAreRejected()
        {
            var queries = new Queries(); var index = Create(Documents(), new[] { Unit(0), Unit(1) }, queries);
            Assert.That(Assert.Throws<TargetInvocationException>(() => Search(index, limit: 0)).InnerException, Is.InstanceOf<ArgumentOutOfRangeException>());
            Assert.That(queries.Calls, Is.EqualTo(0)); queries.Vector = new float[768];
            Assert.That(Assert.Throws<TargetInvocationException>(() => Search(index)).InnerException, Is.InstanceOf<ArgumentException>());
        }
    }
}
