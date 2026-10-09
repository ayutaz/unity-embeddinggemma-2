using System;
using System.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class TextSearchTests
    {
        sealed class FakeEmbedder : ITextEmbedder
        {
            public BackendType Backend => BackendType.CPU;
            public int Documents, Queries, Disposals;
            public bool Fail;
            public float[] Buffer = Vector(3);
            public float[] EmbedDocument(string text, string title = null)
            {
                Documents++;
                if (Fail) throw new InvalidOperationException("inference failed");
                Buffer[0] = text == "opposite" ? -3 : 3;
                return Buffer;
            }
            public float[] EmbedQuery(string text) { Queries++; return Vector(9); }
            public void Dispose() { Disposals++; }
        }

        static float[] Vector(float first)
        {
            var vector = new float[768]; vector[0] = first; return vector;
        }
        static SearchDocument[] Docs() => new[] {
            new SearchDocument("b", "same", "猫"), new SearchDocument("a", "same", "Cats"),
            new SearchDocument("c", "opposite") };

        [Test]
        public void PrecomputesDocumentsCopiesBuffersAndRanksWithDeterministicTies()
        {
            var embedder = new FakeEmbedder();
            var index = new TextSearchIndex(Docs(), embedder);
            Assert.That(embedder.Documents, Is.EqualTo(3));
            embedder.Buffer[0] = 0;
            var result = index.Search("猫");
            Assert.That(result.Select(hit => hit.Document.Id), Is.EqualTo(new[] { "a", "b", "c" }));
            Assert.That(result.Select(hit => hit.Score), Is.EqualTo(new[] { 1.0, 1.0, -1.0 }).Within(1e-12));
            Assert.That(index.Search("cats", 1).Single().Document.Id, Is.EqualTo("a"));
            Assert.That(embedder.Documents, Is.EqualTo(3));
            Assert.That(embedder.Queries, Is.EqualTo(2));
            Assert.That(embedder.Disposals, Is.Zero, "Index does not own its provider");
        }

        [Test]
        public void RejectsInvalidCorpusBeforeEmbeddingAndInvalidQueryBeforeInference()
        {
            var embedder = new FakeEmbedder();
            Assert.Throws<ArgumentException>(() => new TextSearchIndex(Array.Empty<SearchDocument>(), embedder));
            Assert.Throws<ArgumentException>(() => new TextSearchIndex(new[] { Docs()[0], Docs()[0] }, embedder));
            Assert.That(embedder.Documents, Is.Zero);
            var index = new TextSearchIndex(Docs(), embedder);
            Assert.Throws<ArgumentException>(() => index.Search(" \n "));
            Assert.Throws<ArgumentOutOfRangeException>(() => index.Search("cats", 0));
            Assert.That(embedder.Queries, Is.Zero);
            Assert.Throws<ArgumentException>(() => new SearchDocument("猫", "text"));
            Assert.Throws<ArgumentException>(() => new SearchDocument("a", " "));
        }

        [TestCase(0f)]
        [TestCase(float.NaN)]
        [TestCase(float.PositiveInfinity)]
        public void RejectsInvalidEmbeddings(float value)
        {
            var embedder = new FakeEmbedder { Buffer = Vector(value) };
            Assert.Throws<ArgumentException>(() => TextSearchIndex.Cosine(Vector(value), Vector(1)));
            Assert.Throws<ArgumentException>(() => TextSearchIndex.Cosine(new[] { 1f }, new[] { 1f, 0f }));
        }

        [TestCase("a\n")]
        [TestCase("A")]
        [TestCase("")]
        public void DocumentIdsMustMatchTheEntireLowercaseAsciiIdentifier(string id)
        {
            Assert.Throws<ArgumentException>(() => new SearchDocument(id, "text"));
        }

        [Test]
        public void SessionOwnsProviderAndDisposesOnFailureReplacementAndExit()
        {
            var first = new FakeEmbedder(); var replacement = new FakeEmbedder();
            var failed = new FakeEmbedder { Fail = true };
            var session = new TextSearchSession();
            Assert.Throws<InvalidOperationException>(() => session.Search("cats"));
            session.Prepare(Docs(), () => first);
            Assert.That(session.IsReady, Is.True);
            session.Prepare(Docs(), () => replacement);
            Assert.That(first.Disposals, Is.EqualTo(1));
            Assert.Throws<InvalidOperationException>(() => session.Prepare(Docs(), () => failed));
            Assert.That(replacement.Disposals, Is.EqualTo(1));
            Assert.That(failed.Disposals, Is.EqualTo(1));
            Assert.That(session.IsReady, Is.False);
            session.Prepare(Docs(), () => first);
            session.Dispose(); session.Dispose();
            Assert.That(first.Disposals, Is.EqualTo(2));
            Assert.Throws<ObjectDisposedException>(() => session.Prepare(Docs(), () => first));
            Assert.Throws<ObjectDisposedException>(() => session.Search("cats"));
        }
    }
}
