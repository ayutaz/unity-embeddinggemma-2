using System;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Samples.Tests
{
    public sealed class TextSearchSampleTests
    {
        sealed class FakeEmbedder : ITextEmbedder
        {
            public int Documents, Queries, Disposals;
            public bool FailQuery;
            public BackendType Backend => BackendType.CPU;
            public float[] EmbedDocument(string text, string title = null) { Documents++; return Vector(); }
            public float[] EmbedQuery(string text)
            {
                Queries++;
                if (FailQuery) throw new InvalidOperationException("query inference failed");
                return Vector();
            }
            public void Dispose() { Disposals++; }
            static float[] Vector() { var values = new float[768]; values[0] = 1; return values; }
        }

        GameObject owner;
        TextAsset corpus;
        TextSearchSample sample;

        [SetUp]
        public void CreateSample()
        {
            owner = new GameObject("TextSearchSampleTests");
            sample = owner.AddComponent<TextSearchSample>();
            corpus = new TextAsset(@"{""documents"":[{""id"":""b"",""text"":""猫""},{""id"":""a"",""text"":""cats""}]}");
            sample.Corpus = corpus;
            sample.Backend = BackendType.CPU;
        }

        [TearDown]
        public void DestroySample() { UnityEngine.Object.DestroyImmediate(owner); UnityEngine.Object.DestroyImmediate(corpus); }

        [Test]
        public void DisplaysPreparationSearchAndErrorsAndReleasesExplicitly()
        {
            var provider = new FakeEmbedder();
            Assert.That(sample.Search("cats"), Is.False);
            Assert.That(sample.Error, Is.Not.Empty);
            Assert.That(sample.PrepareDocuments(() => provider), Is.True);
            Assert.That(sample.IsReady, Is.True);
            Assert.That(sample.Status, Does.Contain("2"));
            Assert.That(provider.Documents, Is.EqualTo(2));
            Assert.That(sample.Search("猫"), Is.True);
            Assert.That(sample.Results[0].Document.Id, Is.EqualTo("a"));
            Assert.That(sample.Query, Is.EqualTo("猫"));
            Assert.That(sample.Search(" \n "), Is.False);
            Assert.That(sample.Error, Is.Not.Empty);
            Assert.That(sample.Results, Is.Empty);
            Assert.That(provider.Queries, Is.EqualTo(1));
            provider.FailQuery = true;
            Assert.That(sample.Search("cat"), Is.False);
            Assert.That(sample.Error, Does.Contain("query inference failed"));
            sample.ReleaseDocuments();
            Assert.That(sample.IsReady, Is.False);
            Assert.That(provider.Disposals, Is.EqualTo(1));
        }

        [Test]
        public void MissingFilesAndInvalidCorpusAreVisibleAndNeverReady()
        {
            sample.ModelPath = "missing-model.sentis";
            sample.TokenizerPath = "missing-tokenizer.json";
            Assert.That(sample.PrepareDocuments(), Is.False);
            Assert.That(sample.Error, Is.Not.Empty);
            Assert.That(sample.IsReady, Is.False);
            var invalid = new TextAsset(@"{""documents"":[]}");
            try
            {
                sample.Corpus = invalid;
                var calls = 0;
                Assert.That(sample.PrepareDocuments(() => { calls++; return new FakeEmbedder(); }), Is.False);
                Assert.That(calls, Is.Zero);
                Assert.That(sample.Error, Is.Not.Empty);
            }
            finally { UnityEngine.Object.DestroyImmediate(invalid); }
        }

        [Test]
        public void BackendMismatchIsAnErrorAndRepreparationReleasesPreviousWorker()
        {
            var first = new FakeEmbedder(); var second = new FakeEmbedder();
            Assert.That(sample.PrepareDocuments(() => first), Is.True);
            Assert.That(sample.PrepareDocuments(() => second), Is.True);
            Assert.That(first.Disposals, Is.EqualTo(1));
            sample.Backend = BackendType.GPUCompute;
            var mismatch = new FakeEmbedder();
            Assert.That(sample.PrepareDocuments(() => mismatch), Is.False);
            Assert.That(second.Disposals, Is.EqualTo(1));
            Assert.That(mismatch.Disposals, Is.EqualTo(1));
            Assert.That(mismatch.Documents, Is.Zero);
            Assert.That(sample.IsReady, Is.False);
            Assert.That(sample.Error, Is.Not.Empty);
        }
    }
}
