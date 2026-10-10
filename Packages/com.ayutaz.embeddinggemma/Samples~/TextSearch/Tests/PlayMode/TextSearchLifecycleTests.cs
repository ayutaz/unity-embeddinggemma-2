using System.Collections;
using System;
using System.IO;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Samples.Tests
{
    public sealed class TextSearchLifecycleTests
    {
        sealed class Provider : ITextEmbedder
        {
            public int Disposals;
            public BackendType Backend => BackendType.CPU;
            public float[] EmbedQuery(string text) => Vector();
            public float[] EmbedDocument(string text, string title = null) => Vector();
            static float[] Vector() { var result = new float[768]; result[0] = 1; return result; }
            public void Dispose() { Disposals++; }
        }

        [UnityTest]
        public IEnumerator DisablingTheObjectReleasesItsWorkerInPlayMode()
        {
            var owner = new GameObject("Text Search Lifecycle");
            var corpus = new TextAsset(@"{""documents"":[{""id"":""a"",""text"":""猫""}]}");
            var provider = new Provider();
            try
            {
                var sample = owner.AddComponent<TextSearchSample>();
                sample.Corpus = corpus; sample.Backend = BackendType.CPU;
                Assert.That(sample.PrepareDocuments(() => provider), Is.True);
                yield return null;
                owner.SetActive(false);
                Assert.That(sample.IsReady, Is.False);
                Assert.That(provider.Disposals, Is.EqualTo(1));
            }
            finally { UnityEngine.Object.Destroy(owner); UnityEngine.Object.Destroy(corpus); }
            yield return null;
            Assert.That(provider.Disposals, Is.EqualTo(1));
        }
        [UnityTest] public IEnumerator DisablingDuringTransferDisposesItAndPreventsLatePreparation()
        {
            var root = Path.Combine(Path.GetTempPath(), "embeddinggemma-play-cache-" + Guid.NewGuid().ToString("N"));
            var owner = new GameObject("Sample preparation lifecycle"); var disposed = false; var calls = 0;
            IEnumerator Wait(string uri, string path) { try { while (true) yield return null; } finally { disposed = true; } }
            try
            {
                var sample = owner.AddComponent<TextSearchSample>(); sample.ModelPath = "jar:file:///sample.apk!/assets/EmbeddingGemmaTextSearch/model-fp32.sentis";
                sample.TokenizerPath = "jar:file:///sample.apk!/assets/EmbeddingGemmaTextSearch/tokenizer.json";
                sample.StartCoroutine(sample.PrepareDocumentsFromStreamingAssets(root, Wait, (_, _) => { calls++; return new Provider(); }));
                yield return null; Assert.That(sample.IsPreparing, Is.True);
                owner.SetActive(false); yield return null;
                Assert.That(disposed, Is.True); Assert.That(calls, Is.Zero); Assert.That(sample.IsPreparing, Is.False); Assert.That(sample.IsReady, Is.False);
                using var lease = new FileStream(Path.Combine(root, ".lock"), FileMode.Open, FileAccess.ReadWrite, FileShare.None);
            }
            finally { UnityEngine.Object.Destroy(owner); if (Directory.Exists(root)) Directory.Delete(root, true); }
            yield return null;
        }
    }
}
