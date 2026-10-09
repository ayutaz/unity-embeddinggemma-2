using System.Collections;
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
            finally { Object.Destroy(owner); Object.Destroy(corpus); }
            yield return null;
            Assert.That(provider.Disposals, Is.EqualTo(1));
        }
    }
}
