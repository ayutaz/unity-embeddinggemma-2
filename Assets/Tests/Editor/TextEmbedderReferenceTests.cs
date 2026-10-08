using System;
using System.IO;
using System.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class TextEmbedderReferenceTests
    {
        [Serializable] sealed class ReferenceFile { public Metadata metadata; public ReferenceCase[] cases; }
        [Serializable] sealed class Metadata { public string model_revision; public string source_commit; }
        [Serializable] sealed class ReferenceCase
        {
            public string id;
            public string role;
            public string text;
            public string title;
            public string formatted_text;
            public float[] embedding;
        }

        [TestCase(BackendType.CPU)]
        [TestCase(BackendType.GPUCompute)]
        public void PublicApiMatchesAllPinnedCasesAndReusesWorker(BackendType backend)
        {
            var directory = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/m1"));
            Assert.That(File.Exists(Path.Combine(directory, "reference.json")), Is.True, "Stage verified CI reference first.");
            var reference = JsonUtility.FromJson<ReferenceFile>(File.ReadAllText(Path.Combine(directory, "reference.json")));
            Assert.That(reference.metadata.model_revision, Is.EqualTo("914f7f89142e33e77833254d9c9b90c3cef7303b"));
            Assert.That(reference.cases, Has.Length.EqualTo(15));
            Debug.Log($"API environment: Unity={Application.unityVersion}, OS={SystemInfo.operatingSystem}, " +
                      $"backend={backend}, graphics={SystemInfo.graphicsDeviceName}, API={SystemInfo.graphicsDeviceType}, " +
                      $"compute={SystemInfo.supportsComputeShaders}, source_commit={reference.metadata.source_commit}");
            if (backend == BackendType.GPUCompute)
                Assert.That(SystemInfo.supportsComputeShaders, Is.True, "GPUCompute must execute without skip or CPU replacement.");
            var asset = AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/M1Generated/model.pt2");
            Assert.That(asset, Is.Not.Null, "The real model must import successfully.");
            using var embedder = new TextEmbedder(ModelLoader.Load(asset), File.ReadAllText(Path.Combine(directory, "tokenizer.json")), backend);
            Assert.That(embedder.Backend, Is.EqualTo(backend));
            float[] first = null;
            var minimum = 1.0;
            foreach (var item in reference.cases)
            {
                TextRole role;
                float[] actual;
                switch (item.role)
                {
                    case "query": role = TextRole.Query; actual = embedder.EmbedQuery(item.text); break;
                    case "document": role = TextRole.Document; actual = embedder.EmbedDocument(item.text, item.title); break;
                    case "raw": role = TextRole.Raw; actual = embedder.EmbedRaw(item.text); break;
                    default: throw new AssertionException("Unknown reference role: " + item.role);
                }
                Assert.That(TextPrompts.Format(item.text, role, item.title), Is.EqualTo(item.formatted_text), item.id);
                Assert.That(actual, Has.Length.EqualTo(768), item.id);
                Assert.That(actual.All(value => !float.IsNaN(value) && !float.IsInfinity(value)), Is.True, item.id);
                var norm = Math.Sqrt(actual.Sum(value => (double)value * value));
                Assert.That(norm, Is.EqualTo(1).Within(1e-3), item.id);
                var expectedNorm = Math.Sqrt(item.embedding.Sum(value => (double)value * value));
                var cosine = actual.Zip(item.embedding, (a, b) => (double)a * b).Sum() / (norm * expectedNorm);
                Assert.That(cosine, Is.GreaterThanOrEqualTo(0.999), item.id);
                minimum = Math.Min(minimum, cosine);
                first ??= actual.ToArray();
                Debug.Log($"API {backend} {item.id}: cosine={cosine:R}");
            }
            var repeated = embedder.EmbedQuery(reference.cases[0].text);
            Assert.That(repeated, Is.EqualTo(first).Within(1e-5), "Repeated input after other requests");
            Debug.Log($"API {backend}: cases=15, min_cosine={minimum:R}, model_revision={reference.metadata.model_revision}");
        }
    }
}
