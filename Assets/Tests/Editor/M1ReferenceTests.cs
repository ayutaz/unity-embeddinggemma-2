using System;
using System.IO;
using System.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using Unity.InferenceEngine.Tokenization.Parsers.HuggingFace;
using UnityEditor;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    // Acceptance tests written before the Unity runtime API. Not yet executed in an Editor.
    public sealed class M1ReferenceTests
    {
        [Serializable]
        sealed class ReferenceFile
        {
            public int schema_version;
            public Metadata metadata;
            public ReferenceCase[] cases;
        }

        [Serializable]
        sealed class Metadata
        {
            public string model_id;
            public string model_revision;
            public string source_commit;
            public int sequence_length;
            public int embedding_dimension;
        }

        [Serializable]
        sealed class ReferenceCase
        {
            public string id;
            public string formatted_text;
            public int[] input_ids;
            public int[] attention_mask;
            public float[] embedding;
        }

        static string ArtifactDirectory => Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/m1"));

        static ReferenceFile ReadReference()
        {
            var path = Path.Combine(ArtifactDirectory, "reference.json");
            Assert.That(File.Exists(path), Is.True, "Run the model-reference CI job before Unity validation.");
            var reference = JsonUtility.FromJson<ReferenceFile>(File.ReadAllText(path));
            Assert.That(reference.schema_version, Is.EqualTo(1));
            Assert.That(reference.metadata.model_id, Is.EqualTo("google/embeddinggemma-2"));
            Assert.That(reference.metadata.model_revision, Does.Match("^[0-9a-f]{40}$"));
            Assert.That(reference.metadata.embedding_dimension, Is.EqualTo(768), "A tiny test model cannot satisfy M1.");
            Assert.That(reference.cases, Has.Length.EqualTo(15), "Validate the whole fixed test suite.");
            return reference;
        }

        [Test, Category("M1CPU"), Category("M1GPUCompute")]
        public void TokenizerMatchesEveryReferenceCase()
        {
            var reference = ReadReference();
            var tokenizer = HuggingFaceParser.GetDefault().Parse(
                File.ReadAllText(Path.Combine(ArtifactDirectory, "tokenizer.json")));
            foreach (var item in reference.cases)
            {
                var encoded = tokenizer.Encode(item.formatted_text);
                Assert.That(encoded.GetIds(), Is.EqualTo(item.input_ids), item.id + " token IDs");
                Assert.That(encoded.GetAttentionMask(), Is.EqualTo(item.attention_mask), item.id + " attention mask");
            }
        }

        [TestCase(BackendType.CPU, Category = "M1CPU")]
        [TestCase(BackendType.GPUCompute, Category = "M1GPUCompute")]
        public void EmbeddingMatchesEveryReferenceCase(BackendType backend)
        {
            var reference = ReadReference();
            Debug.Log($"M1 environment: Unity={Application.unityVersion}, OS={SystemInfo.operatingSystem}, " +
                      $"backend={backend}, graphics={SystemInfo.graphicsDeviceName}, " +
                      $"API={SystemInfo.graphicsDeviceType}, compute={SystemInfo.supportsComputeShaders}");
            if (backend == BackendType.GPUCompute)
                Assert.That(SystemInfo.supportsComputeShaders, Is.True,
                    "This runner cannot validate GPUCompute. A skipped or CPU fallback test is not a GPU pass.");

            var asset = AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/M1Generated/model.pt2");
            Assert.That(asset, Is.Not.Null, "Sentis must successfully import the exported model.");
            var model = ModelLoader.Load(asset);
            using var worker = new Worker(model, backend);
            Assert.That(worker.backendType, Is.EqualTo(backend));
            var tokenizer = HuggingFaceParser.GetDefault().Parse(
                File.ReadAllText(Path.Combine(ArtifactDirectory, "tokenizer.json")));
            var minimum = 1.0;
            foreach (var item in reference.cases)
            {
                var encoded = tokenizer.Encode(item.formatted_text);
                var ids = encoded.GetIds().ToArray();
                var mask = encoded.GetAttentionMask().ToArray();
                Assert.That(ids, Is.EqualTo(item.input_ids), item.id + " token IDs");
                Assert.That(mask, Is.EqualTo(item.attention_mask), item.id + " attention mask");
                using var inputIds = new Tensor<int>(new TensorShape(1, reference.metadata.sequence_length), ids);
                using var attentionMask = new Tensor<int>(new TensorShape(1, reference.metadata.sequence_length), mask);
                worker.SetInput("input_ids", inputIds);
                worker.SetInput("attention_mask", attentionMask);
                worker.Schedule();
                var output = worker.PeekOutput() as Tensor<float>;
                Assert.That(output, Is.Not.Null);
                Assert.That(output.shape, Is.EqualTo(new TensorShape(1, 768)), item.id);
                var actual = output.DownloadToArray();
                Assert.That(actual.All(value => !float.IsNaN(value) && !float.IsInfinity(value)), Is.True, item.id);
                var norm = Math.Sqrt(actual.Sum(value => (double)value * value));
                var referenceNorm = Math.Sqrt(item.embedding.Sum(value => (double)value * value));
                Assert.That(norm, Is.EqualTo(1.0).Within(1e-3), item.id + " L2 norm");
                var cosine = actual.Zip(item.embedding, (a, b) => (double)a * b).Sum() / (norm * referenceNorm);
                Assert.That(cosine, Is.GreaterThanOrEqualTo(0.999), item.id + " cosine");
                minimum = Math.Min(minimum, cosine);
                Debug.Log($"M1 {backend} {item.id}: cosine={cosine:R}");
            }
            Debug.Log($"M1 {backend}: cases={reference.cases.Length}, min_cosine={minimum:R}, " +
                      $"model_revision={reference.metadata.model_revision}, source_commit={reference.metadata.source_commit}");
        }
    }
}
