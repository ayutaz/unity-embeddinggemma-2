using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    // Small export compatibility probe, never real-model acceptance evidence.
    public sealed class MultimodalPlacementReferenceTests
    {
        static string DirectoryPath => Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/backbone-placement-probe"));
        JObject reference;
        Model model;

        [OneTimeSetUp]
        public void ImportSmallPlacementGraph()
        {
            reference = JObject.Parse(File.ReadAllText(Path.Combine(DirectoryPath, "reference.json")));
            Assert.That(reference.Value<string>("stage"), Is.EqualTo("tiny_feature_placement_import_probe"));
            const string assetPath = "Assets/BackboneGenerated/placement-probe.pt2";
            Directory.CreateDirectory(Path.GetDirectoryName(assetPath));
            File.Copy(Path.Combine(DirectoryPath, "probe.pt2"), assetPath, true);
            AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceSynchronousImport);
            var asset = AssetDatabase.LoadAssetAtPath<ModelAsset>(assetPath);
            Assert.That(asset, Is.Not.Null, "The small feature placement graph must import before scheduling.");
            var loaded = ModelLoader.Load(asset);
            var path = Path.Combine(DirectoryPath, "probe.sentis");
            TextModelFile.Save(loaded, path);
            model = TextModelFile.Load(path);
        }

        [TestCase(BackendType.CPU)]
        [TestCase(BackendType.GPUCompute)]
        public void SavedPlacementPreservesChangedSlotsFeaturesAndPadding(BackendType backend)
        {
            if (backend == BackendType.GPUCompute) Assert.That(SystemInfo.supportsComputeShaders, Is.True);
            using var worker = new Worker(model, backend);
            if (backend == BackendType.GPUCompute)
            {
                var resident = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", BindingFlags.NonPublic | BindingFlags.Instance).GetValue(worker);
                Assert.That(resident.Intersect(model.layers.SelectMany(layer => layer.outputs)), Is.Empty);
                var flush = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", BindingFlags.NonPublic | BindingFlags.Instance).GetValue(worker);
                Assert.That(flush, Is.Empty);
            }
            foreach (var row in (JArray)reference["cases"])
            {
                using var ids = new Tensor<int>(new TensorShape(1, 6), row["input_ids"].ToObject<int[]>());
                using var mask = new Tensor<int>(new TensorShape(1, 6), row["attention_mask"].ToObject<int[]>());
                using var image = new Tensor<float>(new TensorShape(2, 16), row["image_features"].ToObject<float[][]>().SelectMany(values => values).ToArray());
                using var audio = new Tensor<float>(new TensorShape(1, 16), row["audio_features"].ToObject<float[][]>().SelectMany(values => values).ToArray());
                if (backend == BackendType.GPUCompute) { ComputeTensorData.Pin(image); ComputeTensorData.Pin(audio); }
                worker.SetInput("input_ids", ids); worker.SetInput("attention_mask", mask);
                worker.SetInput("image_features", image); worker.SetInput("audio_features", audio);
                worker.Schedule();
                var tensor = worker.PeekOutput() as Tensor<float>;
                Assert.That(tensor, Is.Not.Null);
                Assert.That(tensor.shape, Is.EqualTo(new TensorShape(1, 24)));
                Assert.That(tensor.backendType, Is.EqualTo(backend));
                var actual = tensor.DownloadToArray();
                var expected = row["embedding"].ToObject<float[]>();
                Assert.That(expected, Has.Length.EqualTo(24));
                for (var index = 0; index < actual.Length; index++)
                    Assert.That(actual[index], Is.EqualTo(expected[index]).Within(1e-5 + 1e-4 * Math.Abs(expected[index])), row.Value<string>("id") + "/" + index);
                Assert.That(image.backendType, Is.EqualTo(backend));
                Assert.That(audio.backendType, Is.EqualTo(backend));
            }
        }
    }
}
