using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    // Real text weights and synthetic projected features. Real encoders / prompts
    // and GPU preprocessing remain independent end-to-end acceptance gates.
    public sealed class MultimodalBackboneReferenceTests
    {
        static string DirectoryPath => Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/multimodal-backbone"));
        JObject reference;
        JObject report;
        JArray conditions;
        MultimodalBackboneProfile profile;
        readonly BackboneImportCache<ModelAsset> imported = new BackboneImportCache<ModelAsset>();
        readonly Dictionary<string, string> saved = new Dictionary<string, string>();

        [OneTimeSetUp]
        public void AuditCompletedArtifactAndPinnedConfiguration()
        {
            saved.Clear(); imported.Clear(); conditions = new JArray();
            report = new JObject { ["success"] = false, ["stage"] = "sentis_projected_features_backbone_only",
                ["encoders_validated"] = false, ["end_to_end_validated"] = false,
                ["gpu_preprocessing_validated"] = false, ["expected_conditions"] = 4,
                ["unity"] = Application.unityVersion, ["gpu"] = SystemInfo.graphicsDeviceName,
                ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(),
                ["started_utc"] = DateTime.UtcNow.ToString("O"), ["conditions"] = conditions };
            Save();
            var expectedSource = File.ReadAllText(Path.Combine(DirectoryPath, "expected-source.txt")).Trim();
            Assert.That(expectedSource, Does.Match("^[0-9a-f]{40}$"));
            var validation = JObject.Parse(File.ReadAllText(Path.Combine(DirectoryPath, "backbone-validation.json")));
            Assert.That(validation.Value<bool>("success"), Is.True);
            Assert.That(validation.Value<string>("stage"), Is.EqualTo("projected_features_backbone_only"));
            Assert.That(validation.Value<int>("case_count"), Is.EqualTo(2));
            foreach (var file in (JObject)validation["files"])
            {
                var path = Path.GetFullPath(Path.Combine(DirectoryPath, file.Key));
                Assert.That(path.StartsWith(DirectoryPath + Path.DirectorySeparatorChar, StringComparison.Ordinal), Is.True);
                Assert.That(new FileInfo(path).Length, Is.EqualTo(file.Value.Value<long>("bytes")), file.Key);
                Assert.That(Hash(path), Is.EqualTo(file.Value.Value<string>("sha256")), file.Key);
            }
            reference = JObject.Parse(File.ReadAllText(Path.Combine(DirectoryPath, "backbone-reference.json")));
            var metadata = reference["metadata"];
            Assert.That(metadata.Value<string>("source_commit"), Is.EqualTo(expectedSource));
            Assert.That(metadata.Value<string>("model_revision"), Is.EqualTo("914f7f89142e33e77833254d9c9b90c3cef7303b"));
            Assert.That(metadata.Value<int>("sequence_length"), Is.EqualTo(1024));
            Assert.That(metadata.Value<int>("feature_width"), Is.EqualTo(512));
            Assert.That(metadata.Value<int>("embedding_dimension"), Is.EqualTo(768));
            Assert.That(metadata.Value<int>("image_feature_count"), Is.EqualTo(280));
            Assert.That(metadata.Value<int>("audio_feature_count"), Is.EqualTo(280));
            // The existing artifact records the original config hash but does
            // not bundle config.json. Stage that same pinned file separately.
            var configPath = Path.Combine(DirectoryPath, "config.json");
            Assert.That(Hash(configPath), Is.EqualTo(metadata["input_sha256"].Value<string>("config.json")));
            var config = JObject.Parse(File.ReadAllText(configPath))["text_config"];
            profile = new MultimodalBackboneProfile(1024, metadata.Value<int>("image_token_id"), metadata.Value<int>("audio_token_id"),
                280, 280, config.Value<int>("vocab_size"), config.Value<int>("pad_token_id"));
            report["reference_source"] = expectedSource;
            report["config_sha256"] = Hash(configPath);
            Save();
        }

        [OneTimeTearDown]
        public void CompleteReport()
        {
            if (report == null) return;
            report["success"] = conditions.Count == 4 && conditions.All(row => row.Value<bool>("success"));
            report["completed_utc"] = DateTime.UtcNow.ToString("O");
            Save();
            imported.Clear(); saved.Clear();
        }

        [TestCase("fp32", BackendType.CPU)]
        [TestCase("fp32", BackendType.GPUCompute)]
        [TestCase("float16", BackendType.CPU)]
        [TestCase("float16", BackendType.GPUCompute)]
        [Timeout(1200000)]
        public void SavedRealBackboneMatchesBothIndependentLongFeatureCases(string precision, BackendType backend)
        {
            var condition = new JObject { ["precision"] = precision, ["requested_backend"] = backend.ToString(),
                ["success"] = false, ["phase"] = "prepare_saved_model" };
            conditions.Add(condition); Save();
            try
            {
                if (backend == BackendType.GPUCompute)
                    Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Actual GPU is required, never skip or substitute CPU.");
                var path = SavedModel(precision);
                condition["saved_sha256"] = Hash(path);
                condition["saved_bytes"] = new FileInfo(path).Length;
                condition["phase"] = "create_worker"; Save();
                var model = TextModelFile.Load(path);
                using var backbone = new MultimodalBackbone(model, profile, backend);
                var worker = (Worker)typeof(MultimodalBackbone).GetField("worker", BindingFlags.NonPublic | BindingFlags.Instance).GetValue(backbone);
                condition["actual_worker_backend"] = worker.backendType.ToString();
                var resident = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", BindingFlags.NonPublic | BindingFlags.Instance).GetValue(worker);
                var produced = model.layers.SelectMany(layer => layer.outputs);
                var featureInputs = model.inputs.Where(input => input.name.EndsWith("_features", StringComparison.Ordinal)).Select(input => input.index);
                var cpu = resident.Intersect(produced.Concat(featureInputs)).OrderBy(index => index).ToArray();
                var flush = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", BindingFlags.NonPublic | BindingFlags.Instance).GetValue(worker);
                condition["cpu_computation_or_feature_input_indices"] = new JArray(cpu);
                condition["cpu_readback_flush_indices"] = new JArray(flush.OrderBy(index => index));
                if (backend == BackendType.GPUCompute)
                {
                    Assert.That(cpu, Is.Empty, "No CPU computation or feature readback is permitted.");
                    Assert.That(flush, Is.Empty);
                }
                double minimumCosine = 1;
                foreach (var row in (JArray)reference["cases"])
                {
                    condition["phase"] = "evaluate_" + row.Value<string>("id"); Save();
                    using var image = new Tensor<float>(new TensorShape(280, 512), row["image_features"].ToObject<float[][]>().SelectMany(values => values).ToArray());
                    using var audio = new Tensor<float>(new TensorShape(280, 512), row["audio_features"].ToObject<float[][]>().SelectMany(values => values).ToArray());
                    if (backend == BackendType.GPUCompute) { ComputeTensorData.Pin(image); ComputeTensorData.Pin(audio); }
                    var actual = backbone.Embed(row["input_ids"].ToObject<int[]>(), row["attention_mask"].ToObject<int[]>(), image, audio);
                    condition["actual_output_backend"] = worker.PeekOutput().backendType.ToString();
                    Assert.That(worker.PeekOutput().backendType, Is.EqualTo(backend));
                    Assert.That(image.backendType, Is.EqualTo(backend));
                    Assert.That(audio.backendType, Is.EqualTo(backend));
                    var expected = row["embedding"].ToObject<float[]>();
                    Assert.That(expected, Has.Length.EqualTo(768));
                    double dot = 0, left = 0, right = 0;
                    for (var index = 0; index < 768; index++)
                    {
                        dot += (double)actual[index] * expected[index];
                        left += (double)actual[index] * actual[index];
                        right += (double)expected[index] * expected[index];
                    }
                    Assert.That(left * right, Is.GreaterThan(0));
                    minimumCosine = Math.Min(minimumCosine, dot / Math.Sqrt(left * right));
                }
                condition["minimum_cosine"] = minimumCosine;
                Assert.That(minimumCosine, Is.GreaterThanOrEqualTo(precision == "fp32" ? 0.999 : 0.99));
                condition["phase"] = "completed"; condition["success"] = true;
            }
            catch (Exception error) { condition["error"] = error.ToString(); throw; }
            finally { Save(); }
        }

        string SavedModel(string precision)
        {
            if (saved.TryGetValue(precision, out var existing)) return existing;
            var asset = imported.Get("audited-backbone", () =>
            {
                const string assetPath = "Assets/BackboneGenerated/backbone.pt2";
                var source = Path.Combine(DirectoryPath, "backbone.pt2");
                Directory.CreateDirectory(Path.GetDirectoryName(assetPath));
                if (!File.Exists(assetPath) || Hash(assetPath) != Hash(source)) File.Copy(source, assetPath, true);
                AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceSynchronousImport);
                var value = AssetDatabase.LoadAssetAtPath<ModelAsset>(assetPath);
                Assert.That(value, Is.Not.Null, "Sentis must import the real graph.");
                return value;
            });
            var model = ModelLoader.Load(asset);
            if (precision == "float16") ModelQuantizer.QuantizeWeights(QuantizationType.Float16, ref model);
            var path = Path.Combine(DirectoryPath, precision + ".sentis");
            TextModelFile.Save(model, path);
            saved.Add(precision, path);
            model = null; GC.Collect(); GC.WaitForPendingFinalizers();
            return path;
        }

        void Save()
        {
            var destination = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/multimodal-backbone-results"));
            Directory.CreateDirectory(destination);
            File.WriteAllText(Path.Combine(destination, "results.json"), report.ToString(Formatting.Indented));
        }

        static string Hash(string path)
        {
            var existing = AppDomain.CurrentDomain.GetAssemblies().Select(assembly => assembly.GetType("EmbeddingGemma.Samples.TextSearchFileHash"))
                .FirstOrDefault(type => type != null);
            if (existing != null) return (string)existing.GetMethod("Compute").Invoke(null, new object[] { path, null });
            using var stream = File.OpenRead(path); using var hash = SHA256.Create();
            return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }
    }
}
