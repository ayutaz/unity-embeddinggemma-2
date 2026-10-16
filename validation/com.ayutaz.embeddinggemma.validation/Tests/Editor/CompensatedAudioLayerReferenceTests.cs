using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using EmbeddingGemma.Tests;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class CompensatedAudioLayerReferenceTests
    {
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        [Timeout(1200000)]
        public void CompareBoundNormOutputsWithoutChangingNormalFinalBits(BackendType backend)
        {
            var root = Path.GetFullPath(Path.Combine(Application.dataPath, "../../k/artifacts/audio-real-chain"));
            var contextRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "../../k/artifacts/audio-current-stage-context"));
            var output = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/norm1-reciprocal-scalar-zero-layer-results"));
            Directory.CreateDirectory(output);
            var path = Path.Combine(output, backend + ".json");
            var stages = new JArray();
            var report = new JObject { ["success"] = false, ["scope"] = "silence_norm1_reciprocal_scalar_zero_fp32_norm_outputs_to_native_layer_references",
                ["runtime_source"] = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE"), ["reference_source"] = "4edbe17a8105233c5aaa1514faa2862b08a4f7e0",
                ["requested_backend"] = backend.ToString(), ["runtime_overlay"] = false, ["stages"] = stages,
                ["projection_chunk"] = 64, ["norm_chunk"] = 1,
                ["full_encoder_validated"] = false, ["end_to_end_validated"] = false, ["started_utc"] = DateTime.UtcNow.ToString("O") };
            void Save() => File.WriteAllText(path, report.ToString());
            Save();
            try
            {
                var locked = JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath, "../Packages/packages-lock.json")));
                Assert.That(locked["dependencies"]["com.ayutaz.embeddinggemma"].Value<string>("hash"), Is.EqualTo(report.Value<string>("runtime_source")));
                HashCheck(Path.Combine(contextRoot, "audio-stage-reference.json"), "6620f335563a91ccc484df7d9b89d4d889e58dab4ec3d7d0389b36abe36b454a");
                HashCheck(Path.Combine(contextRoot, "audio-stage-validation.json"), "3edee8acdd9ebebb819ae488b753a3a94949b953aa0906995e516764932b0f9b");
                HashCheck(Path.Combine(root, "audio-runtime-reference/audio-encoder-reference.json"), "ac23f39904b8149dd5ccb6485afcc93d96047f0977b5faf1d640c8bd018c13c9");
                var reference = JObject.Parse(File.ReadAllText(Path.Combine(contextRoot, "audio-stage-reference.json")));
                var audit = JObject.Parse(File.ReadAllText(Path.Combine(contextRoot, "audio-stage-validation.json")));
                var anchors = JObject.Parse(File.ReadAllText(Path.Combine(contextRoot, "audio-norm-anchors.json")));
                Assert.That(anchors.Value<string>("reference_source"), Is.EqualTo(report.Value<string>("reference_source")));
                Assert.That(reference["metadata"].Value<string>("encoder_reference_sha256"), Is.EqualTo("ac23f39904b8149dd5ccb6485afcc93d96047f0977b5faf1d640c8bd018c13c9"));
                var graphPath = Path.Combine(root, "audio-runtime-reference/frames-1119-valid-1119/audio.onnx");
                HashCheck(graphPath, anchors.Value<string>("original_graph_sha256"));
                var cache = JObject.Parse(File.ReadAllText(Path.Combine(root, "frames-1119-valid-1119-model-cache.json")));
                Assert.That(cache["context"].Value<string>("graph_sha256"), Is.EqualTo(anchors.Value<string>("original_graph_sha256")));
                var modelPath = Path.GetFullPath(Path.Combine(root, cache["files"]["fp32"].Value<string>("name")));
                Assert.That(modelPath.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.Ordinal), Is.True);
                HashCheck(modelPath, "3214cc2b179a3abf3f968816c58b83b8ee762661bfab7b58c6f142d48ed3ed3e");
                var model = TextModelFile.Load(modelPath);
                var map = AudioCoefficientLayerMap.Resolve(model, (JArray)anchors["anchors"]);
                model = CompensatedModel.CreateWithCompensatedNorms(model, 64, 1);
                Assert.That(map.Count, Is.EqualTo(12));
                report["coefficient_bound_outputs"] = JObject.FromObject(map); Save();
                var row = ((JArray)reference["cases"]).Single(value => value.Value<string>("id") == "silence");
                var natives = new Dictionary<string, float[]>();
                foreach (var stage in ((JArray)row["stages"]).Where(value => map.ContainsKey(value.Value<string>("name"))))
                {
                    var name = stage.Value<string>("expected"); var file = Path.Combine(contextRoot, Path.GetFileName(name));
                    HashCheck(file, audit["files"][name].Value<string>("sha256"));
                    natives.Add(stage.Value<string>("name"), Floats(file));
                }
                Assert.That(natives.Count, Is.EqualTo(12));
                var inputPath = Path.Combine(root, "audio-runtime-reference/frames-1119-valid-1119/silence-features-fp32.bin");
                HashCheck(inputPath, "366de71cb4e45e4feda58a605babe4ea9d8693a7e37e84eac0735a7c42d0ad24");
                using var input = new Tensor<float>(new TensorShape(1, 1119, 128), Floats(inputPath));
                if (backend == BackendType.GPUCompute) { Assert.That(SystemInfo.supportsComputeShaders, Is.True); ComputeTensorData.Pin(input); }
                var observed = WorkerObservation.ObserveIndexes(model, backend, input, new HashSet<int>(map.Values));
                report["normal_final_matches_observed_bits"] = observed.FinalMatchesBaselineBits;
                report["diagnostic_readback_count"] = observed.Snapshots.Count + 2;
                Assert.That(observed.Snapshots.Count, Is.EqualTo(12));
                foreach (var anchor in (JArray)anchors["anchors"])
                {
                    var name = anchor.Value<string>("stage");
                    var snapshot = observed.Snapshots.Single(value => value.TensorIndex == map[name]);
                    var metrics = Compare(snapshot.Values, natives[name]);
                    metrics["stage"] = name; metrics["layer_index"] = snapshot.LayerIndex; metrics["tensor_index"] = snapshot.TensorIndex;
                    metrics["actual_backend"] = snapshot.Backend.ToString(); metrics["coefficient_sha256"] = anchor["coefficient_sha256"];
                    stages.Add(metrics); Save();
                    Assert.That(snapshot.Backend, Is.EqualTo(backend));
                }
                report["first_failed_stage"] = stages.FirstOrDefault(value => !value.Value<bool>("elementwise_passed"))?["stage"];
                Assert.That(observed.FinalMatchesBaselineBits, Is.True, "Diagnostic must preserve the original failed path.");
                report["success"] = true;
            }
            catch (Exception error) { report["error"] = error.ToString(); throw; }
            finally { report["completed_utc"] = DateTime.UtcNow.ToString("O"); Save(); }
        }
        static JObject Compare(float[] actual, float[] expected)
        {
            Assert.That(actual.Length, Is.EqualTo(expected.Length));
            int mismatches = 0; double maximum = 0, ratio = 0;
            for (var index = 0; index < actual.Length; index++)
            {
                Assert.That(float.IsNaN(actual[index]) || float.IsInfinity(actual[index]), Is.False);
                var difference = Math.Abs((double)actual[index] - expected[index]); var allowed = 1e-5 + 1e-4 * Math.Abs((double)expected[index]);
                maximum = Math.Max(maximum, difference); ratio = Math.Max(ratio, difference / allowed); if (difference > allowed) mismatches++;
            }
            return new JObject { ["elementwise_passed"] = mismatches == 0, ["mismatches"] = mismatches,
                ["maximum_absolute_error"] = maximum, ["maximum_tolerance_ratio"] = ratio };
        }
        static float[] Floats(string path) { var bytes = File.ReadAllBytes(path); var values = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length); return values; }
        static void HashCheck(string path, string expected)
        {
            using var stream = File.OpenRead(path); Assert.That(PlayerFileHash.Compute(stream, out _), Is.EqualTo(expected));
        }
    }
}
