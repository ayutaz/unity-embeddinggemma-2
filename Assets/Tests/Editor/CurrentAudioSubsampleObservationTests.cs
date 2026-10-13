using System;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class CurrentAudioSubsampleObservationTests
    {
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        [Timeout(1200000)]
        public void ObserveSavedModelWithoutChangingNormalFinalBits(BackendType backend)
        {
            var root = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/audio-real-chain"));
            var stageRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/audio-current-stage-context"));
            var output = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/audio-current-observation-results"));
            Directory.CreateDirectory(output);
            var reportPath = Path.Combine(output, backend + ".json");
            var report = new JObject { ["success"] = false, ["scope"] = "silence_saved_fp32_model_live_tensor_candidates_only",
                ["requested_backend"] = backend.ToString(), ["runtime_source"] = "adccab24dbcc3f5dd8c8ac79911a7604a6014021",
                ["reference_source"] = "4edbe17a8105233c5aaa1514faa2862b08a4f7e0", ["runtime_overlay"] = false,
                ["full_encoder_validated"] = false, ["end_to_end_validated"] = false, ["started_utc"] = DateTime.UtcNow.ToString("O") };
            void Save() => File.WriteAllText(reportPath, report.ToString());
            Save();
            try
            {
                var locked = JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath, "../Packages/packages-lock.json")));
                Assert.That(locked["dependencies"]["com.ayutaz.embeddinggemma"].Value<string>("hash"), Is.EqualTo(report.Value<string>("runtime_source")));
                CheckHash(Path.Combine(root, "audio-runtime-reference/audio-encoder-reference.json"), "ac23f39904b8149dd5ccb6485afcc93d96047f0977b5faf1d640c8bd018c13c9");
                CheckHash(Path.Combine(stageRoot, "audio-stage-reference.json"), "6620f335563a91ccc484df7d9b89d4d889e58dab4ec3d7d0389b36abe36b454a");
                CheckHash(Path.Combine(stageRoot, "audio-stage-validation.json"), "3edee8acdd9ebebb819ae488b753a3a94949b953aa0906995e516764932b0f9b");
                CheckHash(Path.Combine(stageRoot, "silence-stage-00-fp32.bin"), "dc54774840fed958a310c6092e5c5c3483de74be52df7914d2bd128a99192969");
                var reference = JObject.Parse(File.ReadAllText(Path.Combine(root, "audio-runtime-reference/audio-encoder-reference.json")));
                var stages = JObject.Parse(File.ReadAllText(Path.Combine(stageRoot, "audio-stage-reference.json")));
                Assert.That(stages["metadata"].Value<string>("source_commit"), Is.EqualTo(report.Value<string>("reference_source")));
                Assert.That(stages["metadata"].Value<string>("encoder_reference_sha256"), Is.EqualTo("ac23f39904b8149dd5ccb6485afcc93d96047f0977b5faf1d640c8bd018c13c9"));
                Assert.That(stages["profiles"]["frames-1119-valid-1119"].Value<string>("original_graph_sha256"), Is.EqualTo("7770a8af258f2de32c5587497b820596ea4e1ed7ec5382c3ee90eada2bec7e4a"));
                var row = ((JArray)reference["cases"]).Single(value => value.Value<string>("id") == "silence");
                var cache = JObject.Parse(File.ReadAllText(Path.Combine(root, "frames-1119-valid-1119-model-cache.json")));
                Assert.That(cache["context"].Value<string>("producer"), Is.EqualTo(report.Value<string>("reference_source")));
                Assert.That(cache["context"].Value<string>("unity"), Is.EqualTo(Application.unityVersion));
                Assert.That(cache["context"].Value<string>("sentis"), Is.EqualTo("2.6.1"));
                var modelPath = Path.GetFullPath(Path.Combine(root, cache["files"]["fp32"].Value<string>("name")));
                Assert.That(modelPath.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.Ordinal), Is.True);
                CheckHash(modelPath, "3214cc2b179a3abf3f968816c58b83b8ee762661bfab7b58c6f142d48ed3ed3e");
                var inputPath = Path.Combine(root, "audio-runtime-reference/" + row.Value<string>("input_features"));
                CheckHash(inputPath, "366de71cb4e45e4feda58a605babe4ea9d8693a7e37e84eac0735a7c42d0ad24");
                var model = TextModelFile.Load(modelPath);
                var signature = model.layers.Select(layer => layer.GetType().Name + ":" + string.Join(",", layer.outputs)).ToArray();
                using var input = new Tensor<float>(new TensorShape(1, 1119, 128), Floats(inputPath));
                if (backend == BackendType.GPUCompute) { Assert.That(SystemInfo.supportsComputeShaders, Is.True); ComputeTensorData.Pin(input); }
                var observed = WorkerObservation.Observe(model, backend, input, new TensorShape(1, 280, 1024), 16);
                report["normal_final_matches_observed_bits"] = observed.FinalMatchesBaselineBits;
                report["model_layers_unchanged"] = signature.SequenceEqual(model.layers.Select(layer => layer.GetType().Name + ":" + string.Join(",", layer.outputs)));
                report["unavailable_tensor_indexes"] = new JArray(observed.UnavailableTensorIndexes);
                report["normal_final_vs_native"] = Compare(observed.Baseline, row["features"].ToObject<float[][]>().SelectMany(value => value).ToArray());
                var native = Floats(Path.Combine(stageRoot, "silence-stage-00-fp32.bin"));
                var candidates = new JArray(); report["candidates"] = candidates;
                foreach (var snapshot in observed.Snapshots)
                {
                    var comparison = Compare(snapshot.Values, native);
                    comparison["layer_index"] = snapshot.LayerIndex; comparison["tensor_index"] = snapshot.TensorIndex;
                    comparison["layer_type"] = snapshot.LayerType; comparison["actual_backend"] = snapshot.Backend.ToString();
                    candidates.Add(comparison);
                }
                report["diagnostic_readback_count"] = observed.Snapshots.Count + 2;
                Assert.That(observed.Snapshots.Count, Is.GreaterThan(0));
                foreach (var snapshot in observed.Snapshots) Assert.That(snapshot.Backend, Is.EqualTo(backend));
                Assert.That(observed.FinalMatchesBaselineBits, Is.True, "Observation must not change the failed numerical path.");
                Assert.That(report.Value<bool>("model_layers_unchanged"), Is.True);
                report["success"] = true;
            }
            catch (Exception error) { report["error"] = error.ToString(); throw; }
            finally { report["completed_utc"] = DateTime.UtcNow.ToString("O"); Save(); }
        }
        static JObject Compare(float[] actual, float[] expected)
        {
            Assert.That(actual.Length, Is.EqualTo(expected.Length));
            var mismatches = 0; double maximum = 0, ratio = 0;
            for (var index = 0; index < actual.Length; index++)
            {
                Assert.That(float.IsNaN(actual[index]) || float.IsInfinity(actual[index]), Is.False);
                var difference = Math.Abs((double)actual[index] - expected[index]);
                var allowed = 1e-5 + 1e-4 * Math.Abs((double)expected[index]);
                maximum = Math.Max(maximum, difference); ratio = Math.Max(ratio, difference / allowed);
                if (difference > allowed) mismatches++;
            }
            return new JObject { ["elementwise_passed"] = mismatches == 0, ["mismatches"] = mismatches,
                ["maximum_absolute_error"] = maximum, ["maximum_tolerance_ratio"] = ratio };
        }
        static float[] Floats(string path) { var bytes = File.ReadAllBytes(path); var values = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length); return values; }
        static void CheckHash(string path, string expected)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(assembly => assembly.GetType("EmbeddingGemma.Validation.PlayerFileHash")).FirstOrDefault(value => value != null);
            Assert.That(type, Is.Not.Null); using var stream = File.OpenRead(path);
            Assert.That((string)type.GetMethod("Compute").Invoke(null, new object[] { stream, null, null }), Is.EqualTo(expected));
        }
    }
}
