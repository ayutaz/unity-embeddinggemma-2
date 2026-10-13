using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class CurrentAudioLayerZeroNormTests
    {
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        [Timeout(1200000)]
        public void CompareNativeInternalNormsOnUnchangedSavedEncoder(BackendType backend)
        {
            var root = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/audio-real-chain"));
            var nativeRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/audio-native-layer-replay"));
            var output = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/audio-layer-zero-norm-results"));
            Directory.CreateDirectory(output);
            var stages = new JArray();
            var report = new JObject { ["success"] = false, ["observation_completed"] = false,
                ["scope"] = "silence_layer_zero_three_internal_norms",
                ["runtime_source"] = "adccab24dbcc3f5dd8c8ac79911a7604a6014021",
                ["reference_source"] = "4edbe17a8105233c5aaa1514faa2862b08a4f7e0",
                ["native_replay_source"] = "9800c52957478fafa1377e2caa215f9f2e1477b0",
                ["requested_backend"] = backend.ToString(), ["runtime_overlay"] = false,
                ["unity_version"] = Application.unityVersion, ["graphics_device"] = SystemInfo.graphicsDeviceName,
                ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(), ["stages"] = stages,
                ["full_encoder_validated"] = false, ["end_to_end_validated"] = false,
                ["started_utc"] = DateTime.UtcNow.ToString("O") };
            void Save() => File.WriteAllText(Path.Combine(output, backend + ".json"), report.ToString());
            Save();
            try
            {
                var locked = JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath, "../Packages/packages-lock.json")));
                Assert.That(locked["dependencies"]["com.ayutaz.embeddinggemma"].Value<string>("hash"), Is.EqualTo(report.Value<string>("runtime_source")));
                var replayPath = Path.Combine(nativeRoot, "audio-layer-replay-report.json");
                HashCheck(replayPath, "39d563e2c8f186f397240b225b5fbb42ea8d7b6321d178bd7120135c2b039150");
                var replay = JObject.Parse(File.ReadAllText(replayPath));
                Assert.That(replay.Value<bool>("success") && replay.Value<bool>("native_final_exact_match"), Is.True);
                Assert.That(replay.Value<string>("reference_source"), Is.EqualTo(report.Value<string>("reference_source")));
                Assert.That(replay.Value<string>("diagnostic_source"), Is.EqualTo(report.Value<string>("native_replay_source")));
                Assert.That(replay.Value<string>("profile"), Is.EqualTo("frames-1119-valid-1119"));
                Assert.That(replay.Value<string>("case"), Is.EqualTo("silence"));
                var anchors = JObject.Parse(File.ReadAllText(Path.Combine(nativeRoot, "audio-layer-zero-norm-anchors.json")));
                Assert.That(anchors.Value<string>("native_replay_report_sha256"), Is.EqualTo(Hash(replayPath)));
                HashCheck(Path.Combine(root, "audio-runtime-reference/frames-1119-valid-1119/audio.onnx"),
                    "7770a8af258f2de32c5587497b820596ea4e1ed7ec5382c3ee90eada2bec7e4a");
                var cache = JObject.Parse(File.ReadAllText(Path.Combine(root, "frames-1119-valid-1119-model-cache.json")));
                Assert.That(cache["context"].Value<string>("graph_sha256"), Is.EqualTo(anchors.Value<string>("original_graph_sha256")));
                var modelPath = Path.GetFullPath(Path.Combine(root, cache["files"]["fp32"].Value<string>("name")));
                Assert.That(modelPath.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.Ordinal), Is.True);
                HashCheck(modelPath, "3214cc2b179a3abf3f968816c58b83b8ee762661bfab7b58c6f142d48ed3ed3e");
                var model = TextModelFile.Load(modelPath);
                var map = AudioCoefficientLayerMap.Resolve(model, (JArray)anchors["anchors"]);
                Assert.That(map.Keys, Is.EquivalentTo(new[] { "norm_pre_attn", "norm_post_attn", "norm_out" }));
                report["coefficient_bound_outputs"] = JObject.FromObject(map); Save();
                var native = new Dictionary<string, float[]>();
                foreach (var name in map.Keys)
                {
                    var part = replay["parts"][name];
                    Assert.That(part["shape"].ToObject<int[]>(), Is.EqualTo(new[] { 1, 280, 1024 }));
                    var path = Path.Combine(nativeRoot, "audio-layer-replay", part.Value<string>("file"));
                    Assert.That(new FileInfo(path).Length, Is.EqualTo(part.Value<long>("bytes")));
                    HashCheck(path, part.Value<string>("sha256")); native[name] = Floats(path);
                }
                var inputPath = Path.Combine(root, "audio-runtime-reference/frames-1119-valid-1119/silence-features-fp32.bin");
                HashCheck(inputPath, "366de71cb4e45e4feda58a605babe4ea9d8693a7e37e84eac0735a7c42d0ad24");
                using var input = new Tensor<float>(new TensorShape(1, 1119, 128), Floats(inputPath));
                if (backend == BackendType.GPUCompute) { Assert.That(SystemInfo.supportsComputeShaders, Is.True); ComputeTensorData.Pin(input); }
                var observed = WorkerObservation.ObserveIndexes(model, backend, input, new HashSet<int>(map.Values));
                Assert.That(observed.Snapshots.Count, Is.EqualTo(3));
                Assert.That(observed.FinalMatchesBaselineBits, Is.True, "Observation changed original final bits.");
                report["normal_final_matches_observed_bits"] = observed.FinalMatchesBaselineBits;
                foreach (var anchor in (JArray)anchors["anchors"])
                {
                    var name = anchor.Value<string>("stage");
                    var snapshot = observed.Snapshots.Single(value => value.TensorIndex == map[name]);
                    Assert.That(snapshot.Backend, Is.EqualTo(backend));
                    var metrics = Compare(snapshot.Values, native[name]);
                    metrics["stage"] = name; metrics["layer_index"] = snapshot.LayerIndex; metrics["tensor_index"] = snapshot.TensorIndex;
                    metrics["actual_backend"] = snapshot.Backend.ToString(); metrics["coefficient_sha256"] = anchor["coefficient_sha256"];
                    stages.Add(metrics); Save();
                }
                report["first_failed_observed_stage"] = stages.FirstOrDefault(value => !value.Value<bool>("elementwise_passed"))?["stage"];
                report["observation_completed"] = true; Save();
                Assert.That(stages.All(value => value.Value<bool>("elementwise_passed")), Is.True,
                    "Fixed native layer-zero numeric gate failed; see saved stages.");
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
        static string Hash(string path)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(assembly => assembly.GetType("EmbeddingGemma.Validation.PlayerFileHash")).FirstOrDefault(value => value != null);
            Assert.That(type, Is.Not.Null); using var stream = File.OpenRead(path);
            return (string)type.GetMethod("Compute").Invoke(null, new object[] { stream, null, null });
        }
        static void HashCheck(string path, string expected) => Assert.That(Hash(path), Is.EqualTo(expected));
    }
}
