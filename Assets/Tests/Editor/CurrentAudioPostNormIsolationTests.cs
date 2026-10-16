using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class CurrentAudioPostNormIsolationTests
    {
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)] [Timeout(1200000)]
        public void CompareSavedPostNormAgainstFrozenExactLinear2Input(BackendType backend)
        {
            var root = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/audio-real-chain"));
            var context = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/audio-native-feed-forward-replay"));
            var output = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/audio-post-norm-isolation-results"));
            Directory.CreateDirectory(output);
            var report = new JObject { ["success"] = false, ["isolated_norm_validated"] = false,
                ["scope"] = "saved_post_norm_with_frozen_exact_native_linear2_input",
                ["runtime_source"] = "adccab24dbcc3f5dd8c8ac79911a7604a6014021", ["runtime_overlay"] = false,
                ["reference_source"] = "4edbe17a8105233c5aaa1514faa2862b08a4f7e0",
                ["native_replay_source"] = "6ac2c92191a62ada4830adeee5235fb3c4b90f15",
                ["requested_backend"] = backend.ToString(), ["unity_version"] = Application.unityVersion,
                ["graphics_device"] = SystemInfo.graphicsDeviceName, ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(),
                ["full_encoder_validated"] = false, ["end_to_end_validated"] = false, ["started_utc"] = DateTime.UtcNow.ToString("O") };
            void Save() => File.WriteAllText(Path.Combine(output, backend + ".json"), report.ToString()); Save();
            try
            {
                var locked = JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath, "../Packages/packages-lock.json")));
                Assert.That(locked["dependencies"]["com.ayutaz.embeddinggemma"].Value<string>("hash"), Is.EqualTo(report.Value<string>("runtime_source")));
                var replayPath = Path.Combine(context, "audio-layer-replay-report.json");
                HashCheck(replayPath, "99529dc0fa6c74825b23fca653d82091797ff1e5267d26c5cdae723fcf86f501");
                var replay = JObject.Parse(File.ReadAllText(replayPath));
                Assert.That(replay.Value<bool>("success") && replay.Value<bool>("native_final_exact_match"), Is.True);
                Assert.That(replay.Value<string>("reference_source"), Is.EqualTo(report.Value<string>("reference_source")));
                Assert.That(replay.Value<string>("diagnostic_source"), Is.EqualTo(report.Value<string>("native_replay_source")));
                var anchorPath = Path.Combine(context, "feed-forward-norm-anchors.json");
                HashCheck(anchorPath, "95f3e32536dce9b3b3820050dac4bc837c0f460c75e4e56735381bf924007c33");
                var manifest = JObject.Parse(File.ReadAllText(anchorPath));
                var anchors = new JArray(((JArray)manifest["anchors"]).Single(row => row.Value<string>("stage") == "post_layer_norm").DeepClone());
                var cache = JObject.Parse(File.ReadAllText(Path.Combine(root, "frames-1119-valid-1119-model-cache.json")));
                Assert.That(cache["context"].Value<string>("graph_sha256"), Is.EqualTo(manifest.Value<string>("original_graph_sha256")));
                var modelPath = Path.GetFullPath(Path.Combine(root, cache["files"]["fp32"].Value<string>("name")));
                Assert.That(modelPath.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.Ordinal), Is.True);
                HashCheck(modelPath, "3214cc2b179a3abf3f968816c58b83b8ee762661bfab7b58c6f142d48ed3ed3e");
                var source = TextModelFile.Load(modelPath); var countBefore = source.layers.Count;
                var originalInput = AudioCoefficientLayerMap.ResolveNormInputs(source, anchors).Single().Value;
                var slice = AudioNormIsolation.Create(source, anchors, new TensorShape(1, 280, 1024));
                report["original_input_index"] = originalInput; report["isolated_input_index"] = slice.inputs[0].index;
                report["output_index"] = slice.outputs[0].index; report["layer_count"] = slice.layers.Count;
                report["constant_count"] = slice.constants.Count; report["constant_payload_bytes"] = slice.constants.Sum(value => value.lengthBytes);
                report["source_layer_count_unchanged"] = source.layers.Count == countBefore;
                Assert.That(slice.layers.Count, Is.EqualTo(6)); Assert.That(slice.constants.Count, Is.EqualTo(2));
                var nativeInput = ReadPart(context, replay, "feed_forward1.ffw_layer_2");
                var expected = ReadPart(context, replay, "feed_forward1.post_layer_norm");
                var clip = source.layers.Single(layer => layer.outputs.Contains(originalInput));
                Assert.That(clip.GetType().Name, Is.EqualTo("Clip"));
                var minimum = Scalar(source, clip.inputs[1]); var maximum = Scalar(source, clip.inputs[2]);
                Assert.That(nativeInput.All(value => value >= minimum && value <= maximum), Is.True);
                report["native_input_unchanged_by_original_clip"] = true;
                using var input = new Tensor<float>(new TensorShape(1, 280, 1024), nativeInput);
                if (backend == BackendType.GPUCompute) { Assert.That(SystemInfo.supportsComputeShaders, Is.True); ComputeTensorData.Pin(input); }
                using var worker = new Worker(slice, backend);
                var fallback = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(worker);
                report["calculation_fallback_count"] = slice.layers.Count(layer => fallback.Contains(layer.outputs[0]));
                if (backend == BackendType.GPUCompute) Assert.That(report.Value<int>("calculation_fallback_count"), Is.EqualTo(0));
                worker.Schedule(input); var tensor = (Tensor<float>)worker.PeekOutput();
                report["actual_backend"] = tensor.backendType.ToString(); Assert.That(tensor.backendType, Is.EqualTo(backend));
                var actual = AudioCoefficientLayerMap.RestoreHalfScale(tensor.DownloadToArray());
                var metrics = Compare(actual, expected); report["metrics"] = metrics; report["norm_execution_completed"] = true;
                report["diagnostic_restore_multiplier"] = 2; Save();
                Assert.That(metrics.Value<bool>("elementwise_passed"), Is.True, "Fixed-input post norm numeric mismatch.");
                report["isolated_norm_validated"] = true; report["success"] = true;
            }
            catch (Exception error) { report["error"] = error.ToString(); throw; }
            finally { report["completed_utc"] = DateTime.UtcNow.ToString("O"); Save(); }
        }
        static float[] ReadPart(string root, JObject reference, string name)
        {
            var part = reference["parts"][name]; Assert.That(part["shape"].ToObject<int[]>(), Is.EqualTo(new[] { 1, 280, 1024 }));
            var path = Path.Combine(root, "audio-layer-replay", part.Value<string>("file"));
            Assert.That(new FileInfo(path).Length, Is.EqualTo(part.Value<long>("bytes"))); HashCheck(path, part.Value<string>("sha256"));
            var bytes = File.ReadAllBytes(path); var values = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length); return values;
        }
        static JObject Compare(float[] actual, float[] expected)
        {
            Assert.That(actual.Length, Is.EqualTo(expected.Length)); int mismatches = 0; double maximum = 0, ratio = 0;
            for (var index = 0; index < actual.Length; index++)
            {
                Assert.That(float.IsNaN(actual[index]) || float.IsInfinity(actual[index]), Is.False);
                var error = Math.Abs((double)actual[index] - expected[index]); var allowed = 1e-5 + 1e-4 * Math.Abs((double)expected[index]);
                maximum = Math.Max(maximum, error); ratio = Math.Max(ratio, error / allowed); if (error > allowed) mismatches++;
            }
            return new JObject { ["elementwise_passed"] = mismatches == 0, ["mismatches"] = mismatches,
                ["maximum_absolute_error"] = maximum, ["maximum_tolerance_ratio"] = ratio };
        }
        static float Scalar(Model model, int index)
        {
            var value = model.constants.Single(constant => constant.index == index); Assert.That(value.lengthBytes, Is.EqualTo(4));
            var bytes = (ArraySegment<byte>)typeof(Constant).GetField("array", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(value);
            return BitConverter.ToSingle(bytes.Array, bytes.Offset);
        }
        static void HashCheck(string path, string expected)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(assembly => assembly.GetType("EmbeddingGemma.Validation.PlayerFileHash")).FirstOrDefault(value => value != null);
            Assert.That(type, Is.Not.Null); using var stream = File.OpenRead(path);
            Assert.That((string)type.GetMethod("Compute").Invoke(null, new object[] { stream, null, null }), Is.EqualTo(expected));
        }
    }
}
