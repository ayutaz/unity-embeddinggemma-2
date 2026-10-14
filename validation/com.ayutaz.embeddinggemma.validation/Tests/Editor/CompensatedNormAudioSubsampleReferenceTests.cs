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
    public sealed class CompensatedNormAudioSubsampleReferenceTests
    {
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)] [Timeout(1200000)]
        public void ComparePinnedSubsampleStagesWithoutChangingNormalFinal(BackendType backend)
        {
            var root = Path.GetFullPath(Path.Combine(Application.dataPath, "../../k/artifacts/audio-real-chain"));
            var nativeRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "../../numerical-acceptance-state/subsample-native-replay"));
            var output = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/compensated-norm1-subsample-results"));
            Directory.CreateDirectory(output);
            var stages = new JArray();
            var report = new JObject { ["scope"] = "silence_compensated_norm1_subsample_observation_only", ["success"] = false,
                ["observation_completed"] = false, ["runtime_source"] = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE"),
                ["native_replay_source"] = "e868446e44fb96494605b61b99f86cf3a5f5f973",
                ["requested_backend"] = backend.ToString(), ["stages"] = stages, ["unity"] = Application.unityVersion,
                ["gpu"] = SystemInfo.graphicsDeviceName, ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(),
                ["full_encoder_validated"] = false, ["end_to_end_validated"] = false };
            void Save() => File.WriteAllText(Path.Combine(output, backend + ".json"), report.ToString()); Save();
            try
            {
                var locked = JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath, "../Packages/packages-lock.json")));
                Assert.That(locked["dependencies"]["com.ayutaz.embeddinggemma"].Value<string>("hash"), Is.EqualTo(report.Value<string>("runtime_source")));
                var replayPath = Path.Combine(nativeRoot, "audio-layer-replay-report.json");
                HashCheck(replayPath, "664a8242d079f53bf98d2032f9398de6d2c8d6bcd2312d1baaa35bb498ce5fb8");
                var replay = JObject.Parse(File.ReadAllText(replayPath));
                Assert.That(replay.Value<bool>("success") && replay.Value<bool>("subsample_native_final_exact_match"), Is.True);
                Assert.That(replay.Value<string>("diagnostic_source"), Is.EqualTo(report.Value<string>("native_replay_source")));
                Assert.That(replay.Value<string>("reference_source"), Is.EqualTo("4edbe17a8105233c5aaa1514faa2862b08a4f7e0"));
                Assert.That(replay.Value<string>("model_revision"), Is.EqualTo("914f7f89142e33e77833254d9c9b90c3cef7303b"));
                Assert.That(replay.Value<string>("case"), Is.EqualTo("silence"));
                Assert.That(replay.Value<string>("profile"), Is.EqualTo("frames-1119-valid-1119"));
                var cache = JObject.Parse(File.ReadAllText(Path.Combine(root, "frames-1119-valid-1119-model-cache.json")));
                var saved = Path.GetFullPath(Path.Combine(root, cache["files"]["fp32"].Value<string>("name")));
                Assert.That(saved.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.Ordinal), Is.True);
                HashCheck(saved, "3214cc2b179a3abf3f968816c58b83b8ee762661bfab7b58c6f142d48ed3ed3e");
                var model = TextModelFile.Load(saved);
                var prefix = new List<Layer>();
                foreach (var layer in model.layers) { prefix.Add(layer); if (layer.outputs.Contains(49)) break; }
                Assert.That(prefix.Last().outputs, Does.Contain(49));
                var constants = model.constants.ToDictionary(value => value.index);
                report["original_prefix"] = new JArray(prefix.Select(layer => new JObject {
                    ["kind"] = layer.GetType().Name, ["inputs"] = new JArray(layer.inputs), ["outputs"] = new JArray(layer.outputs),
                    ["constant_inputs"] = new JArray(layer.inputs.Where(constants.ContainsKey).Select(index => new JObject {
                        ["index"] = index, ["bytes"] = constants[index].lengthBytes, ["sha256"] = ConstantHash(constants[index]) })) })); Save();
                int Weighted(string kind, string hash)
                {
                    var matches = prefix.Where(layer => layer.GetType().Name == kind && layer.inputs.Length >= 2 &&
                        constants.TryGetValue(layer.inputs[1], out var weight) && ConstantHash(weight) == hash).ToArray();
                    Assert.That(matches.Length, Is.EqualTo(1), "Pinned subsample coefficient/path is missing or ambiguous: " + kind);
                    return matches[0].outputs[0];
                }
                int Activation(int norm)
                {
                    var index = norm;
                    for (var count = 0; count < 4; count++)
                    {
                        var matches = prefix.Where(layer => layer.inputs.Length > 0 && layer.inputs[0] == index).ToArray();
                        Assert.That(matches.Length, Is.EqualTo(1), "Pinned norm-to-activation edge is ambiguous.");
                        var next = matches[0]; var kind = next.GetType().Name;
                        if (kind == "Relu") return next.outputs[0];
                        Assert.That(kind == "Transpose" || kind == "Identity", Is.True, "Unexpected norm-to-activation operation: " + kind);
                        index = next.outputs[0];
                    }
                    throw new InvalidOperationException("Pinned activation edge is missing.");
                }
                var map = new Dictionary<string, int> {
                    ["layer0.conv"] = Weighted("Conv", "e4d4ee18dc86a9945d38c159a861784a2c3d9467d6e0be9c48a414e60e2b92fd"),
                    ["layer0.norm"] = Weighted("LayerNormalization", "dbb0fc3431144ac4a6265e965a3bac5bb1c0ee9e87b266bc4cb13d2b7d4d8c24"),
                    ["layer1.conv"] = Weighted("Conv", "68309f01ea3e738fccb24a0db5db608841a3a8e59b0a4fa5d0fb521ed60bf43c"),
                    ["layer1.norm"] = Weighted("LayerNormalization", "7b744a9e296e2331f5553570fdb993412b8f760f1a000179f366554cc84b1434"),
                    ["input_proj_linear"] = 49 };
                map["layer0.act"] = Activation(map["layer0.norm"]); map["layer1.act"] = Activation(map["layer1.norm"]);
                report["coefficient_bound_outputs"] = JObject.FromObject(map); Save();
                model = CompensatedModel.CreateWithCompensatedNorms(model, 64, 1);
                var inputPath = Path.Combine(root, "audio-runtime-reference/frames-1119-valid-1119/silence-features-fp32.bin");
                HashCheck(inputPath, "366de71cb4e45e4feda58a605babe4ea9d8693a7e37e84eac0735a7c42d0ad24");
                using var input = new Tensor<float>(new TensorShape(1, 1119, 128), Floats(inputPath));
                if (backend == BackendType.GPUCompute) { Assert.That(SystemInfo.supportsComputeShaders, Is.True); ComputeTensorData.Pin(input); }
                var observed = WorkerObservation.ObserveIndexes(model, backend, input, new HashSet<int>(map.Values));
                Assert.That(observed.Snapshots.Count, Is.EqualTo(7)); Assert.That(observed.FinalMatchesBaselineBits, Is.True);
                report["normal_final_matches_observed_bits"] = observed.FinalMatchesBaselineBits;
                foreach (var name in new[] { "layer0.conv", "layer0.norm", "layer0.act", "layer1.conv", "layer1.norm", "layer1.act", "input_proj_linear" })
                {
                    var part = replay["parts"]["subsample." + name];
                    var path = Path.Combine(nativeRoot, "audio-layer-replay", part.Value<string>("file"));
                    HashCheck(path, part.Value<string>("sha256")); Assert.That(new FileInfo(path).Length, Is.EqualTo(part.Value<long>("bytes")));
                    var snapshot = observed.Snapshots.Single(value => value.TensorIndex == map[name]);
                    Assert.That(snapshot.Backend, Is.EqualTo(backend));
                    var actual = snapshot.Values; var expected = Floats(path); Assert.That(actual.Length, Is.EqualTo(expected.Length));
                    var mismatches = 0; double maximum = 0, ratio = 0;
                    for (var index = 0; index < actual.Length; index++)
                    {
                        if (float.IsNaN(actual[index]) || float.IsInfinity(actual[index])) throw new InvalidOperationException("Nonfinite observed value.");
                        var difference = Math.Abs((double)actual[index] - expected[index]); var allowed = 1e-5 + 1e-4 * Math.Abs((double)expected[index]);
                        maximum = Math.Max(maximum, difference); ratio = Math.Max(ratio, difference / allowed); if (difference > allowed) mismatches++;
                    }
                    var capturePath = Path.Combine(output, backend + "-" + name + "-fp32.bin");
                    var bytes = new byte[actual.Length * 4]; Buffer.BlockCopy(actual, 0, bytes, 0, bytes.Length); File.WriteAllBytes(capturePath, bytes);
                    stages.Add(new JObject { ["stage"] = name, ["tensor_index"] = snapshot.TensorIndex, ["layer_index"] = snapshot.LayerIndex,
                        ["actual_backend"] = snapshot.Backend.ToString(), ["elementwise_passed"] = mismatches == 0,
                        ["mismatches"] = mismatches, ["maximum_absolute_error"] = maximum, ["maximum_tolerance_ratio"] = ratio,
                        ["capture_file"] = Path.GetFileName(capturePath), ["capture_sha256"] = Hash(capturePath) }); Save();
                }
                report["observation_completed"] = true;
                report["first_failed_observed_stage"] = stages.FirstOrDefault(value => !value.Value<bool>("elementwise_passed"))?["stage"]; Save();
                Assert.That(stages.All(value => value.Value<bool>("elementwise_passed")), Is.True, "Pinned subsample numeric gate failed.");
                report["success"] = true;
            }
            catch (Exception error) { report["error"] = error.ToString(); throw; }
            finally { report["completed_utc"] = DateTime.UtcNow.ToString("O"); Save(); }
        }
        static float[] Floats(string path) { var bytes = File.ReadAllBytes(path); var values = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length); return values; }
        static string ConstantHash(Constant value)
        {
            var bytes = (ArraySegment<byte>)typeof(Constant).GetField("array", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(value);
            using var stream = new MemoryStream(bytes.Array, bytes.Offset, value.lengthBytes, false); return PlayerFileHash.Compute(stream, out _);
        }
        static string Hash(string path) { using var stream = File.OpenRead(path); return PlayerFileHash.Compute(stream, out _); }
        static void HashCheck(string path, string expected) => Assert.That(Hash(path), Is.EqualTo(expected));
    }
}


