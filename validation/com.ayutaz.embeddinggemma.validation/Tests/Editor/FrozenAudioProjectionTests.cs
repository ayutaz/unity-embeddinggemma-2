using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using EmbeddingGemma.Tests;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class FrozenAudioProjectionTests
    {
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)] [Timeout(1200000)]
        public void CompareCompensatedSavedWeightsAgainstFrozenNativeInputs(BackendType backend)
        {
            var root = Path.GetFullPath(Path.Combine(Application.dataPath, "../../k/artifacts/audio-real-chain"));
            var context = Path.GetFullPath(Path.Combine(Application.dataPath, "../../k/artifacts/audio-native-feed-forward-replay"));
            var output = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/frozen-projection-results"));
            Directory.CreateDirectory(output);
            var stages = new JArray();
            var report = new JObject { ["success"] = false, ["scope"] = "fixed_native_inputs_saved_fp32_compensated_projection_only",
                ["source"] = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE"), ["requested_backend"] = backend.ToString(),
                ["unity"] = Application.unityVersion, ["sentis"] = "2.6.1", ["gpu"] = SystemInfo.graphicsDeviceName,
                ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(), ["stages"] = stages,
                ["full_encoder_validated"] = false, ["end_to_end_validated"] = false,
                ["started_utc"] = DateTime.UtcNow.ToString("O") };
            void Save() => File.WriteAllText(Path.Combine(output, backend + ".json"), report.ToString()); Save();
            try
            {
                var replayPath = Path.Combine(context, "audio-layer-replay-report.json");
                RequireHash(replayPath, "99529dc0fa6c74825b23fca653d82091797ff1e5267d26c5cdae723fcf86f501");
                var replay = JObject.Parse(File.ReadAllText(replayPath));
                Assert.That(replay.Value<bool>("native_final_exact_match") && replay.Value<bool>("success"), Is.True);
                var anchorsPath = Path.Combine(context, "feed-forward-norm-anchors.json");
                RequireHash(anchorsPath, "95f3e32536dce9b3b3820050dac4bc837c0f460c75e4e56735381bf924007c33");
                var anchors = JObject.Parse(File.ReadAllText(anchorsPath));
                var cache = JObject.Parse(File.ReadAllText(Path.Combine(root, "frames-1119-valid-1119-model-cache.json")));
                Assert.That(cache["context"].Value<string>("producer"), Is.EqualTo("4edbe17a8105233c5aaa1514faa2862b08a4f7e0"));
                Assert.That(cache["context"].Value<string>("graph_sha256"), Is.EqualTo(anchors.Value<string>("original_graph_sha256")));
                var saved = Path.GetFullPath(Path.Combine(root, cache["files"]["fp32"].Value<string>("name")));
                Assert.That(saved.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.Ordinal), Is.True);
                RequireHash(saved, "3214cc2b179a3abf3f968816c58b83b8ee762661bfab7b58c6f142d48ed3ed3e");
                var source = TextModelFile.Load(saved);
                var map = AudioCoefficientLayerMap.ResolveFeedForward(source, (JArray)anchors["anchors"]);
                var sourceCount = source.layers.Count;
                foreach (var name in new[] { "ffw_layer_1", "ffw_layer_2" })
                {
                    var inputName = name == "ffw_layer_1" ? "pre_layer_norm" : "act_fn";
                    var nativeInput = ReadPart(context, replay, inputName, out var shape);
                    var expected = ReadPart(context, replay, name, out var targetShape);
                    var matrixIndex = map[name];
                    var producer = source.layers.Single(layer => layer.outputs.Contains(matrixIndex));
                    for (var count = 0; producer.GetType().Name == "Clip" && count < 5; count++)
                    {
                        var minimum = Scalar(source, producer.inputs[1]); var maximum = Scalar(source, producer.inputs[2]);
                        Assert.That(expected.All(value => value >= minimum && value <= maximum), Is.True);
                        matrixIndex = producer.inputs[0]; producer = source.layers.Single(layer => layer.outputs.Contains(matrixIndex));
                    }
                    Assert.That(producer.GetType().Name, Is.AnyOf("MatMul", "MatMul2D"));
                    foreach (var flag in new[] { "transposeA", "transposeB" })
                    {
                        var field = producer.GetType().GetField(flag); if (field != null) Assert.That((bool)field.GetValue(producer), Is.False);
                    }
                    var weight = source.constants.Single(value => value.index == producer.inputs[1]);
                    Assert.That(weight.shape, Is.EqualTo(new TensorShape(shape[2], targetShape[2])));
                    var values = ConstantFloats(weight); var candidate = CompensatedProjection.BuildForShape(values, targetShape[2], 64, shape);
                    using var input = new Tensor<float>(new TensorShape(shape), nativeInput);
                    if (backend == BackendType.GPUCompute) { Assert.That(SystemInfo.supportsComputeShaders, Is.True); ComputeTensorData.Pin(input); }
                    using var worker = new Worker(candidate, backend); worker.Schedule(input);
                    var tensor = (Tensor<float>)worker.PeekOutput(); Assert.That(tensor.backendType, Is.EqualTo(backend));
                    var fallback = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(worker);
                    var fallbackCount = candidate.layers.Count(layer => fallback.Contains(layer.outputs[0]));
                    if (backend == BackendType.GPUCompute) Assert.That(fallbackCount, Is.EqualTo(0));
                    JObject normMetrics = null;
                    if (name == "ffw_layer_2")
                    {
                        var post = new JArray(((JArray)anchors["anchors"]).Single(row => row.Value<string>("stage") == "post_layer_norm").DeepClone());
                        var norm = AudioNormIsolation.Create(source, post, new TensorShape(targetShape));
                        using var normWorker = new Worker(norm, backend); normWorker.Schedule(tensor);
                        var normalized = (Tensor<float>)normWorker.PeekOutput(); Assert.That(normalized.backendType, Is.EqualTo(backend));
                        normMetrics = Metrics(AudioCoefficientLayerMap.RestoreHalfScale(normalized.DownloadToArray()), ReadPart(context, replay, "post_layer_norm", out _));
                        normMetrics["stage"] = "linear2_post_norm";
                    }
                    var metrics = Metrics(tensor.DownloadToArray(), expected);
                    metrics["stage"] = name; metrics["weight_index"] = weight.index; metrics["weight_sha256"] = ArrayHash(weight);
                    metrics["actual_backend"] = tensor.backendType.ToString(); metrics["calculation_fallback_count"] = fallbackCount;
                    stages.Add(metrics); if (normMetrics != null) stages.Add(normMetrics); Save();
                }
                Assert.That(source.layers.Count, Is.EqualTo(sourceCount));
                report["comparison_completed"] = true; Save();
                Assert.That(stages.All(value => value.Value<bool>("elementwise_passed")), Is.True, "Saved-weight candidate numeric gate failed.");
                report["success"] = true;
            }
            catch (Exception error) { report["error"] = error.ToString(); throw; }
            finally { report["completed_utc"] = DateTime.UtcNow.ToString("O"); Save(); }
        }

        static float[] ReadPart(string root, JObject replay, string name, out int[] shape)
        {
            var part = replay["parts"]["feed_forward1." + name]; shape = part["shape"].ToObject<int[]>();
            Assert.That(shape.Length, Is.EqualTo(3)); Assert.That(shape[0], Is.EqualTo(1)); Assert.That(shape[1], Is.EqualTo(280));
            var path = Path.Combine(root, "audio-layer-replay", part.Value<string>("file")); RequireHash(path, part.Value<string>("sha256"));
            var bytes = File.ReadAllBytes(path); Assert.That(bytes.Length, Is.EqualTo(part.Value<int>("bytes")));
            var values = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length); return values;
        }
        static ArraySegment<byte> Bytes(Constant value) => (ArraySegment<byte>)typeof(Constant).GetField("array", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(value);
        static float[] ConstantFloats(Constant value)
        {
            Assert.That(value.dataType, Is.EqualTo(DataType.Float)); var bytes = Bytes(value);
            var result = new float[value.lengthBytes / 4]; Buffer.BlockCopy(bytes.Array, bytes.Offset, result, 0, value.lengthBytes); return result;
        }
        static float Scalar(Model source, int index) { var value = source.constants.Single(row => row.index == index); Assert.That(value.lengthBytes, Is.EqualTo(4)); return ConstantFloats(value)[0]; }
        static string ArrayHash(Constant value) { var bytes = Bytes(value); using var hash = SHA256.Create(); return BitConverter.ToString(hash.ComputeHash(bytes.Array, bytes.Offset, bytes.Count)).Replace("-", "").ToLowerInvariant(); }
        static void RequireHash(string path, string expected) { using var stream = File.OpenRead(path); using var hash = SHA256.Create(); Assert.That(BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant(), Is.EqualTo(expected)); }
        static JObject Metrics(float[] actual, float[] expected)
        {
            Assert.That(actual.Length, Is.EqualTo(expected.Length)); var mismatches = 0; double maximum = 0, ratio = 0;
            for (var index = 0; index < actual.Length; index++)
            {
                Assert.That(float.IsNaN(actual[index]) || float.IsInfinity(actual[index]), Is.False);
                var difference = Math.Abs((double)actual[index] - expected[index]); var allowed = 1e-5 + 1e-4 * Math.Abs((double)expected[index]);
                maximum = Math.Max(maximum, difference); ratio = Math.Max(ratio, difference / allowed); if (difference > allowed) mismatches++;
            }
            return new JObject { ["elementwise_passed"] = mismatches == 0, ["mismatches"] = mismatches, ["maximum_absolute_error"] = maximum, ["maximum_tolerance_ratio"] = ratio };
        }
    }
}
