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
    public sealed class CompensatedVisionReferenceTests
    {
        JObject reference, audit, report;
        JArray conditions;
        string root, output;
        static readonly Dictionary<string, string> ModelHashes = new Dictionary<string, string> {
            ["square"] = "4ecd0106463e833588c7584a58ea4973fdbab1eb1a56389c2f1597d3e7df34e7",
            ["landscape"] = "b9bc2336a34167eb0fdfdd99e49c6486d66b20afc23d72da8df1e10da4a1e632",
            ["portrait"] = "41c0ade742b09cf291c0b14343b1cf8bffa6f44c9bb2e8d782bc7ba508371e6b" };
        [OneTimeSetUp] public void AuditImmutableRealVisionArtifacts()
        {
            root = Path.GetFullPath(Path.Combine(Application.dataPath, "../../k/artifacts/vision-encoder"));
            output = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/compensated-vision-results/results.json"));
            conditions = new JArray();
            report = new JObject { ["scope"] = "fixed_real_vision_fp32_compensated_projection_encoder_api_only", ["success"] = false,
                ["runtime_source"] = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE"),
                ["reference_source"] = "e2a6dc01fbca8c9be1f3bbe875325ad338de0ca0", ["runtime_overlay"] = false,
                ["unity"] = Application.unityVersion, ["sentis"] = "2.6.1", ["gpu"] = SystemInfo.graphicsDeviceName,
                ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(), ["conditions"] = conditions,
                ["expected_conditions"] = 6, ["float16_validated"] = false, ["preprocessing_validated"] = false,
                ["end_to_end_validated"] = false, ["started_utc"] = DateTime.UtcNow.ToString("O") }; Save();
            var locked = JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath, "../Packages/packages-lock.json")));
            Assert.That(locked["dependencies"]["com.ayutaz.embeddinggemma"].Value<string>("hash"), Is.EqualTo(report.Value<string>("runtime_source")));
            HashCheck(Path.Combine(root, "vision-validation.json"), "ab1aa775b553f33a5e2892abed2cdcef7768e35fbad6fa5af9ec3286a7ba90ba");
            HashCheck(Path.Combine(root, "vision-reference.json"), "3dfccd1a487ce04df3f8d640483ca045f7a7a56231bc1096ad0f845c923a29c6");
            audit = JObject.Parse(File.ReadAllText(Path.Combine(root, "vision-validation.json")));
            reference = JObject.Parse(File.ReadAllText(Path.Combine(root, "vision-reference.json")));
            Assert.That(audit.Value<bool>("success"), Is.True); Assert.That(audit.Value<int>("profile_count"), Is.EqualTo(3));
            Assert.That(audit.Value<int>("case_count"), Is.EqualTo(6));
            Assert.That(JToken.DeepEquals(audit["metadata"], reference["metadata"]), Is.True);
            Assert.That(reference["metadata"].Value<string>("source_commit"), Is.EqualTo(report.Value<string>("reference_source")));
            Assert.That(reference["metadata"].Value<string>("model_revision"), Is.EqualTo("914f7f89142e33e77833254d9c9b90c3cef7303b"));
            foreach (var file in (JObject)audit["files"])
            {
                var path = Path.GetFullPath(Path.Combine(root, file.Key));
                Assert.That(path.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.Ordinal), Is.True);
                Assert.That(new FileInfo(path).Length, Is.EqualTo(file.Value.Value<long>("bytes")));
                HashCheck(path, file.Value.Value<string>("sha256"));
            }
            foreach (var pair in ModelHashes) HashCheck(Path.Combine(root, pair.Key + "-fp32.sentis"), pair.Value);
            report["reference_audited"] = true; Save();
        }
        [TestCase("square", BackendType.CPU)] [TestCase("square", BackendType.GPUCompute)]
        [TestCase("landscape", BackendType.CPU)] [TestCase("landscape", BackendType.GPUCompute)]
        [TestCase("portrait", BackendType.CPU)] [TestCase("portrait", BackendType.GPUCompute)]
        [Timeout(1200000)]
        public void CompensatedSavedModelMatchesBothFixedImages(string name, BackendType backend)
        {
            var cases = new JArray();
            var condition = new JObject { ["profile"] = name, ["requested_backend"] = backend.ToString(),
                ["success"] = false, ["phase"] = "load", ["cases"] = cases, ["saved_sha256"] = ModelHashes[name] };
            conditions.Add(condition); Save();
            try
            {
                var profile = reference["profiles"][name];
                Assert.That(profile["pixel_shape"].ToObject<int[]>(), Is.EqualTo(new[] { 1, 2520, 768 }));
                Assert.That(profile.Value<int>("feature_width"), Is.EqualTo(512));
                var positions = Ints(Path.Combine(root, profile.Value<string>("positions")));
                var width = (positions.Where((value, index) => index % 2 == 0).Max() + 1) * 16;
                var height = (positions.Where((value, index) => index % 2 == 1).Max() + 1) * 16;
                var geometry = new ImageEncoderProfile(width, height);
                Assert.That(geometry.FeatureCount, Is.EqualTo(profile.Value<int>("feature_count")));
                var original = TextModelFile.Load(Path.Combine(root, name + "-fp32.sentis"));
                var candidate = CompensatedModel.Create(original, 64);
                condition["original_layer_count"] = original.layers.Count; condition["candidate_layer_count"] = candidate.layers.Count;
                condition["largest_constant_bytes"] = candidate.constants.Max(value => value.lengthBytes); Save();
                using var api = new ImageFeatureEncoder(candidate, backend, geometry);
                var worker = (Worker)typeof(ImageFeatureEncoder).GetField("worker", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(api);
                var resident = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(worker);
                var fallback = resident.Intersect(candidate.layers.SelectMany(value => value.outputs).Concat(candidate.inputs.Select(value => value.index))).Count();
                var flush = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(worker);
                condition["cpu_computation_or_input_count"] = fallback; condition["readback_flush_count"] = flush.Count;
                if (backend == BackendType.GPUCompute) { Assert.That(SystemInfo.supportsComputeShaders, Is.True); Assert.That(fallback, Is.Zero); Assert.That(flush, Is.Empty); }
                foreach (var row in ((JArray)reference["cases"]).Where(value => value.Value<string>("profile") == name))
                {
                    condition["phase"] = "evaluate_" + row.Value<string>("id"); Save();
                    using var pixels = new Tensor<float>(new TensorShape(1, 2520, 768), Floats(Path.Combine(root, row.Value<string>("pixels"))));
                    if (backend == BackendType.GPUCompute) ComputeTensorData.Pin(pixels);
                    using var input = (ImagePreprocessingResult)Activator.CreateInstance(typeof(ImagePreprocessingResult),
                        BindingFlags.Instance | BindingFlags.NonPublic, null,
                        new object[] { pixels, width, height, geometry.FeatureCount, positions, new List<Tensor>(), new List<ComputeBuffer>() }, null);
                    using var encoded = api.Encode(input); Assert.That(encoded.SoftTokens.backendType, Is.EqualTo(backend));
                    var actual = encoded.SoftTokens.DownloadToArray();
                    var expected = row["features"].ToObject<float[][]>().SelectMany(value => value).ToArray();
                    Assert.That(actual.Length, Is.EqualTo(expected.Length));
                    var mismatches = 0; double maximum = 0, ratio = 0;
                    for (var index = 0; index < actual.Length; index++)
                    {
                        if (float.IsNaN(actual[index]) || float.IsInfinity(actual[index])) throw new InvalidOperationException("Nonfinite candidate feature.");
                        var difference = Math.Abs((double)actual[index] - expected[index]); var allowed = 1e-5 + 1e-4 * Math.Abs((double)expected[index]);
                        maximum = Math.Max(maximum, difference); ratio = Math.Max(ratio, difference / allowed); if (difference > allowed) mismatches++;
                    }
                    cases.Add(new JObject { ["id"] = row.Value<string>("id"), ["success"] = mismatches == 0,
                        ["mismatched_elements"] = mismatches, ["maximum_absolute_error"] = maximum,
                        ["maximum_tolerance_ratio"] = ratio, ["actual_backend"] = encoded.SoftTokens.backendType.ToString() }); Save();
                }
                condition["phase"] = "completed"; Assert.That(cases.Count, Is.EqualTo(2)); Save();
                Assert.That(cases.All(value => value.Value<bool>("success")), Is.True, "Keep the original FP32 elementwise gate.");
                condition["success"] = true;
            }
            catch (Exception error) { condition["error"] = error.ToString(); throw; }
            finally { Save(); }
        }
        [OneTimeTearDown] public void CompleteReport()
        {
            if (report == null) return;
            report["success"] = conditions.Count == 6 && conditions.All(value => value.Value<bool>("success"));
            report["completed_utc"] = DateTime.UtcNow.ToString("O"); Save();
        }
        static float[] Floats(string path) { var bytes = File.ReadAllBytes(path); var result = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, result, 0, bytes.Length); return result; }
        static int[] Ints(string path) { var bytes = File.ReadAllBytes(path); var result = new int[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, result, 0, bytes.Length); return result; }
        static string Hash(string path) { using var stream = File.OpenRead(path); return PlayerFileHash.Compute(stream, out _); }
        static void HashCheck(string path, string expected) => Assert.That(Hash(path), Is.EqualTo(expected));
        void Save() { Directory.CreateDirectory(Path.GetDirectoryName(output)); File.WriteAllText(output, report.ToString()); }
    }
}
