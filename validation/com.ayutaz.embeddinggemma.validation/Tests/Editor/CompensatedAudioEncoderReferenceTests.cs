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
    // Encoder-only acceptance against exactly the producer used by the latest full PCM chain.
    public sealed class CompensatedAudioEncoderReferenceTests
    {
        const string Producer = "4edbe17a8105233c5aaa1514faa2862b08a4f7e0";
        static string RuntimeSource => Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
        string root, resultPath;
        JObject reference, audit, report;
        JArray conditions;
        readonly Dictionary<string, string> models = new Dictionary<string, string>();

        [OneTimeSetUp]
        public void AuditCurrentNativeReferenceAndSavedModels()
        {
            var locked = JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath, "../Packages/packages-lock.json")));
            Assert.That(locked["dependencies"]["com.ayutaz.embeddinggemma"].Value<string>("hash"), Is.EqualTo(RuntimeSource));
            var requested = JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath, "../Packages/manifest.json")));
            Assert.That(requested["dependencies"].Value<string>("com.ayutaz.embeddinggemma").EndsWith("#" + RuntimeSource), Is.True);
            root = Path.GetFullPath(Path.Combine(Application.dataPath, "../../k/artifacts/audio-real-chain"));
            resultPath = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/compensated-full-encoder/results.json"));
            conditions = new JArray();
            report = new JObject
            {
                ["success"] = false, ["scope"] = "current_native_features_to_compensated_fp32_encoder_public_api_only",
                ["runtime_source"] = RuntimeSource, ["runtime_overlay"] = false, ["reference_source"] = Producer,
                ["unity"] = Application.unityVersion, ["sentis"] = "2.6.1", ["gpu"] = SystemInfo.graphicsDeviceName,
                ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(), ["conditions"] = conditions,
                ["hash_backends"] = new JObject(), ["feature_rtol"] = 1e-4, ["feature_atol"] = 1e-5,
                ["minimum_cosine"] = 0.999999, ["end_to_end_validated"] = false,
                ["float16_validated"] = false, ["pcm_preprocessing_validated"] = false,
                ["started_utc"] = DateTime.UtcNow.ToString("O")
            };
            Save();
            report["saved_model_unity"] = "6000.3.21f1";
            AuditHash(Resolve("audio-runtime-reference/audio-encoder-validation.json"), "68d29c96012ef0a44114f2d3edbbff8d14e116ac697abee1a36d0b212155cfa1");
            AuditHash(Resolve("audio-runtime-reference/audio-encoder-reference.json"), "ac23f39904b8149dd5ccb6485afcc93d96047f0977b5faf1d640c8bd018c13c9");
            audit = Read("audio-runtime-reference/audio-encoder-validation.json");
            reference = Read("audio-runtime-reference/audio-encoder-reference.json");
            Assert.That(reference["metadata"].Value<string>("source_commit"), Is.EqualTo(Producer));
            Assert.That(JToken.DeepEquals(reference["metadata"], audit["metadata"]), Is.True);
            Assert.That(reference["metadata"].Value<string>("model_revision"), Is.EqualTo("914f7f89142e33e77833254d9c9b90c3cef7303b"));
            Assert.That(reference["metadata"].Value<double>("feature_rtol"), Is.EqualTo(1e-4));
            Assert.That(reference["metadata"].Value<double>("feature_atol"), Is.EqualTo(1e-5));
            var cases = (JArray)reference["cases"];
            Assert.That(cases.Select(row => row.Value<string>("id")), Is.EquivalentTo(
                new[] { "silence", "impulse", "sine", "nyquist", "noise", "padded_noise" }));
            Assert.That(cases.Count, Is.EqualTo(6));
            Assert.That(((JObject)reference["profiles"]).Count, Is.EqualTo(2));
            foreach (var entry in (JObject)reference["profiles"])
            {
                var profile = entry.Value;
                Assert.That(profile.Value<string>("import_format"), Is.EqualTo("onnx"));
                Assert.That(profile.Value<int>("feature_width"), Is.EqualTo(512));
                var graph = profile.Value<string>("graph");
                AuditFile("audio-runtime-reference/" + graph, audit["files"][graph]);
                var mask = profile.Value<string>("frame_mask");
                AuditFile("audio-runtime-reference/" + mask, audit["files"][mask]);
                var cache = Read(entry.Key + "-model-cache.json");
                var context = new JObject { ["producer"] = Producer, ["graph_sha256"] = audit["files"][graph]["sha256"],
                    ["unity"] = "6000.3.21f1", ["sentis"] = "2.6.1", ["quantized"] = true };
                Assert.That(JToken.DeepEquals(cache["context"], context), Is.True);
                var pinnedModelHash = entry.Key == "frames-1119-valid-1119"
                    ? "3214cc2b179a3abf3f968816c58b83b8ee762661bfab7b58c6f142d48ed3ed3e"
                    : "954c7808fb2d47aa9b87404eac9ef0f2115c658e1b60cc494f1b4bc18c9025ed";
                Assert.That(cache["files"]["fp32"].Value<string>("sha256"), Is.EqualTo(pinnedModelHash));
                var name = cache["files"]["fp32"].Value<string>("name");
                AuditFile(name, cache["files"]["fp32"]);
                models.Add(entry.Key, Resolve(name));
            }
            foreach (var row in cases)
            {
                var name = row.Value<string>("input_features");
                AuditFile("audio-runtime-reference/" + name, audit["files"][name]);
            }
            report["metadata_and_files_audited"] = true;
            report["encoder_python_gate_passed"] = audit.Value<bool>("success");
            report["saved_models_reused"] = true;
            Save();
        }

        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        [Timeout(1200000)]
        public void SavedFp32EncoderApiMatchesCurrentSixNativeSignals(BackendType backend)
        {
            var condition = new JObject { ["requested_backend"] = backend.ToString(), ["success"] = false,
                ["cases"] = new JArray(), ["cpu_fallback_count"] = 0, ["readback_flush_count"] = 0 };
            conditions.Add(condition); Save();
            try
            {
                if (backend == BackendType.GPUCompute) Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Actual GPU required; never skip.");
                foreach (var entry in (JObject)reference["profiles"])
                {
                    condition["phase"] = "load_" + entry.Key; Save();
                    var frameMask = Ints(Resolve("audio-runtime-reference/" + entry.Value.Value<string>("frame_mask")));
                    var validFrames = frameMask.Count(value => value == 1);
                    Assert.That(frameMask, Is.EqualTo(Enumerable.Repeat(1, validFrames).Concat(Enumerable.Repeat(0, 1119 - validFrames)).ToArray()));
                    var originalModel = TextModelFile.Load(models[entry.Key]);
                    var model = CompensatedModel.Create(originalModel, 64);
                    condition["original_layer_count"] = originalModel.layers.Count;
                    condition["candidate_layer_count"] = model.layers.Count;
                    using (var api = new AudioFeatureEncoder(model, backend, new AudioEncoderProfile(validFrames)))
                    {
                        var worker = (Worker)typeof(AudioFeatureEncoder).GetField("worker", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(api);
                        var resident = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(worker);
                        var fallback = resident.Intersect(model.layers.SelectMany(layer => layer.outputs).Concat(model.inputs.Select(input => input.index))).Count();
                        var flush = ((HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(worker)).Count;
                        if (backend == BackendType.GPUCompute)
                        {
                            condition["cpu_fallback_count"] = (int)condition["cpu_fallback_count"] + fallback;
                            condition["readback_flush_count"] = (int)condition["readback_flush_count"] + flush;
                            Assert.That(fallback, Is.Zero); Assert.That(flush, Is.Zero);
                        }
                        foreach (var row in ((JArray)reference["cases"]).Where(row => row.Value<string>("profile") == entry.Key))
                        {
                            var id = row.Value<string>("id"); condition["phase"] = "evaluate_" + id; Save();
                            var values = Floats(Resolve("audio-runtime-reference/" + row.Value<string>("input_features")));
                            var features = new Tensor<float>(new TensorShape(1, 1119, 128), values);
                            var mask = new Tensor<int>(new TensorShape(1, 1119), frameMask);
                            if (backend == BackendType.GPUCompute) { ComputeTensorData.Pin(features); ComputeTensorData.Pin(mask); }
                            // Test-only construction from independently audited native features; this is not PCM preprocessing proof.
                            using var input = (AudioPreprocessingResult)Activator.CreateInstance(typeof(AudioPreprocessingResult),
                                BindingFlags.Instance | BindingFlags.NonPublic, null, new object[] { features, mask, validFrames }, null);
                            using var encoded = api.Encode(input);
                            Assert.That(encoded.SoftTokens.backendType, Is.EqualTo(backend));
                            var actual = encoded.SoftTokens.DownloadToArray();
                            var expected = row["features"].ToObject<float[][]>().SelectMany(vector => vector).ToArray();
                            var comparison = Compare(actual, expected);
                            comparison["id"] = id; comparison["profile"] = entry.Key; comparison["actual_backend"] = encoded.SoftTokens.backendType.ToString();
                            ((JArray)condition["cases"]).Add(comparison); Save();
                        }
                    }
                    model = null; GC.Collect(); GC.WaitForPendingFinalizers();
                }
                Assert.That(((JArray)condition["cases"]).Count, Is.EqualTo(6));
                condition["phase"] = "completed"; Save();
                Assert.That(((JArray)condition["cases"]).All(row => row.Value<bool>("success")), Is.True,
                    "Every current native feature must satisfy the unchanged encoder gate.");
                condition["success"] = true;
            }
            catch (Exception error) { condition["error"] = error.ToString(); throw; }
            finally { Save(); }
        }

        static JObject Compare(float[] actual, float[] expected)
        {
            Assert.That(actual.Length, Is.EqualTo(expected.Length));
            int mismatches = 0; double maximum = 0, ratio = 0, cosine = 1;
            for (var token = 0; token < actual.Length / 512; token++)
            {
                double dot = 0, left = 0, right = 0;
                for (var channel = 0; channel < 512; channel++)
                {
                    var index = token * 512 + channel;
                    Assert.That(float.IsNaN(actual[index]) || float.IsInfinity(actual[index]), Is.False);
                    var difference = Math.Abs((double)actual[index] - expected[index]);
                    var allowed = 1e-5 + 1e-4 * Math.Abs((double)expected[index]);
                    maximum = Math.Max(maximum, difference); ratio = Math.Max(ratio, difference / allowed);
                    if (difference > allowed) mismatches++;
                    dot += (double)actual[index] * expected[index]; left += (double)actual[index] * actual[index]; right += (double)expected[index] * expected[index];
                }
                Assert.That(left * right, Is.GreaterThan(0));
                cosine = Math.Min(cosine, dot / Math.Sqrt(left * right));
            }
            return new JObject { ["success"] = mismatches == 0 && cosine >= 0.999999, ["mismatched_elements"] = mismatches,
                ["maximum_absolute_error"] = maximum, ["maximum_tolerance_ratio"] = ratio, ["minimum_cosine"] = cosine };
        }

        [OneTimeTearDown] public void Complete()
        {
            if (report == null) return;
            report["success"] = conditions.Count == 2 && conditions.All(value => value.Value<bool>("success"));
            report["sentis_encoder_numeric_validated"] = report["success"];
            report["completed_utc"] = DateTime.UtcNow.ToString("O"); Save();
        }
        void AuditFile(string name, JToken spec)
        { Assert.That(spec, Is.Not.Null); var path = Resolve(name); Assert.That(new FileInfo(path).Length, Is.EqualTo(spec.Value<long>("bytes"))); AuditHash(path, spec.Value<string>("sha256")); }
        void AuditHash(string path, string expected)
        {
            using var stream = File.OpenRead(path); using var hash = SHA256.Create();
            Assert.That(BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant(), Is.EqualTo(expected));
            ((JObject)report["hash_backends"])[Path.GetRelativePath(root, path)] = "System.Security.Cryptography.SHA256";
        }
        string Resolve(string name)
        { var path = Path.GetFullPath(Path.Combine(root, name)); Assert.That(path.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.Ordinal), Is.True); return path; }
        JObject Read(string name) => JObject.Parse(File.ReadAllText(Resolve(name)));
        static float[] Floats(string path) { var bytes = File.ReadAllBytes(path); var values = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length); return values; }
        static int[] Ints(string path) { var bytes = File.ReadAllBytes(path); var values = new int[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length); return values; }
        void Save() { Directory.CreateDirectory(Path.GetDirectoryName(resultPath)); File.WriteAllText(resultPath, report.ToString()); }
    }
}

