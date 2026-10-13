using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    // Complete FP32 encoder/projector reference, including all six original signals.
    // NumPy failures, Float16, preprocessing/backbone and final embeddings are separate.
    public sealed class AudioEncoderReferenceTests
    {
        string root, resultPath;
        JObject reference, validation, report;
        JArray conditions;
        readonly Dictionary<string, string> saved = new Dictionary<string, string>();

        [OneTimeSetUp]
        public void AuditCompleteNativeReferenceAndSelectedOnnxFiles()
        {
            root = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/audio-encoder-runtime"));
            resultPath = Path.Combine(root, "../audio-encoder-runtime-results/results.json");
            conditions = new JArray();
            report = new JObject { ["success"] = false, ["scope"] = "real_fp32_audio_encoder_projector_only",
                ["float16_validated"] = false, ["end_to_end_validated"] = false,
                ["gpu_preprocessing_validated"] = false, ["unity"] = Application.unityVersion,
                ["gpu"] = SystemInfo.graphicsDeviceName, ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(),
                ["conditions"] = conditions, ["started_utc"] = DateTime.UtcNow.ToString("O") };
            Save();
            validation = JObject.Parse(File.ReadAllText(Resolve("audio-encoder-validation.json")));
            var source = File.ReadAllText(Resolve("expected-source.txt")).Trim();
            Assert.That(source, Does.Match("^[0-9a-f]{40}$"));
            Assert.That(validation["metadata"].Value<string>("source_commit"), Is.EqualTo(source));
            Assert.That(validation.Value<bool>("native_reference_complete"), Is.True);
            Assert.That(validation.Value<int>("case_count"), Is.EqualTo(6));
            Assert.That(validation.Value<int>("profile_count"), Is.EqualTo(2));
            Assert.That(validation.Value<double>("minimum_cosine"), Is.GreaterThanOrEqualTo(0.999999));
            report["numpy_validation_success"] = validation["success"];
            report["numpy_onnx_numerically_validated"] = validation["onnx_numerically_validated"];
            report["numpy_failed_case_count"] = validation["onnx_failed_case_count"];
            report["reference_source"] = source;
            Audit("audio-encoder-reference.json");
            reference = JObject.Parse(File.ReadAllText(Resolve("audio-encoder-reference.json")));
            Assert.That(reference["metadata"].Value<string>("source_commit"), Is.EqualTo(source));
            Assert.That(reference["metadata"].Value<string>("model_revision"), Is.EqualTo("914f7f89142e33e77833254d9c9b90c3cef7303b"));
            Assert.That(reference["metadata"].Value<double>("feature_rtol"), Is.EqualTo(1e-4));
            Assert.That(reference["metadata"].Value<double>("feature_atol"), Is.EqualTo(1e-5));
            var cases = (JArray)reference["cases"];
            Assert.That(cases.Select(row => row.Value<string>("id")), Is.EquivalentTo(
                new[] { "silence", "impulse", "sine", "nyquist", "noise", "padded_noise" }));
            Assert.That(cases.Select(row => row.Value<string>("id")).Distinct().Count(), Is.EqualTo(6));
            Assert.That(((JObject)reference["profiles"]).Count, Is.EqualTo(2));
            foreach (var profile in (JObject)reference["profiles"])
            {
                Assert.That(profile.Value.Value<string>("import_format"), Is.EqualTo("onnx"));
                Assert.That(profile.Value.Value<int>("feature_width"), Is.EqualTo(512));
                Audit(profile.Value.Value<string>("graph"));
                Audit(profile.Value.Value<string>("frame_mask"));
            }
            foreach (var row in cases) Audit(row.Value<string>("input_features"));
            Save();
        }

        [TestCase(BackendType.CPU)]
        [TestCase(BackendType.GPUCompute)]
        [Timeout(1200000)]
        public void SavedFp32FullEncoderMatchesAllSixNativeSignals(BackendType backend)
        {
            var condition = new JObject { ["requested_backend"] = backend.ToString(), ["success"] = false,
                ["cases"] = new JArray(), ["phase"] = "begin" };
            conditions.Add(condition); Save();
            try
            {
                if (backend == BackendType.GPUCompute)
                    Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Execute GPU; never skip.");
                foreach (var entry in (JObject)reference["profiles"])
                {
                    condition["phase"] = "load_" + entry.Key; Save();
                    var model = TextModelFile.Load(Saved(entry.Key, (JObject)entry.Value));
                    using (var worker = new Worker(model, backend))
                    {
                        Assert.That(worker.backendType, Is.EqualTo(backend));
                        const BindingFlags flags = BindingFlags.NonPublic | BindingFlags.Instance;
                        var resident = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", flags).GetValue(worker);
                        var cpu = resident.Intersect(model.layers.SelectMany(layer => layer.outputs)
                            .Concat(model.inputs.Select(input => input.index))).ToArray();
                        var flush = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", flags).GetValue(worker);
                        condition["cpu_computation_or_inputs"] = new JArray(cpu);
                        condition["readback_flush"] = new JArray(flush);
                        if (backend == BackendType.GPUCompute) { Assert.That(cpu, Is.Empty); Assert.That(flush, Is.Empty); }
                        foreach (var row in ((JArray)reference["cases"]).Where(row => row.Value<string>("profile") == entry.Key))
                        {
                            condition["phase"] = "evaluate_" + row.Value<string>("id"); Save();
                            using var input = new Tensor<float>(new TensorShape(entry.Value["feature_shape"].ToObject<int[]>()),
                                Floats(Resolve(row.Value<string>("input_features"))));
                            if (backend == BackendType.GPUCompute) ComputeTensorData.Pin(input);
                            worker.Schedule(input);
                            var output = worker.PeekOutput() as Tensor<float>;
                            Assert.That(output, Is.Not.Null);
                            Assert.That(output.backendType, Is.EqualTo(backend));
                            Assert.That(output.shape, Is.EqualTo(new TensorShape(entry.Value.Value<int>("feature_count"), 512)));
                            var values = output.DownloadToArray();
                            var expected = row["features"].ToObject<float[][]>().SelectMany(vector => vector).ToArray();
                            Assert.That(values.Length, Is.EqualTo(expected.Length));
                            int mismatches = 0; double maximum = 0, ratio = 0, minimumCosine = 1;
                            for (var token = 0; token < values.Length / 512; token++)
                            {
                                double dot = 0, left = 0, right = 0;
                                for (var channel = 0; channel < 512; channel++)
                                {
                                    var index = token * 512 + channel;
                                    Assert.That(float.IsNaN(values[index]) || float.IsInfinity(values[index]), Is.False);
                                    var difference = Math.Abs((double)values[index] - expected[index]);
                                    var bound = 1e-5 + 1e-4 * Math.Abs((double)expected[index]);
                                    maximum = Math.Max(maximum, difference); ratio = Math.Max(ratio, difference / bound);
                                    if (difference > bound) mismatches++;
                                    dot += (double)values[index] * expected[index];
                                    left += (double)values[index] * values[index]; right += (double)expected[index] * expected[index];
                                }
                                Assert.That(left * right, Is.GreaterThan(0));
                                minimumCosine = Math.Min(minimumCosine, dot / Math.Sqrt(left * right));
                            }
                            var result = new JObject { ["id"] = row["id"], ["profile"] = entry.Key,
                                ["actual_backend"] = output.backendType.ToString(), ["mismatched_elements"] = mismatches,
                                ["maximum_absolute_error"] = maximum, ["maximum_tolerance_ratio"] = ratio,
                                ["minimum_cosine"] = minimumCosine,
                                ["success"] = mismatches == 0 && minimumCosine >= 0.999999 };
                            ((JArray)condition["cases"]).Add(result); Save();
                        }
                    }
                    model = null; GC.Collect(); GC.WaitForPendingFinalizers();
                }
                Assert.That(((JArray)condition["cases"]).Count, Is.EqualTo(6));
                Assert.That(((JArray)condition["cases"]).All(row => row.Value<bool>("success")), Is.True,
                    "Every original signal must satisfy the unchanged numeric conditions.");
                condition["success"] = true; condition["phase"] = "completed";
            }
            catch (Exception error) { condition["error"] = error.ToString(); throw; }
            finally { Save(); }
        }

        string Saved(string id, JObject profile)
        {
            if (saved.TryGetValue(id, out var existing)) return existing;
            report["import_phase"] = id; Save();
            var assetPath = "Assets/AudioEncoderGenerated/" + id + ".onnx";
            Directory.CreateDirectory(Path.GetDirectoryName(assetPath));
            File.Copy(Resolve(profile.Value<string>("graph")), assetPath, true);
            AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceSynchronousImport);
            var asset = AssetDatabase.LoadAssetAtPath<ModelAsset>(assetPath);
            Assert.That(asset, Is.Not.Null);
            var model = ModelLoader.Load(asset);
            Assert.That(model.inputs.Count, Is.EqualTo(1)); Assert.That(model.outputs.Count, Is.EqualTo(1));
            var path = Path.Combine(root, id + "-fp32.sentis"); TextModelFile.Save(model, path);
            saved.Add(id, path); model = null; Resources.UnloadAsset(asset); GC.Collect(); GC.WaitForPendingFinalizers();
            return path;
        }
        [OneTimeTearDown]
        public void Complete()
        {
            if (report == null) return;
            report["success"] = conditions.Count == 2 && conditions.All(row => row.Value<bool>("success"));
            report["completed_utc"] = DateTime.UtcNow.ToString("O"); Save();
        }
        string Resolve(string name)
        {
            var path = Path.GetFullPath(Path.Combine(root, name));
            Assert.That(path.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.Ordinal), Is.True); return path;
        }
        void Audit(string name)
        {
            var spec = validation["files"][name]; Assert.That(spec, Is.Not.Null);
            var path = Resolve(name); Assert.That(new FileInfo(path).Length, Is.EqualTo(spec.Value<long>("bytes")));
            Assert.That(Hash(path), Is.EqualTo(spec.Value<string>("sha256")));
        }
        static float[] Floats(string path)
        {
            var bytes = File.ReadAllBytes(path); Assert.That(bytes.Length % 4, Is.Zero);
            var values = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length); return values;
        }
        static string Hash(string path)
        {
            var fast = AppDomain.CurrentDomain.GetAssemblies().Select(assembly => assembly.GetType("EmbeddingGemma.Samples.TextSearchFileHash"))
                .FirstOrDefault(type => type != null);
            if (fast != null) return (string)fast.GetMethod("Compute").Invoke(null, new object[] { path, null });
            using var file = File.OpenRead(path); using var hash = SHA256.Create();
            return BitConverter.ToString(hash.ComputeHash(file)).Replace("-", "").ToLowerInvariant();
        }
        void Save() { Directory.CreateDirectory(Path.GetDirectoryName(resultPath)); File.WriteAllText(resultPath, report.ToString()); }
    }
}
