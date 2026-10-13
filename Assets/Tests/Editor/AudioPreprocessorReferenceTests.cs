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
    // Unweighted PCM preprocessing, independently of audio encoder / backbone.
    public sealed class AudioPreprocessorReferenceTests
    {
        static string DirectoryPath => Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/audio-preprocessing"));
        JObject reference, report;
        JArray conditions;
        HashSet<string> audited;
        string savedPath;

        [OneTimeSetUp]
        public void AuditOfficialPcmReferencesAndSavePreprocessingModel()
        {
            conditions = new JArray();
            report = new JObject { ["success"] = false, ["stage"] = "sentis_audio_preprocessing_only",
                ["unity_validated"] = false, ["encoders_validated"] = false, ["end_to_end_validated"] = false,
                ["gpu_preprocessing_validated"] = false,
                ["diagnostic_output_readback"] = true, ["expected_conditions"] = 2, ["expected_cases_per_condition"] = 6,
                ["unity"] = Application.unityVersion, ["gpu"] = SystemInfo.graphicsDeviceName,
                ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(), ["started_utc"] = DateTime.UtcNow.ToString("O"),
                ["conditions"] = conditions };
            Save();
            var expectedSource = File.ReadAllText(Path.Combine(DirectoryPath, "expected-source.txt")).Trim();
            Assert.That(expectedSource, Does.Match("^[0-9a-f]{40}$"));
            var validation = JObject.Parse(File.ReadAllText(Path.Combine(DirectoryPath, "audio-validation.json")));
            Assert.That(validation.Value<bool>("success"), Is.True);
            Assert.That(validation.Value<string>("stage"), Is.EqualTo("audio_preprocessing_only"));
            Assert.That(validation.Value<int>("case_count"), Is.EqualTo(6));
            audited = new HashSet<string>();
            foreach (var entry in (JObject)validation["files"])
            {
                var path = StagePath(entry.Key);
                Assert.That(new FileInfo(path).Length, Is.EqualTo(entry.Value.Value<long>("bytes")), entry.Key);
                Assert.That(Hash(path), Is.EqualTo(entry.Value.Value<string>("sha256")), entry.Key);
                audited.Add(entry.Key);
            }
            Assert.That(audited, Is.SupersetOf(new[] { "audio-preprocessor.pt2", "audio-reference.json", "processor_config.json" }));
            reference = JObject.Parse(File.ReadAllText(StagePath("audio-reference.json")));
            var metadata = reference["metadata"];
            Assert.That(metadata.Value<string>("source_commit"), Is.EqualTo(expectedSource));
            Assert.That(metadata.Value<string>("model_revision"), Is.EqualTo("914f7f89142e33e77833254d9c9b90c3cef7303b"));
            Assert.That(metadata.Value<string>("reference"), Is.EqualTo("official_Gemma4AudioFeatureExtractor"));
            Assert.That(metadata.Value<string>("fixture_provenance"), Is.EqualTo("procedurally_generated_pcm_no_user_audio"));
            Assert.That(metadata.Value<string>("processor_sha256"), Is.EqualTo(Hash(StagePath("processor_config.json"))));
            Assert.That(metadata.Value<int>("num_samples"), Is.EqualTo(179200));
            Assert.That(metadata.Value<int>("sampling_rate"), Is.EqualTo(16000));
            Assert.That(metadata["feature_shape"].ToObject<int[]>(), Is.EqualTo(new[] { 1, 1119, 128 }));
            Assert.That(metadata["sample_shape"].ToObject<int[]>(), Is.EqualTo(new[] { 1, 179200 }));
            Assert.That(metadata["frame_mask_shape"].ToObject<int[]>(), Is.EqualTo(new[] { 1, 1119 }));
            Assert.That(metadata.Value<double>("feature_atol"), Is.EqualTo(0.004));
            Assert.That(metadata.Value<double>("feature_rtol"), Is.EqualTo(1e-4));
            Assert.That(((JArray)reference["cases"]).Select(row => row.Value<string>("id")),
                Is.EquivalentTo(new[] { "silence", "impulse", "sine", "nyquist", "noise", "padded_noise" }));
            foreach (var row in (JArray)reference["cases"])
                foreach (var field in new[] { "waveform", "sample_mask", "features", "frame_mask" })
                    Assert.That(audited, Does.Contain(row.Value<string>(field)));
            report["reference_source"] = expectedSource;
            report["reference_sha256"] = Hash(StagePath("audio-reference.json"));
            report["graph_sha256"] = Hash(StagePath("audio-preprocessor.pt2"));
            Save();
            const string assetPath = "Assets/AudioPreprocessingGenerated/audio-preprocessor.pt2";
            Directory.CreateDirectory(Path.GetDirectoryName(assetPath));
            if (!File.Exists(assetPath) || Hash(assetPath) != Hash(StagePath("audio-preprocessor.pt2")))
                File.Copy(StagePath("audio-preprocessor.pt2"), assetPath, true);
            AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceSynchronousImport);
            var asset = AssetDatabase.LoadAssetAtPath<ModelAsset>(assetPath);
            Assert.That(asset, Is.Not.Null, "Sentis must import the fixed FFT / mel graph.");
            var model = ModelLoader.Load(asset);
            savedPath = Path.Combine(DirectoryPath, "audio-preprocessor.sentis");
            TextModelFile.Save(model, savedPath);
            report["saved_sha256"] = Hash(savedPath); report["saved_bytes"] = new FileInfo(savedPath).Length;
            Save();
        }

        [OneTimeTearDown]
        public void CompleteReport()
        {
            if (report == null) return;
            var gpuPassed = conditions.Any(row => row.Value<string>("requested_backend") == "GPUCompute" && row.Value<bool>("success"));
            report["gpu_preprocessing_validated"] = gpuPassed;
            var passed = conditions.Count == 2 && conditions.All(row => row.Value<bool>("success"));
            report["success"] = passed; report["unity_validated"] = passed;
            report["completed_utc"] = DateTime.UtcNow.ToString("O");
            Save();
        }

        [TestCase(BackendType.CPU)]
        [TestCase(BackendType.GPUCompute)]
        [Timeout(1200000)]
        public void SavedFftAndMelMatchSixOfficialSignalsAndMasks(BackendType backend)
        {
            var cases = new JArray();
            var condition = new JObject { ["requested_backend"] = backend.ToString(), ["success"] = false,
                ["phase"] = "create_worker", ["cases"] = cases };
            conditions.Add(condition); Save();
            try
            {
                if (backend == BackendType.GPUCompute) Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Actual GPU is required; never skip.");
                var model = TextModelFile.Load(savedPath);
                using var worker = new Worker(model, backend);
                condition["actual_worker_backend"] = worker.backendType.ToString();
                var resident = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", BindingFlags.NonPublic | BindingFlags.Instance).GetValue(worker);
                var cpu = resident.Intersect(model.layers.SelectMany(layer => layer.outputs).Concat(model.inputs.Select(input => input.index))).OrderBy(index => index).ToArray();
                var flush = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", BindingFlags.NonPublic | BindingFlags.Instance).GetValue(worker);
                condition["cpu_computation_or_input_indices"] = new JArray(cpu);
                condition["cpu_readback_flush_indices"] = new JArray(flush.OrderBy(index => index));
                if (backend == BackendType.GPUCompute) { Assert.That(cpu, Is.Empty); Assert.That(flush, Is.Empty); }
                foreach (var row in (JArray)reference["cases"])
                {
                    var item = new JObject { ["id"] = row.Value<string>("id"), ["success"] = false };
                    cases.Add(item); condition["phase"] = "evaluate_" + row.Value<string>("id"); Save();
                    using var waveform = new Tensor<float>(new TensorShape(1, 179200), Read<float>(row.Value<string>("waveform"), 179200));
                    using var sampleMask = new Tensor<int>(new TensorShape(1, 179200), Read<int>(row.Value<string>("sample_mask"), 179200));
                    if (backend == BackendType.GPUCompute) { ComputeTensorData.Pin(waveform); ComputeTensorData.Pin(sampleMask); }
                    worker.SetInput("waveform", waveform); worker.SetInput("sample_mask", sampleMask); worker.Schedule();
                    var features = worker.PeekOutput(model.outputs[0].name) as Tensor<float>;
                    var frameMask = worker.PeekOutput(model.outputs[1].name) as Tensor<int>;
                    Assert.That(features, Is.Not.Null); Assert.That(frameMask, Is.Not.Null);
                    Assert.That(features.shape, Is.EqualTo(new TensorShape(1, 1119, 128)));
                    Assert.That(frameMask.shape, Is.EqualTo(new TensorShape(1, 1119)));
                    Assert.That(features.backendType, Is.EqualTo(backend)); Assert.That(frameMask.backendType, Is.EqualTo(backend));
                    Assert.That(waveform.backendType, Is.EqualTo(backend)); Assert.That(sampleMask.backendType, Is.EqualTo(backend));
                    item["actual_feature_backend"] = features.backendType.ToString(); item["actual_mask_backend"] = frameMask.backendType.ToString();
                    // Diagnostic reads after the preprocessing stage. Production
                    // encoder/backbone tensor chaining remains a separate gate.
                    var actual = features.DownloadToArray(); var expected = Read<float>(row.Value<string>("features"), 1119 * 128);
                    var actualMask = frameMask.DownloadToArray(); var expectedMask = Read<int>(row.Value<string>("frame_mask"), 1119);
                    double maximum = 0, ratio = 0; var mismatches = 0;
                    for (var index = 0; index < actual.Length; index++)
                    {
                        Assert.That(float.IsNaN(actual[index]) || float.IsInfinity(actual[index]), Is.False);
                        Assert.That(float.IsNaN(expected[index]) || float.IsInfinity(expected[index]), Is.False);
                        var difference = Math.Abs((double)actual[index] - expected[index]);
                        var tolerance = 0.004 + 1e-4 * Math.Abs(expected[index]);
                        maximum = Math.Max(maximum, difference); ratio = Math.Max(ratio, difference / tolerance);
                        if (difference > tolerance) mismatches++;
                    }
                    item["maximum_absolute_error"] = maximum; item["maximum_tolerance_ratio"] = ratio;
                    item["mismatched_elements"] = mismatches;
                    Assert.That(mismatches, Is.EqualTo(0), item.ToString());
                    Assert.That(actualMask, Is.EqualTo(expectedMask));
                    item["success"] = true; Save();
                }
                Assert.That(cases.Count, Is.EqualTo(6));
                condition["phase"] = "completed"; condition["success"] = true;
            }
            catch (Exception error) { condition["error"] = error.ToString(); throw; }
            finally { Save(); }
        }

        static string StagePath(string name)
        {
            var path = Path.GetFullPath(Path.Combine(DirectoryPath, name));
            Assert.That(path.StartsWith(DirectoryPath + Path.DirectorySeparatorChar, StringComparison.Ordinal), Is.True);
            return path;
        }

        T[] Read<T>(string name, int count) where T : struct
        {
            Assert.That(audited, Does.Contain(name));
            Assert.That(BitConverter.IsLittleEndian, Is.True);
            var bytes = File.ReadAllBytes(StagePath(name)); Assert.That(bytes, Has.Length.EqualTo(count * 4));
            var values = new T[count]; Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length); return values;
        }

        static string Hash(string path)
        {
            using var stream = File.OpenRead(path); using var hash = SHA256.Create();
            return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }

        void Save()
        {
            var destination = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/audio-preprocessing-results"));
            Directory.CreateDirectory(destination);
            File.WriteAllText(Path.Combine(destination, "results.json"), report.ToString(Formatting.Indented));
        }
    }
}
