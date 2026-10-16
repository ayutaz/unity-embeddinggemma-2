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
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    // Existing audited saved graph and official PCM references; no weight download or new export.
    public sealed class AudioPreprocessorRuntimeReferenceTests
    {
        static string DirectoryPath => Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/audio-preprocessing"));
        JObject reference, report;
        JArray conditions;
        HashSet<string> audited;

        [OneTimeSetUp]
        public void AuditExistingSavedGraphAndIndependentReferences()
        {
            conditions = new JArray();
            report = new JObject { ["success"] = false, ["stage"] = "audio_preprocessing_runtime_only",
                ["real_preprocessing_validated"] = false, ["encoders_validated"] = false,
                ["end_to_end_validated"] = false, ["fresh_consumer_validated"] = false,
                ["runtime_type_assembly"] = typeof(AudioPreprocessor).Assembly.GetName().Name,
                ["unity"] = Application.unityVersion, ["gpu"] = SystemInfo.graphicsDeviceName,
                ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(),
                ["started_utc"] = DateTime.UtcNow.ToString("O"), ["conditions"] = conditions };
            Save();
            Assert.That(report.Value<string>("runtime_type_assembly"), Is.EqualTo("EmbeddingGemma.Runtime"));
            var validation = JObject.Parse(File.ReadAllText(StagePath("audio-validation.json")));
            Assert.That(validation.Value<bool>("success"), Is.True);
            Assert.That(validation.Value<string>("stage"), Is.EqualTo("audio_preprocessing_only"));
            audited = new HashSet<string>();
            foreach (var entry in (JObject)validation["files"])
            {
                Assert.That(new FileInfo(StagePath(entry.Key)).Length, Is.EqualTo(entry.Value.Value<long>("bytes")));
                Assert.That(Hash(StagePath(entry.Key)), Is.EqualTo(entry.Value.Value<string>("sha256")));
                audited.Add(entry.Key);
            }
            Assert.That(audited, Is.SupersetOf(new[] { "audio-preprocessor.pt2", "audio-reference.json", "processor_config.json" }));
            reference = JObject.Parse(File.ReadAllText(StagePath("audio-reference.json")));
            var metadata = reference["metadata"];
            var source = File.ReadAllText(StagePath("expected-source.txt")).Trim();
            Assert.That(source, Does.Match("^[0-9a-f]{40}$"));
            Assert.That(metadata.Value<string>("source_commit"), Is.EqualTo(source));
            Assert.That(metadata.Value<string>("reference"), Is.EqualTo("official_Gemma4AudioFeatureExtractor"));
            Assert.That(metadata.Value<string>("model_revision"), Is.EqualTo("914f7f89142e33e77833254d9c9b90c3cef7303b"));
            Assert.That(metadata.Value<double>("feature_atol"), Is.EqualTo(0.004));
            Assert.That(metadata.Value<double>("feature_rtol"), Is.EqualTo(1e-4));
            Assert.That(metadata.Value<string>("processor_sha256"), Is.EqualTo(Hash(StagePath("processor_config.json"))));
            Assert.That(((JArray)reference["cases"]).Select(row => row.Value<string>("id")),
                Is.EquivalentTo(new[] { "silence", "impulse", "sine", "nyquist", "noise", "padded_noise" }));
            var previous = JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath, "../artifacts/audio-preprocessing-results/results.json")));
            Assert.That(previous.Value<bool>("success"), Is.True, "Requires the existing saved-model acceptance receipt.");
            Assert.That(previous.Value<string>("reference_sha256"), Is.EqualTo(Hash(StagePath("audio-reference.json"))));
            Assert.That(previous.Value<string>("graph_sha256"), Is.EqualTo(Hash(StagePath("audio-preprocessor.pt2"))));
            Assert.That(Hash(StagePath("audio-preprocessor.sentis")), Is.EqualTo(previous.Value<string>("saved_sha256")));
            report["reference_source"] = source;
            report["reference_sha256"] = Hash(StagePath("audio-reference.json"));
            report["saved_graph_sha256"] = Hash(StagePath("audio-preprocessor.sentis"));
            report["saved_graph_bytes"] = new FileInfo(StagePath("audio-preprocessor.sentis")).Length;
            Save();
        }

        [OneTimeTearDown]
        public void CompleteReport()
        {
            if (report == null) return;
            var passed = conditions.Count == 2 && conditions.All(row => row.Value<bool>("success"));
            report["success"] = passed; report["real_preprocessing_validated"] = passed;
            report["completed_utc"] = DateTime.UtcNow.ToString("O"); Save();
        }

        [TestCase(BackendType.CPU)]
        [TestCase(BackendType.GPUCompute)]
        [Timeout(600000)]
        public void GitRuntimeMatchesSixSignalsAndChainsOwnedOutputs(BackendType backend)
        {
            var cases = new JArray();
            var condition = new JObject { ["requested_backend"] = backend.ToString(),
                ["success"] = false, ["cases"] = cases };
            conditions.Add(condition); Save();
            try
            {
                if (backend == BackendType.GPUCompute)
                    Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Actual GPU required; never skip.");
                var model = TextModelFile.Load(StagePath("audio-preprocessor.sentis"));
                using var preprocessor = new AudioPreprocessor(model, backend);
                condition["actual_backend"] = preprocessor.Backend.ToString();
                var flags = BindingFlags.NonPublic | BindingFlags.Instance;
                var worker = (Worker)typeof(AudioPreprocessor).GetField("worker", flags).GetValue(preprocessor);
                AuditFallback(worker, model, backend, condition);
                var graph = new FunctionalGraph();
                var features = graph.AddInput<float>(new TensorShape(1, 1119, 128), "features");
                var mask = graph.AddInput<int>(new TensorShape(1, 1119), "frame_mask");
                // Consume every feature and mask element without an unrelated large FP32 reduction.
                var joined = features + Functional.Unsqueeze(mask.Float(), 2);
                var consumer = graph.Compile(joined);
                using var next = new Worker(consumer, backend);
                var consumerAudit = new JObject(); AuditFallback(next, consumer, backend, consumerAudit);
                condition["downstream_audit"] = consumerAudit;
                foreach (var row in (JArray)reference["cases"])
                {
                    var item = new JObject { ["id"] = row.Value<string>("id"), ["success"] = false };
                    cases.Add(item); Save();
                    var pcm = Read<float>(row.Value<string>("waveform"), 179200);
                    var input = new AudioPcmInput(pcm.Take(row.Value<int>("valid_samples")).ToArray());
                    using var result = preprocessor.Process(input);
                    Assert.That(result.Features.backendType, Is.EqualTo(backend));
                    Assert.That(result.FrameMask.backendType, Is.EqualTo(backend));
                    next.SetInput("features", result.Features); next.SetInput("frame_mask", result.FrameMask); next.Schedule();
                    var final = next.PeekOutput() as Tensor<float>;
                    Assert.That(final.backendType, Is.EqualTo(backend));
                    var expected = Read<float>(row.Value<string>("features"), 1119 * 128);
                    var expectedMask = Read<int>(row.Value<string>("frame_mask"), 1119);
                    Assert.That(final.shape, Is.EqualTo(new TensorShape(1, 1119, 128)));
                    var chained = final.DownloadToArray();
                    double chainMaximum = 0; var chainMismatches = 0;
                    for (var index = 0; index < chained.Length; index++)
                    {
                        var expectedValue = expected[index] + expectedMask[index / 128];
                        Assert.That(float.IsNaN(chained[index]) || float.IsInfinity(chained[index]), Is.False);
                        var difference = Math.Abs((double)chained[index] - expectedValue);
                        // Same original preprocessing bound, plus a fixed 2e-6 allowance
                        // for the single FP32 addition. Original features are checked below unchanged.
                        var tolerance = 0.004 + 1e-4 * Math.Abs(expected[index]) + 2e-6;
                        chainMaximum = Math.Max(chainMaximum, difference);
                        if (difference > tolerance) chainMismatches++;
                    }
                    item["downstream_maximum_absolute_error"] = chainMaximum;
                    item["downstream_mismatched_elements"] = chainMismatches;
                    Assert.That(chainMismatches, Is.EqualTo(0));
                    item["feature_download_before_chaining"] = false;
                    item["mask_download_before_chaining"] = false;
                    // These diagnostic reads occur only after the downstream Worker has consumed both tensors.
                    var actual = result.Features.DownloadToArray();
                    var actualMask = result.FrameMask.DownloadToArray();
                    double maximum = 0, ratio = 0; var mismatches = 0;
                    for (var index = 0; index < actual.Length; index++)
                    {
                        Assert.That(float.IsNaN(actual[index]) || float.IsInfinity(actual[index]), Is.False);
                        var difference = Math.Abs((double)actual[index] - expected[index]);
                        var tolerance = 0.004 + 1e-4 * Math.Abs(expected[index]);
                        maximum = Math.Max(maximum, difference); ratio = Math.Max(ratio, difference / tolerance);
                        if (difference > tolerance) mismatches++;
                    }
                    item["maximum_absolute_error"] = maximum; item["maximum_tolerance_ratio"] = ratio;
                    item["mismatched_elements"] = mismatches;
                    Assert.That(mismatches, Is.EqualTo(0), item.ToString());
                    Assert.That(actualMask, Is.EqualTo(expectedMask));
                    Assert.That(result.ValidFrameCount, Is.EqualTo(expectedMask.Sum()));
                    item["success"] = true; Save();
                }
                Assert.That(cases.Count, Is.EqualTo(6)); condition["success"] = true;
            }
            catch (Exception error) { condition["error"] = error.ToString(); throw; }
            finally { Save(); }
        }

        static void AuditFallback(Worker worker, Model model, BackendType backend, JObject report)
        {
            Assert.That(worker.backendType, Is.EqualTo(backend));
            var flags = BindingFlags.NonPublic | BindingFlags.Instance;
            var cpu = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", flags).GetValue(worker);
            var inputsOrCompute = cpu.Intersect(model.layers.SelectMany(layer => layer.outputs)
                .Concat(model.inputs.Select(input => input.index))).OrderBy(index => index).ToArray();
            var flush = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", flags).GetValue(worker);
            report["cpu_computation_or_input_indices"] = new JArray(inputsOrCompute);
            report["cpu_readback_flush_indices"] = new JArray(flush.OrderBy(index => index));
            if (backend == BackendType.GPUCompute) { Assert.That(inputsOrCompute, Is.Empty); Assert.That(flush, Is.Empty); }
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
            var bytes = File.ReadAllBytes(StagePath(name)); Assert.That(bytes.Length, Is.EqualTo(count * 4));
            var values = new T[count]; Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length); return values;
        }

        static string Hash(string path)
        {
            using var stream = File.OpenRead(path); using var hash = SHA256.Create();
            return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }

        void Save()
        {
            var destination = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/audio-preprocessing-runtime-results"));
            Directory.CreateDirectory(destination);
            File.WriteAllText(Path.Combine(destination, "results.json"), report.ToString(Formatting.Indented));
        }
    }
}
