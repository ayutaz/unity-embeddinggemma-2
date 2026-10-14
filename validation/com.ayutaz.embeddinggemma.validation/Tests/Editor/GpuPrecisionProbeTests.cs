using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using EmbeddingGemma.Tests;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEditor;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class GpuPrecisionProbeTests
    {
        JObject specification, report;
        JArray rows;
        string root, output;

        [OneTimeSetUp]
        public void AuditWeightsFreeProbe()
        {
            Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Actual GPU required; no CPU substitute.");
            root = Path.Combine(Application.dataPath, "GpuPrecisionProbe");
            specification = JObject.Parse(File.ReadAllText(Path.Combine(root, "precision-probe.json")));
            Assert.That(specification.Value<bool>("gpu_validated"), Is.False);
            var source = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var locked = JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath, "../Packages/packages-lock.json")));
            foreach (var package in new[] { "com.ayutaz.embeddinggemma", "com.ayutaz.embeddinggemma.validation" })
                Assert.That(locked["dependencies"][package].Value<string>("hash"), Is.EqualTo(source));
            foreach (var entry in (JObject)specification["files"])
            {
                using var stream = File.OpenRead(Path.Combine(root, entry.Key));
                Assert.That(PlayerFileHash.Compute(stream, out _), Is.EqualTo(entry.Value.Value<string>("sha256")));
                Assert.That(new FileInfo(Path.Combine(root, entry.Key)).Length, Is.EqualTo(entry.Value.Value<long>("bytes")));
            }
            rows = new JArray();
            report = new JObject { ["success"] = false, ["scope"] = "weights_free_gpu_fma_and_welford_components_only",
                ["source"] = source, ["unity"] = Application.unityVersion, ["sentis"] = "2.6.1",
                ["gpu"] = SystemInfo.graphicsDeviceName, ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(),
                ["full_encoder_validated"] = false, ["end_to_end_validated"] = false, ["cases"] = rows };
            output = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/gpu-precision-probe-" + source + ".json"));
            Assert.That(File.Exists(output), Is.False, "Preserve previous probe evidence.");
            Save();
        }

        [TestCase("fma")]
        [TestCase("welford32")]
        [TestCase("welford128")]
        public void GpuCandidateMatchesIndependentNativeExpected(string name) => Execute(name, false);

        [Test]
        public void BaselineGpuExposesSeparateRoundingCounterexample() => Execute("fma", true);

        void Execute(string name, bool baseline)
        {
            var modelName = baseline ? "baseline-fma" : name;
            var row = new JObject { ["case"] = modelName, ["success"] = false, ["requested_backend"] = "GPUCompute" };
            rows.Add(row); Save();
            try
            {
                var asset = AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuPrecisionProbe/" + modelName + ".onnx");
                Assert.That(asset, Is.Not.Null, "The standard ONNX probe must import before execution.");
                var model = ModelLoader.Load(asset);
                using var worker = new Worker(model, BackendType.GPUCompute);
                var cpu = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", BindingFlags.NonPublic | BindingFlags.Instance).GetValue(worker);
                var count = cpu.Intersect(model.layers.SelectMany(l => l.outputs).Concat(model.inputs.Select(i => i.index))).Count();
                var flush = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", BindingFlags.NonPublic | BindingFlags.Instance).GetValue(worker);
                row["cpu_computation_or_input_count"] = count; row["readback_flush_count"] = flush.Count;
                Assert.That(count, Is.Zero); Assert.That(flush.Count, Is.Zero);
                using var input = new Tensor<float>(new TensorShape(specification["cases"][name]["input_shape"].ToObject<int[]>()), Floats(name + "-input.bin"));
                ComputeTensorData.Pin(input); worker.Schedule(input);
                var tensor = (Tensor<float>)worker.PeekOutput();
                row["actual_backend"] = tensor.backendType.ToString();
                Assert.That(tensor.backendType, Is.EqualTo(BackendType.GPUCompute));
                var actual = tensor.DownloadToArray(); var expected = Floats(name + "-expected.bin");
                Assert.That(actual.Length, Is.EqualTo(expected.Length));
                Assert.That(actual.All(v => !float.IsNaN(v) && !float.IsInfinity(v)), Is.True);
                var bits = 0; var mismatches = 0; double maximum = 0;
                for (var i = 0; i < actual.Length; i++)
                {
                    if (BitConverter.SingleToInt32Bits(actual[i]) != BitConverter.SingleToInt32Bits(expected[i])) bits++;
                    var difference = Math.Abs((double)actual[i] - expected[i]); maximum = Math.Max(maximum, difference);
                    if (difference > 1e-5 + 1e-4 * Math.Abs((double)expected[i])) mismatches++;
                }
                row["exact_differences"] = bits; row["mismatched_elements"] = mismatches; row["maximum_absolute_error"] = maximum; Save();
                if (baseline) Assert.That(bits, Is.GreaterThan(0), "The baseline must expose the rounding difference.");
                else if (name == "fma") Assert.That(bits, Is.Zero, "Single-rounding outputs must match exact bits.");
                else Assert.That(mismatches, Is.Zero, "Keep the original native component tolerance.");
                row["success"] = true;
            }
            catch (Exception error) { row["error"] = error.ToString(); throw; }
            finally { Save(); }
        }

        float[] Floats(string name) { var bytes = File.ReadAllBytes(Path.Combine(root, name)); var values = new float[bytes.Length/4]; Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length); return values; }
        void Save() { Directory.CreateDirectory(Path.GetDirectoryName(output)); File.WriteAllText(output, report.ToString()); }
        [OneTimeTearDown]
        public void CompleteReport() { if (report == null) return; report["success"] = rows.Count == 4 && rows.All(r => r.Value<bool>("success")); Save(); }
    }
}
