using System;
using System.Collections.Generic;
using System.Diagnostics;
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
    public static class GpuExecutionProfiler
    {
        public static JObject Run(bool real)
        {
            Assert.That(SystemInfo.supportsComputeShaders, Is.True, "GPU required; no CPU fallback or skip.");
            var source = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var output = Path.GetFullPath(Path.Combine(Application.dataPath,
                "../artifacts/gpu-execution-profile/" + (real ? "real-square" : "tiny") + "-" + source + ".json"));
            Assert.That(File.Exists(output), Is.False, "Do not overwrite an existing measurement.");
            var phases = new JArray();
            var report = new JObject { ["success"] = false, ["real_model"] = real,
                ["source"] = source, ["unity"] = Application.unityVersion, ["sentis"] = "2.6.1",
                ["gpu"] = SystemInfo.graphicsDeviceName, ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(),
                ["requested_backend"] = "GPUCompute", ["numeric_acceptance_claimed"] = false,
                ["schedule_is_submission_time"] = true, ["final_readback_includes_gpu_wait"] = true,
                ["phases"] = phases, ["started_utc"] = DateTime.UtcNow.ToString("O") };
            using var process = Process.GetCurrentProcess();
            void Save() { Directory.CreateDirectory(Path.GetDirectoryName(output)); File.WriteAllText(output, report.ToString()); }
            void Phase(string name, Action action)
            {
                process.Refresh(); var cpu = process.TotalProcessorTime; var watch = Stopwatch.StartNew();
                var row = new JObject { ["name"] = name, ["completed"] = false };
                phases.Add(row); report["active_phase"] = name; Save();
                try { action(); row["completed"] = true; }
                finally
                {
                    process.Refresh(); row["wall_seconds"] = watch.Elapsed.TotalSeconds;
                    row["cpu_seconds"] = (process.TotalProcessorTime - cpu).TotalSeconds;
                    row["working_set_bytes"] = process.WorkingSet64;
                    row["peak_working_set_bytes"] = process.PeakWorkingSet64;
                    row["unity_allocated_bytes"] = UnityEngine.Profiling.Profiler.GetTotalAllocatedMemoryLong(); Save();
                }
            }
            Worker worker = null; Tensor<float> input = null;
            Model original = null, model = null; float[] values = null, actual = null;
            var shape = new TensorShape(1, 2);
            try
            {
                var locked = JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath, "../Packages/packages-lock.json")));
                foreach (var package in new[] { "com.ayutaz.embeddinggemma", "com.ayutaz.embeddinggemma.validation" })
                    Assert.That(locked["dependencies"][package].Value<string>("hash"), Is.EqualTo(source));
                Phase("load", () => {
                    if (real)
                    {
                        var root = Path.GetFullPath(Path.Combine(Application.dataPath, "../../k/artifacts/vision-encoder"));
                        var modelPath = Path.Combine(root, "square-fp32.sentis");
                        using (var stream = File.OpenRead(modelPath))
                            Assert.That(PlayerFileHash.Compute(stream, out _), Is.EqualTo("4ecd0106463e833588c7584a58ea4973fdbab1eb1a56389c2f1597d3e7df34e7"));
                        var referencePath = Path.Combine(root, "vision-reference.json");
                        using (var stream = File.OpenRead(referencePath))
                            Assert.That(PlayerFileHash.Compute(stream, out _), Is.EqualTo("3dfccd1a487ce04df3f8d640483ca045f7a7a56231bc1096ad0f845c923a29c6"));
                        var reference = JObject.Parse(File.ReadAllText(referencePath));
                        var row = ((JArray)reference["cases"]).First(r => r.Value<string>("profile") == "square");
                        var bytes = File.ReadAllBytes(Path.Combine(root, row.Value<string>("pixels")));
                        values = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
                        shape = new TensorShape(1, 2520, 768);
                        original = TextModelFile.Load(modelPath);
                        report["case_id"] = row.Value<string>("id");
                        report["model_sha256"] = "4ecd0106463e833588c7584a58ea4973fdbab1eb1a56389c2f1597d3e7df34e7";
                    }
                    else
                    {
                        var graph = new FunctionalGraph(); var x = graph.AddInput<float>(shape, "input");
                        original = graph.Compile(x + 1f); values = new[] { 1f, 2f };
                    }
                    report["original_layer_count"] = original.layers.Count;
                });
                Phase("transform", () => {
                    model = real ? CompensatedModel.CreateWithCompensatedNorms(original, 64, 4) : original;
                    report["candidate_layer_count"] = model.layers.Count;
                    report["transform"] = real ? "compensated_projection64_centered_norm4" : "identity_tiny";
                });
                Phase("worker", () => {
                    worker = new Worker(model, BackendType.GPUCompute);
                    Assert.That(worker.backendType, Is.EqualTo(BackendType.GPUCompute));
                    var resident = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(worker);
                    var count = resident.Intersect(model.layers.SelectMany(l => l.outputs).Concat(model.inputs.Select(i => i.index))).Count();
                    var flush = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(worker);
                    report["cpu_computation_or_input_count"] = count; report["readback_flush_count"] = flush.Count;
                    Assert.That(count, Is.Zero); Assert.That(flush.Count, Is.Zero);
                });
                Phase("upload", () => { input = new Tensor<float>(shape, values); ComputeTensorData.Pin(input); });
                Phase("schedule", () => {
                    worker.Schedule(input);
                    var tensor = (Tensor<float>)worker.PeekOutput();
                    report["actual_backend"] = tensor.backendType.ToString();
                    Assert.That(tensor.backendType, Is.EqualTo(BackendType.GPUCompute));
                });
                Phase("final_readback", () => {
                    actual = ((Tensor<float>)worker.PeekOutput()).DownloadToArray();
                    Assert.That(actual.Length, Is.GreaterThan(0));
                    Assert.That(actual.All(v => !float.IsNaN(v) && !float.IsInfinity(v)), Is.True);
                    report["output_elements"] = actual.Length;
                    if (!real) report["output"] = JArray.FromObject(actual);
                });
                Phase("dispose", () => { input.Dispose(); input = null; worker.Dispose(); worker = null; });
                report["success"] = true; report["completed_utc"] = DateTime.UtcNow.ToString("O");
                return report;
            }
            catch (Exception error) { report["error"] = error.ToString(); throw; }
            finally { input?.Dispose(); worker?.Dispose(); Save(); }
        }
    }
}
