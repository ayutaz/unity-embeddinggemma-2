using System;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class GpuConstantProbeRunnerTests
    {
        [TestCase(false)] [TestCase(true)]
        public void RunnerRecordsBothBackendsAndNeverAcceptsInjectedChecks(bool corruptGpu)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies()
                .Select(assembly => assembly.GetType("EmbeddingGemma.Validation.GpuConstantProbeRunner"))
                .FirstOrDefault(value => value != null);
            Assert.That(type, Is.Not.Null, "Executable probe runner is missing.");
            var calls = 0; var exit = -1;
            Func<BackendType, JObject> evaluate = backend =>
            {
                calls++;
                return new JObject
                {
                    ["success"] = true, ["actual_backend"] = backend.ToString(), ["output_elements"] = 2097153,
                    ["mismatches"] = backend == BackendType.GPUCompute && corruptGpu ? 1 : 0,
                    ["uploaded_constants"] = backend == BackendType.CPU ? 0 : 1,
                    ["upload_calls"] = backend == BackendType.CPU ? 0 : 2,
                    ["maximum_upload_bytes"] = backend == BackendType.CPU ? 0 : 8388608,
                    ["borrowed_model_reusable"] = true
                };
            };
            var output = Path.GetFullPath(Path.Combine("artifacts", "gpu-probe-runner-" + Guid.NewGuid().ToString("N"), "results.json"));
            var marker = new JObject
            {
                ["source_commit"] = new string('a', 40), ["injected_build"] = false,
                ["scripting_backend"] = "IL2CPP", ["stripping"] = "High", ["compiler"] = "Release", ["graphics_api"] = "Direct3D12",
                ["source_sha256"] = new JObject { ["runtime/GpuConstantWorker.cs"] = new string('b', 64), ["runtime/link.xml"] = new string('c', 64) }
            };
            type.GetMethod("Run").Invoke(null, new object[] { marker, output, evaluate, new Action<int>(value => exit = value) });
            var report = JObject.Parse(File.ReadAllText(output));
            Assert.That(calls, Is.EqualTo(2)); Assert.That(exit, Is.EqualTo(1));
            Assert.That(report.Value<bool>("success"), Is.False);
            Assert.That(report.Value<bool>("checks_passed"), Is.EqualTo(!corruptGpu));
            Assert.That(report.Value<bool>("injected_checks"), Is.True);
            Assert.That(report["build_info"]["source_sha256"], Is.EqualTo(marker["source_sha256"]));
            Assert.That(report["cases"].Count(), Is.EqualTo(2));
            Assert.That(report.Value<string>("completed_utc"), Is.Not.Empty);
        }
    }
}
