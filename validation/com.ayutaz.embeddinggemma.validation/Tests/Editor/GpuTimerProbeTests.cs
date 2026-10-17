using System;
using System.Collections;
using System.IO;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class GpuTimerProbeTests
    {
        [UnityTearDown]
        public IEnumerator ReturnToEditMode()
        { if (Application.isPlaying) yield return new ExitPlayMode(); }

        [UnityTest]
        public IEnumerator OpaqueGpuCallCanBeTimedByGpuEnabledSamplerWithoutChangingItsCommandBuffer()
        {
            yield return new EnterPlayMode();
            var report = new JObject { ["success"] = false, ["real_model_qualified"] = false, ["opaque_timing_requested"] = true };
            var output = Environment.GetEnvironmentVariable("EG_GPU_TIMER_REPORT");
            Assert.That(output, Is.Not.Null.And.Not.Empty); Assert.That(File.Exists(output), Is.False);
            try
            {
                yield return GpuTimerProbe.Run(report);
                Assert.That(report.Value<bool>("available"), Is.True, "First confirm command-buffer GPU calibration.");
                Assert.That(report.Value<long>("opaque_gpu_elapsed_ns"), Is.GreaterThan(0),
                    "A CPU sample is not proof of GPU timing around opaque GPU calls.");
                Assert.That(report["output"].ToObject<float[]>(), Is.EqualTo(new[] {2f, 3f}));
                report["success"] = true;
            }
            finally { File.WriteAllText(output, report.ToString()); }
        }

        [UnityTest]
        public IEnumerator TinyGpuMarkerReportsActualGpuNanosecondsSeparatelyFromSubmissionWallTime()
        {
            yield return new EnterPlayMode();
            var report = new JObject { ["success"] = false, ["real_model_qualified"] = false };
            var output = Environment.GetEnvironmentVariable("EG_GPU_TIMER_REPORT");
            Assert.That(output, Is.Not.Null.And.Not.Empty);
            Assert.That(File.Exists(output), Is.False, "Preserve earlier calibration evidence.");
            try
            {
                yield return GpuTimerProbe.Run(report);
                Assert.That(report.Value<bool>("available"), Is.True, "Real GPU timer unavailable; CPU wall time is not a substitute.");
                Assert.That(report.Value<string>("actual_backend"), Is.EqualTo("GPUCompute"));
                Assert.That(report.Value<long>("gpu_elapsed_ns"), Is.GreaterThan(0));
                Assert.That(report.Value<int>("gpu_sample_blocks"), Is.GreaterThan(0));
                Assert.That(report.Value<long>("submission_wall_ns"), Is.GreaterThanOrEqualTo(0));
                Assert.That(report["output"].ToObject<float[]>(), Is.EqualTo(new[] {2f, 3f}));
                Assert.That(report.Value<bool>("real_model_qualified"), Is.False);
                report["success"] = true;
            }
            finally { File.WriteAllText(output, report.ToString()); }
        }
    }
}
