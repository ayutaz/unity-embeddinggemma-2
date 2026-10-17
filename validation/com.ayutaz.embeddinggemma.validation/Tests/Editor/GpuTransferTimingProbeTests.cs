using System;
using System.Collections;
using System.IO;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class GpuTransferTimingProbeTests
    {
        [UnityTearDown]
        public IEnumerator ReturnToEditMode()
        { if (Application.isPlaying) yield return new ExitPlayMode(); }

        [UnityTest, Timeout(180000)]
        public IEnumerator TransferAndRetirementMarkersRequireTheirOwnGpuEvidence()
        {
            var output = Environment.GetEnvironmentVariable("EG_GPU_TRANSFER_TIMING_REPORT");
            Assert.That(output, Is.Not.Null.And.Not.Empty);
            Assert.That(File.Exists(output), Is.False);
            yield return new EnterPlayMode();
            var report = new JObject { ["success"] = false, ["real_model_qualified"] = false,
                ["runtime_source"] = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE"),
                ["validation_source"] = Environment.GetEnvironmentVariable("EG_VALIDATION_SOURCE"),
                ["scope"] = "Tiny transfer/retirement calibration only, not real model load/release or pure kernel time." };
            try
            {
                var control = new JObject(); report["dispatch_control"] = control;
                yield return GpuTimerProbe.Run(control);
                Assert.That(control.Value<bool>("available"), Is.True, "A working dispatch GPU clock is required first.");
                Assert.That(control["output"].ToObject<float[]>(), Is.EqualTo(new[] {2f, 3f}));
                yield return ObserveTransfers(report);
                Assert.That(report["transfer_observations"] is JArray rows && rows.Count == 4, Is.True,
                    "Transfer and retirement evidence must be recorded independently of the dispatch control.");
                foreach (var row in (JArray)report["transfer_observations"])
                {
                    Assert.That(row.Value<bool>("data_integrity_verified_outside_marker"), Is.True);
                    Assert.That(row.Value<bool>("gpu_available"), Is.True,
                        "Missing transfer/retirement GPU time cannot become zero or inherit the dispatch control's pass.");
                    Assert.That(row.Value<long>("gpu_ns"), Is.GreaterThan(0));
                    Assert.That(row.Value<int>("gpu_blocks"), Is.GreaterThan(0));
                }
                report["success"] = true;
            }
            finally { File.WriteAllText(output, report.ToString()); }
        }

        static IEnumerator ObserveTransfers(JObject report)
        { yield break; }
    }
}
