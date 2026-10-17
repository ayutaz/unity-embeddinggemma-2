using System;
using System.Collections;
using System.IO;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class EditorFrameAllocationCalibrationTests
    {
        [UnityTest, Timeout(120000)]
        public IEnumerator KnownPayloadAcrossEditorCounterSamplesHasFrameWideAllocationEvidence()
        {
            Assert.That(Application.isPlaying, Is.False);
            var output = Environment.GetEnvironmentVariable("EG_EDITOR_FRAME_ALLOCATION_CALIBRATION_REPORT");
            Assert.That(output, Is.Not.Null.And.Not.Empty); Assert.That(File.Exists(output), Is.False);
            var report = new JObject { ["success"] = false, ["model_loaded"] = false,
                ["gpu_execution_validated"] = false, ["unity"] = Application.unityVersion,
                ["validation_source"] = Environment.GetEnvironmentVariable("EG_VALIDATION_SOURCE"),
                ["scope"] = "Known Editor-loop allocations in containing counter frames; includes unrelated Editor work." };
            try
            {
                yield return Observe(report);
                Assert.That(report["cases"] is JArray rows && rows.Count == 2, Is.True,
                    "EditMode requires its own known-payload frame counter calibration.");
                report["success"] = true;
            }
            finally { File.WriteAllText(output, report.ToString()); }
        }

        static IEnumerator Observe(JObject report) { yield break; }
    }
}
