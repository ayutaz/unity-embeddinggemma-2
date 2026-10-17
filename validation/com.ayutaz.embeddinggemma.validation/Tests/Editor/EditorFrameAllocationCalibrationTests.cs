using System;
using System.Collections;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditorInternal;
using UnityEngine;
using UnityEngine.Profiling;
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

        static IEnumerator Observe(JObject report)
        {
            var cases = new JArray(); report["cases"] = cases;
            report["background_and_observer_ceiling_bytes"] = 1024 * 1024;
            var enabled = Profiler.enabled; var driver = ProfilerDriver.enabled;
            var editor = ProfilerDriver.profileEditor; var memory = Profiler.GetAreaEnabled(ProfilerArea.Memory);
            try
            {
                // ProfilerRecorder owns this metric capture. Full Editor profiling
                // is unnecessary and can interfere with SearchImporter's scopes.
                ProfilerDriver.enabled = false; Profiler.enabled = false;
                Profiler.SetAreaEnabled(ProfilerArea.Memory, true);
                report["full_profiler_recording_during_capture"] = Profiler.enabled;
                for (var i = 0; i < 4; i++) { EditorApplication.QueuePlayerLoopUpdate(); yield return null; }
                foreach (var bytes in new[] {0, 8 * 1024 * 1024})
                {
                    byte[] first = null, second = null;
                    using var capture = new FrameManagedAllocationCapture("EditorAllocation.Control." + bytes);
                    var before = capture.CollectedSamples;
                    if (bytes > 0) { first = new byte[bytes / 2]; first[0] = 1; }
                    var wait = Stopwatch.StartNew();
                    while (capture.CollectedSamples <= before && wait.Elapsed.TotalSeconds < 15)
                    { EditorApplication.QueuePlayerLoopUpdate(); yield return null; }
                    Assert.That(capture.CollectedSamples, Is.GreaterThan(before), "The Editor GC counter must finalize the first payload before allocating the second.");
                    if (bytes > 0) { second = new byte[bytes / 2]; second[0] = 1; }
                    capture.Complete(); wait.Restart(); var closed = false;
                    while (!(closed = capture.TryClose()) && wait.Elapsed.TotalSeconds < 15)
                    { EditorApplication.QueuePlayerLoopUpdate(); yield return null; }
                    Assert.That(closed, Is.True, "A post-completion Editor counter sample is required.");
                    var counter = capture.Snapshot();
                    cases.Add(new JObject { ["payload_bytes"] = bytes, ["counter"] = counter });
                    GC.KeepAlive(first); GC.KeepAlive(second);
                    var total = counter.Value<long>("managed_bytes_in_recorded_frames");
                    Assert.That(total, Is.InRange((long)bytes, (long)bytes + 1024 * 1024),
                        "Predeclared 1MiB background/observer ceiling; failure does not relax it.");
                    if (bytes > 0)
                        Assert.That(counter["samples"].Values<long>().Count(value => value >= bytes / 2), Is.GreaterThanOrEqualTo(2),
                            "Both distinct Editor counter samples must contain their payload.");
                }
                var difference = cases[1]["counter"].Value<long>("managed_bytes_in_recorded_frames")
                    - cases[0]["counter"].Value<long>("managed_bytes_in_recorded_frames");
                report["payload_delta_observed_bytes"] = difference;
                Assert.That(difference, Is.InRange(7L * 1024 * 1024, 9L * 1024 * 1024));
            }
            finally
            {
                ProfilerDriver.enabled = false; Profiler.enabled = false;
                ProfilerDriver.profileEditor = editor; Profiler.SetAreaEnabled(ProfilerArea.Memory, memory);
                Profiler.enabled = enabled; ProfilerDriver.enabled = driver;
            }
        }
    }
}
