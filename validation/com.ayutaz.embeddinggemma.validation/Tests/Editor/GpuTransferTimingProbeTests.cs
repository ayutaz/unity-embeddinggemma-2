using System;
using System.Collections;
using System.IO;
using System.Diagnostics;
using System.Threading;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditorInternal;
using UnityEngine;
using UnityEngine.Profiling;
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
            yield return new EnterPlayMode();
            var output = Environment.GetEnvironmentVariable("EG_GPU_TRANSFER_TIMING_REPORT");
            Assert.That(output, Is.Not.Null.And.Not.Empty);
            Assert.That(File.Exists(output), Is.False);
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
        {
            const int words = 2 * 1024 * 1024;
            var values = new float[words];
            for (var i = 0; i < values.Length; i++) values[i] = (i % 257) * .25f;
            var readback = new float[words];
            var rows = new JArray(); report["transfer_observations"] = rows;
            report["transfer_scope_interpretation_validated"] = false;
            report["retirement_completion_measured"] = false;
            var enabled = Profiler.enabled; var gpu = Profiler.GetAreaEnabled(ProfilerArea.GPU);
            var driver = ProfilerDriver.enabled; var editor = ProfilerDriver.profileEditor;
            var background = Application.runInBackground; var fps = Application.targetFrameRate;
            var vsync = QualitySettings.vSyncCount;
            var host = new GameObject("Owned GPU transfer timing camera");
            var target = new RenderTexture(16, 16, 0); var camera = host.AddComponent<Camera>();
            camera.targetTexture = target; camera.cullingMask = 0; camera.clearFlags = CameraClearFlags.SolidColor;
            try
            {
                Application.runInBackground = true; Application.targetFrameRate = 60; QualitySettings.vSyncCount = 0;
                ProfilerDriver.profileEditor = true; ProfilerDriver.enabled = true;
                Profiler.enabled = true; Profiler.SetAreaEnabled(ProfilerArea.GPU, true);
                for (var i = 0; i < 5; i++) { EditorApplication.QueuePlayerLoopUpdate(); yield return null; }
                foreach (var kind in new[] {"one-upload", "two-uploads", "one-upload-after-cpu-gap", "retire-two-buffers"})
                {
                    var retire = kind == "retire-two-buffers";
                    var count = kind == "two-uploads" || retire ? 2 : 1;
                    var buffers = new ComputeBuffer[count];
                    var row = new JObject { ["kind"] = kind, ["gpu_available"] = false,
                        ["gpu_ns"] = null, ["gpu_blocks"] = null, ["data_integrity_verified_outside_marker"] = false,
                        ["upload_bytes_inside_marker"] = retire ? 0 : (long)count * words * sizeof(float),
                        ["retired_buffer_capacity_bytes"] = retire ? (long)count * words * sizeof(float) : 0,
                        ["cpu_gap_before_marker_ms"] = kind == "one-upload-after-cpu-gap" ? 250 : 0,
                        ["observations"] = new JArray(),
                        ["scope"] = retire ? "Synchronous ComputeBuffer.Dispose call marker; driver retirement completion is not measured."
                            : "Synchronous ComputeBuffer.SetData call marker; excludes buffer creation, CPU gap and verification readback. GPU timestamp attribution to copy engines is unqualified." };
                    rows.Add(row);
                    var sampler = CustomSampler.Create("EmbeddingGemma.TransferTiming." + Guid.NewGuid().ToString("N"), true);
                    var recorder = sampler.GetRecorder(); recorder.enabled = true;
                    try
                    {
                        Assert.That(recorder.isValid, Is.True);
                        for (var i = 0; i < count; i++) buffers[i] = new ComputeBuffer(words, sizeof(float));
                        if (retire)
                        {
                            foreach (var buffer in buffers)
                            { buffer.SetData(values); buffer.GetData(readback); CollectionAssert.AreEqual(values, readback); }
                            row["data_integrity_verified_outside_marker"] = true;
                        }
                        if (kind == "one-upload-after-cpu-gap") Thread.Sleep(250);
                        row["begin_utc"] = DateTime.UtcNow.ToString("O");
                        var watch = Stopwatch.StartNew(); sampler.Begin();
                        try
                        {
                            for (var i = 0; i < count; i++)
                                if (retire) { buffers[i].Dispose(); buffers[i] = null; }
                                else buffers[i].SetData(values);
                        }
                        finally { sampler.End(); watch.Stop(); }
                        row["end_utc"] = DateTime.UtcNow.ToString("O");
                        row["cpu_call_wall_seconds"] = watch.Elapsed.TotalSeconds;
                        var lastFrame = -1; var firstPositiveFrame = -1; var drain = Stopwatch.StartNew();
                        while (drain.Elapsed.TotalSeconds < 12)
                        {
                            EditorApplication.QueuePlayerLoopUpdate(); yield return null;
                            if (lastFrame == Time.frameCount) continue;
                            lastFrame = Time.frameCount;
                            ((JArray)row["observations"]).Add(new JObject { ["render_frame"] = lastFrame,
                                ["gpu_ns"] = recorder.gpuElapsedNanoseconds, ["gpu_blocks"] = recorder.gpuSampleBlockCount,
                                ["cpu_marker_ns"] = recorder.elapsedNanoseconds });
                            if (recorder.gpuElapsedNanoseconds > 0 && recorder.gpuSampleBlockCount > 0 && firstPositiveFrame < 0)
                            {
                                firstPositiveFrame = lastFrame; row["gpu_available"] = true;
                                row["gpu_ns"] = recorder.gpuElapsedNanoseconds; row["gpu_blocks"] = recorder.gpuSampleBlockCount;
                            }
                            if (firstPositiveFrame >= 0 && lastFrame - firstPositiveFrame >= 6) break;
                        }
                        row["distinct_render_frames_observed"] = ((JArray)row["observations"]).Count;
                        if (!retire)
                        {
                            foreach (var buffer in buffers)
                            { buffer.GetData(readback); CollectionAssert.AreEqual(values, readback); }
                            row["data_integrity_verified_outside_marker"] = true;
                        }
                    }
                    finally
                    {
                        recorder.enabled = false;
                        foreach (var buffer in buffers) buffer?.Dispose();
                    }
                }
            }
            finally
            {
                ProfilerDriver.enabled = false; ProfilerDriver.profileEditor = editor; ProfilerDriver.enabled = driver;
                Profiler.SetAreaEnabled(ProfilerArea.GPU, gpu); Profiler.enabled = enabled;
                Application.runInBackground = background; Application.targetFrameRate = fps; QualitySettings.vSyncCount = vsync;
                camera.targetTexture = null; UnityEngine.Object.DestroyImmediate(host);
                target.Release(); UnityEngine.Object.DestroyImmediate(target);
            }
        }
    }
}
