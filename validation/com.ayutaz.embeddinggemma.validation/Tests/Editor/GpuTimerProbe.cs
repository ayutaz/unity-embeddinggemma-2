using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEditorInternal;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.Rendering;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Calibration only; a tiny GPU operation never qualifies the real model.</summary>
    public static class GpuTimerProbe
    {
        public static IEnumerator Run(JObject report)
        {
            report["available"] = false;
            report["unity"] = Application.unityVersion;
            report["gpu"] = SystemInfo.graphicsDeviceName;
            report["graphics_api"] = SystemInfo.graphicsDeviceType.ToString();
            report["supports_gpu_recorder"] = SystemInfo.supportsGpuRecorder;
            report["scope"] = "isolated tiny GPU marker calibration; excludes upload, readback, wait and real model qualification";
            if (!SystemInfo.supportsGpuRecorder || !SystemInfo.supportsComputeShaders) yield break;
            var previousEnabled = Profiler.enabled;
            var previousGpu = Profiler.GetAreaEnabled(ProfilerArea.GPU);
            var previousEditor = ProfilerDriver.profileEditor;
            var previousDriver = ProfilerDriver.enabled;
            var previousConnection = ProfilerDriver.connectedProfiler;
            var cameraHost = new GameObject("Tiny GPU timer calibration camera");
            var target = new RenderTexture(16, 16, 0);
            var camera = cameraHost.AddComponent<Camera>(); camera.targetTexture = target;
            camera.clearFlags = CameraClearFlags.SolidColor; camera.cullingMask = 0;
            Recorder recorder = null;
            Recorder opaqueRecorder = null;
            try
            {
                if (report.Value<bool>("history_requested"))
                {
                    // Recorder collection alone does not populate Editor history.
                    // Select only this Editor's built-in local target, never a Player.
                    int? localEditor = null; var candidates = new List<int>();
                    var isLocal = typeof(ProfilerDriver).GetMethod("IsIdentifierOnLocalhost", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic);
                    if (isLocal == null) throw new System.NotSupportedException("Pinned local profiler target audit is unavailable.");
                    var availableTargets = ProfilerDriver.GetAvailableProfilers();
                    foreach (var targetId in availableTargets)
                        if (ProfilerDriver.GetConnectionIdentifier(targetId).IndexOf("Editor", System.StringComparison.OrdinalIgnoreCase) >= 0
                            && (bool)isLocal.Invoke(null, new object[] {targetId})) candidates.Add(targetId);
                    if (candidates.Contains(previousConnection)) localEditor = previousConnection;
                    else if (candidates.Count == 1) localEditor = candidates[0];
                    if (!localEditor.HasValue) throw new System.NotSupportedException("Local Editor profiler target unavailable/ambiguous: " + availableTargets.Length + " targets, " + candidates.Count + " local Editor candidates.");
                    ProfilerDriver.connectedProfiler = localEditor.Value;
                    ProfilerDriver.enabled = true;
                    report["history_local_editor_selected"] = true;
                    report["history_driver_enabled"] = ProfilerDriver.enabled;
                }
                // EditMode test coroutines submit through EditorLoop even after
                // entering PlayMode; include that CPU tree when collecting history.
                ProfilerDriver.profileEditor = report.Value<bool>("history_requested");
                Profiler.enabled = true; Profiler.SetAreaEnabled(ProfilerArea.GPU, true);
                for (var warm = 0; warm < 5; warm++) yield return null;
                var sampler = CustomSampler.Create("EmbeddingGemma.TimerCalibration", true);
                recorder = sampler.GetRecorder(); recorder.enabled = true;
                var opaque = report.Value<bool>("opaque_timing_requested") ? CustomSampler.Create("EmbeddingGemma.TimerOpaqueCall", true) : null;
                if (opaque != null) { opaqueRecorder = opaque.GetRecorder(); opaqueRecorder.enabled = true; }
                report["recorder_valid"] = recorder.isValid;
                report["batch_mode"] = Application.isBatchMode;
                report["play_mode"] = Application.isPlaying;
                report["profiler_enabled"] = Profiler.enabled;
                report["gpu_area_enabled"] = Profiler.GetAreaEnabled(ProfilerArea.GPU);
                var handles = new List<Unity.Profiling.LowLevel.Unsafe.ProfilerRecorderHandle>();
                Unity.Profiling.LowLevel.Unsafe.ProfilerRecorderHandle.GetAvailable(handles);
                foreach (var handle in handles)
                {
                    var description = Unity.Profiling.LowLevel.Unsafe.ProfilerRecorderHandle.GetDescription(handle);
                    if (description.Name == sampler.name) report["marker_flags"] = description.Flags.ToString();
                }
                var graph = new FunctionalGraph(); var x = graph.AddInput<float>(new TensorShape(1, 2), "input");
                var model = graph.Compile(x + 1f);
                using var worker = new Worker(model, BackendType.GPUCompute);
                using var input = new Tensor<float>(new TensorShape(1, 2), new[] {1f, 2f}); ComputeTensorData.Pin(input);
                using var commands = new CommandBuffer { name = "EmbeddingGemma GPU timer calibration" };
                commands.BeginSample(sampler); commands.ScheduleWorker(worker, input); commands.EndSample(sampler);
                var fence = commands.CreateGraphicsFence(GraphicsFenceType.AsyncQueueSynchronisation, SynchronisationStageFlags.AllGPUOperations);
                opaque?.Begin();
                var watch = Stopwatch.StartNew();
                try { Graphics.ExecuteCommandBuffer(commands); }
                finally { watch.Stop(); opaque?.End(); }
                report["submission_wall_ns"] = (long)(watch.ElapsedTicks * (1000000000.0 / Stopwatch.Frequency));
                report["actual_backend"] = ((Tensor<float>)worker.PeekOutput()).backendType.ToString();
                var frames = new JArray(); report["observations"] = frames;
                // Recorder GPU values arrive three frames late. Drive frames and drain
                // a bounded window; absent data stays unavailable, never CPU-timed.
                for (var attempt = 0; attempt < 60; attempt++)
                {
                    EditorApplication.QueuePlayerLoopUpdate(); yield return null;
                    var ns = recorder.gpuElapsedNanoseconds; var blocks = recorder.gpuSampleBlockCount;
                    frames.Add(new JObject { ["attempt"] = attempt, ["frame"] = Time.frameCount,
                        ["gpu_ns"] = ns, ["gpu_blocks"] = blocks, ["fence_passed"] = fence.passed,
                        ["cpu_marker_ns"] = recorder.elapsedNanoseconds, ["cpu_marker_blocks"] = recorder.sampleBlockCount,
                        ["opaque_gpu_ns"] = opaqueRecorder?.gpuElapsedNanoseconds ?? 0,
                        ["opaque_gpu_blocks"] = opaqueRecorder?.gpuSampleBlockCount ?? 0,
                        ["opaque_cpu_ns"] = opaqueRecorder?.elapsedNanoseconds ?? 0 });
                    if (ns > 0 && blocks > 0)
                    {
                        report["gpu_elapsed_ns"] = ns; report["gpu_sample_blocks"] = blocks;
                        report["available"] = true;
                    }
                    if (opaqueRecorder != null && opaqueRecorder.gpuElapsedNanoseconds > 0 && opaqueRecorder.gpuSampleBlockCount > 0)
                        report["opaque_gpu_elapsed_ns"] = opaqueRecorder.gpuElapsedNanoseconds;
                    if (report.Value<bool>("available") && (opaqueRecorder == null || report.Value<long>("opaque_gpu_elapsed_ns") > 0)) break;
                }
                while (!fence.passed) { EditorApplication.QueuePlayerLoopUpdate(); yield return null; }
                report["output"] = JArray.FromObject(((Tensor<float>)worker.PeekOutput()).DownloadToArray());
                if (report.Value<bool>("history_requested"))
                {
                    // Editor history transport is independent of Recorder delivery.
                    // Drain a bounded tail before stopping collection.
                    for (var drain = 0; drain < 12; drain++) yield return null;
                    var diagnostics = new JObject(); report["history_diagnostics"] = diagnostics;
                    report["gpu_history"] = GpuProfilerHistory.ReadExactMarker(sampler.name, diagnostics);
                }
            }
            finally
            {
                if (recorder != null) recorder.enabled = false;
                if (opaqueRecorder != null) opaqueRecorder.enabled = false;
                if (report.Value<bool>("history_requested"))
                {
                    ProfilerDriver.enabled = false; ProfilerDriver.connectedProfiler = previousConnection;
                    ProfilerDriver.enabled = previousDriver;
                }
                Profiler.SetAreaEnabled(ProfilerArea.GPU, previousGpu); Profiler.enabled = previousEnabled;
                ProfilerDriver.profileEditor = previousEditor;
                camera.targetTexture = null; UnityEngine.Object.DestroyImmediate(cameraHost);
                target.Release(); UnityEngine.Object.DestroyImmediate(target);
            }
        }
    }
}
