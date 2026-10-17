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
            var cameraHost = new GameObject("Tiny GPU timer calibration camera");
            var target = new RenderTexture(16, 16, 0);
            var camera = cameraHost.AddComponent<Camera>(); camera.targetTexture = target;
            camera.clearFlags = CameraClearFlags.SolidColor; camera.cullingMask = 0;
            Recorder recorder = null;
            try
            {
                ProfilerDriver.profileEditor = false;
                Profiler.enabled = true; Profiler.SetAreaEnabled(ProfilerArea.GPU, true);
                for (var warm = 0; warm < 5; warm++) yield return null;
                var sampler = CustomSampler.Create("EmbeddingGemma.TimerCalibration", true);
                recorder = sampler.GetRecorder(); recorder.enabled = true;
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
                var watch = Stopwatch.StartNew(); Graphics.ExecuteCommandBuffer(commands); watch.Stop();
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
                        ["cpu_marker_ns"] = recorder.elapsedNanoseconds, ["cpu_marker_blocks"] = recorder.sampleBlockCount });
                    if (ns > 0 && blocks > 0)
                    {
                        report["gpu_elapsed_ns"] = ns; report["gpu_sample_blocks"] = blocks;
                        report["available"] = true; break;
                    }
                }
                while (!fence.passed) { EditorApplication.QueuePlayerLoopUpdate(); yield return null; }
                report["output"] = JArray.FromObject(((Tensor<float>)worker.PeekOutput()).DownloadToArray());
            }
            finally
            {
                if (recorder != null) recorder.enabled = false;
                Profiler.SetAreaEnabled(ProfilerArea.GPU, previousGpu); Profiler.enabled = previousEnabled;
                ProfilerDriver.profileEditor = previousEditor;
                camera.targetTexture = null; UnityEngine.Object.DestroyImmediate(cameraHost);
                target.Release(); UnityEngine.Object.DestroyImmediate(target);
            }
        }
    }
}
