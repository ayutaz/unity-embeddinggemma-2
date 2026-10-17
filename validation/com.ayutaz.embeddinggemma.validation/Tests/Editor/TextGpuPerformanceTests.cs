using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Threading;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEditorInternal;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.Rendering;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Existing synchronous text API, fixed same-input rounds, actual GPU timestamps; no throughput claim.</summary>
    public sealed class TextGpuPerformanceTests
    {
        [UnityTearDown] public IEnumerator Exit() { if (Application.isPlaying) yield return new ExitPlayMode(); }
        static string Hash(string path)
        { using var stream=File.OpenRead(path); using var hash=SHA256.Create(); return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-","").ToLowerInvariant(); }
        static TimeSpan Cpu() { using var process=Process.GetCurrentProcess(); return process.TotalProcessorTime; }
        [UnityTest, Timeout(900000)] public IEnumerator FixedTextFirstWarmRepeatedCallsHaveActualGpuTimingsAndPinnedVectors()
        {
            // EnterPlayMode reloads the domain; initialize captured locals afterwards.
            yield return new EnterPlayMode();
            var output=Environment.GetEnvironmentVariable("EG_TEXT_PERFORMANCE_REPORT");
            Assert.That(output,Is.Not.Null.And.Not.Empty); Assert.That(File.Exists(output),Is.False);
            Assert.That(SystemInfo.supportsComputeShaders && SystemInfo.supportsGpuRecorder,Is.True,"Actual GPU timing required; no skip.");
            var directory=Environment.GetEnvironmentVariable("EG_MEDIA_BUNDLE_OUTPUT");
            GpuMediaExecutionPolicy.VerifyCurrent(directory);
            var referencePath=Path.GetFullPath(Path.Combine(Application.dataPath,"../../player-bundle/windows-unity21-20261012/search-reference.json"));
            Assert.That(Hash(referencePath),Is.EqualTo("f316d5199c920431c7f721d43b2cedbe4a6384fb0e0664511b75bc2e98fe80fe"));
            var query=JObject.Parse(File.ReadAllText(referencePath))["queries"].First();
            var expected=query["embedding"].Values<float>().ToArray();
            var directCommandsRequested=Environment.GetEnvironmentVariable("EG_TEXT_GPU_COMMAND_BUFFER")=="1";
            var allocationBeforeProbe=GC.GetAllocatedBytesForCurrentThread();
            var knownAllocation=new byte[65536]; knownAllocation[knownAllocation.Length-1]=1;
            var allocationProbeDelta=GC.GetAllocatedBytesForCurrentThread()-allocationBeforeProbe; GC.KeepAlive(knownAllocation);
            var allocationCounterAvailable=allocationProbeDelta>=65536;
            var report=new JObject { ["success"]=false,["runtime_source"]=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE"),
                ["precision"]=GpuMediaExecutionPolicy.CurrentPrecision,["descriptor_sha256"]=Hash(Path.Combine(directory,"bundle.json")),
                ["query_id"]=query["id"],["query_text"]=query["text"],["reference_sha256"]=Hash(referencePath),
                ["unity"]=Application.unityVersion,["gpu"]=SystemInfo.graphicsDeviceName,["graphics_api"]=SystemInfo.graphicsDeviceType.ToString(),
                ["host_memory_limit_bytes"]=GpuMediaExecutionPolicy.CurrentMemoryLimitBytes,["rounds"]=new JArray(),
                ["cold_definition"]="New process/session and first inference after load. OS file/shader caches are not purged.",
                ["timing_scope"]="Profiled synchronous API calls. GPU inclusive timestamp intervals can include queue/idle gaps. GPU parent and child timings overlap; never add them.",
                ["allocation_scope"]="GC cumulative bytes allocated on this main thread during each interval, including measurement wrapper allocations. Async load/release intervals include unrelated Editor callbacks; excludes other threads and native allocations.",
                ["throughput_measured"]=false,["microphone_started"]=false };
            report["dispatch_mode"]=directCommandsRequested?"direct_worker_command_buffer":"sdk_query_api";
            report["managed_allocation_counter_available"]=allocationCounterAvailable;
            report["allocation_probe_expected_minimum_bytes"]=65536; report["allocation_probe_raw_delta_bytes"]=allocationProbeDelta;
            report["gpu_timing_interpretation_validated"]=false;
            if(directCommandsRequested) report["timing_scope"]="Explicit GPU command-buffer marker around the same text Worker schedule; pre-pinned inputs exclude tokenizer/upload. Wall/CPU span command construction, submission, fence frame-waits and final readback; this is not the public query API latency.";
            void Save()=>File.WriteAllText(output,report.ToString()); Save();
            var previousEnabled=Profiler.enabled; var previousGpu=Profiler.GetAreaEnabled(ProfilerArea.GPU);
            var previousDriver=ProfilerDriver.enabled; var previousEditor=ProfilerDriver.profileEditor; var previousTarget=ProfilerDriver.connectedProfiler;
            var previousBackground=Application.runInBackground; var previousFrameRate=Application.targetFrameRate; var previousVsync=QualitySettings.vSyncCount;
            var recorders=new List<Recorder>(); var host=new GameObject("Owned text benchmark camera");
            var target=new RenderTexture(16,16,0); var camera=host.AddComponent<Camera>(); camera.targetTexture=target;
            camera.clearFlags=CameraClearFlags.SolidColor; camera.cullingMask=0;
            GpuMediaSession session=null; using var cancellation=new CancellationTokenSource();
            CommandBuffer directCommands=null; Tensor<int> directIds=null,directMask=null; Worker directWorker=null;
            try
            {
                Application.runInBackground=true; Application.targetFrameRate=60; QualitySettings.vSyncCount=0;
                report["run_in_background"]=Application.runInBackground; report["target_frame_rate"]=Application.targetFrameRate; report["vsync_count"]=QualitySettings.vSyncCount;
                var isLocal=typeof(ProfilerDriver).GetMethod("IsIdentifierOnLocalhost",BindingFlags.Static|BindingFlags.NonPublic);
                Assert.That(isLocal,Is.Not.Null);
                var candidates=ProfilerDriver.GetAvailableProfilers().Where(id=>ProfilerDriver.GetConnectionIdentifier(id).IndexOf("Editor",StringComparison.OrdinalIgnoreCase)>=0 && (bool)isLocal.Invoke(null,new object[] {id})).ToArray();
                Assert.That(candidates.Contains(previousTarget) || candidates.Length==1,Is.True,"Local Editor profiler target must be unambiguous.");
                ProfilerDriver.connectedProfiler=candidates.Contains(previousTarget)?previousTarget:candidates[0];
                // Loading is timed separately, without retaining its unrelated Editor history.
                ProfilerDriver.enabled=false; Profiler.enabled=false;
                report["phase"]="load"; report["load_begin_utc"]=DateTime.UtcNow.ToString("O"); Save();
                var loadAllocated=GC.GetAllocatedBytesForCurrentThread(); var loadCpu=Cpu(); var load=Stopwatch.StartNew();
                var loading=GpuMediaBundle.Open(directory).LoadAsync("text",cancellation.Token);
                while(!loading.IsCompleted)
                { if(MediaExecutionMemory.Peak()>GpuMediaExecutionPolicy.CurrentMemoryLimitBytes)cancellation.Cancel(); EditorApplication.QueuePlayerLoopUpdate(); yield return null; }
                session=loading.GetAwaiter().GetResult(); load.Stop();
                Assert.That(session.Backend,Is.EqualTo(Unity.InferenceEngine.BackendType.GPUCompute));
                report["actual_backend"]=session.Backend.ToString();
                report["load_wall_seconds"]=load.Elapsed.TotalSeconds; report["load_process_cpu_seconds"]=(Cpu()-loadCpu).TotalSeconds;
                report["load_main_thread_managed_allocated_bytes"]=allocationCounterAvailable?new JValue(GC.GetAllocatedBytesForCurrentThread()-loadAllocated):JValue.CreateNull();
                report["load_end_utc"]=DateTime.UtcNow.ToString("O");
                report["load_gpu_elapsed_ns"]=null; report["load_host_lifetime_peak_bytes"]=MediaExecutionMemory.Peak(); Save();
                if(directCommandsRequested)
                {
                    const BindingFlags flags=BindingFlags.Instance|BindingFlags.NonPublic;
                    var text=(TextEmbedder)typeof(GpuMediaSession).GetField("query",flags).GetValue(session);
                    directWorker=(Worker)typeof(TextEmbedder).GetField("worker",flags).GetValue(text);
                    var tokenizer=(TextTokenizer)typeof(TextEmbedder).GetField("tokenizer",flags).GetValue(text);
                    var encoded=tokenizer.Encode(TextPrompts.Format((string)query["text"],TextRole.Query));
                    var ids=encoded.GetIds().ToArray(); var mask=encoded.GetAttentionMask().ToArray();
                    Assert.That(ids.Length,Is.EqualTo(128)); Assert.That(mask.Length,Is.EqualTo(128));
                    directIds=new Tensor<int>(new TensorShape(1,128),ids); directMask=new Tensor<int>(new TensorShape(1,128),mask);
                    ComputeTensorData.Pin(directIds); ComputeTensorData.Pin(directMask);
                    directWorker.SetInput("input_ids",directIds); directWorker.SetInput("attention_mask",directMask);
                    directCommands=new CommandBuffer {name="Owned text model timing"};
                    report["input_ids"]=JArray.FromObject(ids); report["attention_mask"]=JArray.FromObject(mask);
                }
                ProfilerDriver.profileEditor=true; ProfilerDriver.enabled=true; Profiler.enabled=true; Profiler.SetAreaEnabled(ProfilerArea.GPU,true);
                report["gpu_history_scope"]="Owned Editor history is enabled after load and cleared before each measured call; earlier rounds are saved as JSON before clearing.";
                for(var frame=0;frame<5;frame++) yield return null;
                foreach(var name in new[] {"first","warm","repeat-1","repeat-2"})
                {
                    ProfilerDriver.enabled=true; Profiler.enabled=true;
                    ProfilerDriver.ClearAllFrames();
                    report["phase"]=name; Save(); var beginUtc=DateTime.UtcNow;
                    var sampler=CustomSampler.Create("EmbeddingGemma.TextBenchmark."+name,true);
                    var recorder=sampler.GetRecorder(); recorder.enabled=true; recorders.Add(recorder);
                    Assert.That(recorder.isValid,Is.True);
                    var allocatedBefore=GC.GetAllocatedBytesForCurrentThread(); var cpu=Cpu(); var watch=Stopwatch.StartNew(); float[] actual;
                    if(directCommandsRequested)
                    {
                        directCommands.Clear(); directCommands.BeginSample(sampler);
                        directCommands.ScheduleWorker(directWorker); directCommands.EndSample(sampler);
                        var fence=directCommands.CreateGraphicsFence(GraphicsFenceType.AsyncQueueSynchronisation,SynchronisationStageFlags.AllGPUOperations);
                        Graphics.ExecuteCommandBuffer(directCommands);
                        while(!fence.passed) { EditorApplication.QueuePlayerLoopUpdate(); yield return null; }
                        var result=(Tensor<float>)directWorker.PeekOutput();
                        Assert.That(result.backendType,Is.EqualTo(BackendType.GPUCompute));
                        Assert.That(result.shape,Is.EqualTo(new TensorShape(1,768))); actual=result.DownloadToArray(); watch.Stop();
                    }
                    else
                    {
                        sampler.Begin();
                        try { actual=session.EmbedQuery((string)query["text"]); }
                        finally { sampler.End(); watch.Stop(); }
                    }
                    var endUtc=DateTime.UtcNow;
                    var allocatedBytes=GC.GetAllocatedBytesForCurrentThread()-allocatedBefore;
                    var cpuSeconds=(Cpu()-cpu).TotalSeconds;
                    var cosine=TextSearchIndex.Cosine(actual,expected); var norm=actual.Sum(value=>(double)value*value);
                    Assert.That(actual.Length,Is.EqualTo(768)); Assert.That(cosine,Is.GreaterThanOrEqualTo(GpuMediaExecutionPolicy.CurrentMinimumCosine));
                    Assert.That(actual.All(value=>!float.IsNaN(value)&&!float.IsInfinity(value)),Is.True);
                    Assert.That(Math.Abs(norm-1),Is.LessThanOrEqualTo(.002));
                    var capture=Path.ChangeExtension(output,null)+"-"+name+".bin"; Assert.That(File.Exists(capture),Is.False);
                    var bytes=new byte[actual.Length*4]; Buffer.BlockCopy(actual,0,bytes,0,bytes.Length); File.WriteAllBytes(capture,bytes);
                    var row=new JObject { ["round"]=name,["wall_seconds"]=watch.Elapsed.TotalSeconds,["process_cpu_seconds"]=cpuSeconds,
                        ["begin_utc"]=beginUtc.ToString("O"),["end_utc"]=endUtc.ToString("O"),
                        ["main_thread_managed_allocated_bytes"]=allocationCounterAvailable?new JValue(allocatedBytes):JValue.CreateNull(),
                        ["main_thread_managed_allocated_bytes_per_second"]=allocationCounterAvailable?new JValue(allocatedBytes/watch.Elapsed.TotalSeconds):JValue.CreateNull(),
                        ["host_lifetime_peak_bytes"]=MediaExecutionMemory.Peak(),["unity_allocated_bytes"]=Profiler.GetTotalAllocatedMemoryLong(),
                        ["managed_heap_bytes"]=GC.GetTotalMemory(false),["cosine"]=cosine,["norm_squared"]=norm,
                        ["capture"]=Path.GetFileName(capture),["capture_sha256"]=Hash(capture),["observations"]=new JArray() };
                    ((JArray)report["rounds"]).Add(row); Save();
                    var positiveFrame=-1; var lastObservedFrame=-1; var draining=Stopwatch.StartNew();
                    for(var attempt=0;attempt<6000 && draining.Elapsed.TotalSeconds<15;attempt++)
                    {
                        EditorApplication.QueuePlayerLoopUpdate(); yield return null;
                        if(lastObservedFrame!=Time.frameCount)
                        {
                            lastObservedFrame=Time.frameCount;
                            ((JArray)row["observations"]).Add(new JObject { ["frame"]=Time.frameCount,["history_last_frame"]=ProfilerDriver.lastFrameIndex,["gpu_ns"]=recorder.gpuElapsedNanoseconds,["gpu_blocks"]=recorder.gpuSampleBlockCount });
                            if(positiveFrame<0 && recorder.gpuElapsedNanoseconds>0 && recorder.gpuSampleBlockCount>0)
                            { positiveFrame=Time.frameCount;row["gpu_elapsed_ns"]=recorder.gpuElapsedNanoseconds;row["gpu_sample_blocks"]=recorder.gpuSampleBlockCount; }
                        }
                        if(positiveFrame>=0 && Time.frameCount-positiveFrame>=6) break;
                    }
                    row["render_frames_after_first_gpu_value"]=positiveFrame<0?0:Time.frameCount-positiveFrame;
                    Assert.That(row.Value<int>("render_frames_after_first_gpu_value"),Is.GreaterThanOrEqualTo(6),"Actual render frames must advance; repeated Editor coroutine steps are not GPU frame completion.");
                    // Read a stopped snapshot, not a tree that can profile its own traversal.
                    ProfilerDriver.enabled=false; Profiler.enabled=false;
                    var diagnostics=new JObject(); row["gpu_history_diagnostics"]=diagnostics;
                    var history=GpuProfilerHistory.ReadExactMarker(sampler.name,diagnostics); row["gpu_history"]=history; Save();
                    Assert.That(row.Value<long>("gpu_elapsed_ns"),Is.GreaterThan(0),"Absent GPU timing is a failure, never zero-time success.");
                    Assert.That(history.Count,Is.GreaterThan(0),"The local Editor GPU history must independently contain this exact marker.");
                    Assert.That(MediaExecutionMemory.Peak(),Is.LessThanOrEqualTo(GpuMediaExecutionPolicy.CurrentMemoryLimitBytes));
                    recorder.enabled=false;
                }
                report["phase"]="release"; report["release_begin_utc"]=DateTime.UtcNow.ToString("O"); Save();
                var closeAllocated=GC.GetAllocatedBytesForCurrentThread(); var closeCpu=Cpu(); var close=Stopwatch.StartNew(); var closing=session.CloseAsync(); while(!closing.IsCompleted) yield return null;
                closing.GetAwaiter().GetResult(); close.Stop(); MultimodalSearchSampleGpuTests.AssertReleased(session);
                report["release_wall_seconds"]=close.Elapsed.TotalSeconds; report["release_process_cpu_seconds"]=(Cpu()-closeCpu).TotalSeconds;
                report["release_main_thread_managed_allocated_bytes"]=allocationCounterAvailable?new JValue(GC.GetAllocatedBytesForCurrentThread()-closeAllocated):JValue.CreateNull();
                report["release_end_utc"]=DateTime.UtcNow.ToString("O"); report["phase"]="closed";
                report["release_host_lifetime_peak_bytes"]=MediaExecutionMemory.Peak(); report["session_closed"]=true; report["success"]=true;
            }
            finally
            {
                session?.Dispose(); directCommands?.Dispose(); directIds?.Dispose(); directMask?.Dispose(); foreach(var recorder in recorders)recorder.enabled=false;
                ProfilerDriver.enabled=false; ProfilerDriver.connectedProfiler=previousTarget; ProfilerDriver.enabled=previousDriver;
                Profiler.SetAreaEnabled(ProfilerArea.GPU,previousGpu); Profiler.enabled=previousEnabled; ProfilerDriver.profileEditor=previousEditor;
                Application.runInBackground=previousBackground; Application.targetFrameRate=previousFrameRate; QualitySettings.vSyncCount=previousVsync;
                camera.targetTexture=null; UnityEngine.Object.DestroyImmediate(host); target.Release(); UnityEngine.Object.DestroyImmediate(target); Save();
            }
            yield return new ExitPlayMode();
        }
    }
}
