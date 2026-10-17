using System;
using System.Collections;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using UnityEngine.TestTools;
using UnityEditorInternal;
using UnityEngine.Profiling;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class MediaRepeatedInputPerformanceTests
    {
        static string Artifact(string path)=>Path.GetFullPath(Path.Combine(Application.dataPath,"../../",path));
        static string Hash(byte[] bytes){using var hash=SHA256.Create();return BitConverter.ToString(hash.ComputeHash(bytes)).Replace("-","").ToLowerInvariant();}
        static string Hash(string path)=>Hash(File.ReadAllBytes(path));
        static TimeSpan Cpu(){using var p=Process.GetCurrentProcess();return p.TotalProcessorTime;}
        [UnityTearDown] public IEnumerator Exit(){if(Application.isPlaying)yield return new ExitPlayMode();}

        [UnityTest,Timeout(1200000)] public IEnumerator FixedMediaInputFirstWarmRepeatedCallsKeepReferenceAndMeasureStages()
        {
            yield return new EnterPlayMode();
            var previous=Application.runInBackground;var fps=Application.targetFrameRate;var vsync=QualitySettings.vSyncCount;
            Application.runInBackground=true;Application.targetFrameRate=60;QualitySettings.vSyncCount=0;
            var host=new GameObject("Owned media measurement camera");var target=new RenderTexture(16,16,0);
            var camera=host.AddComponent<Camera>();camera.targetTexture=target;camera.clearFlags=CameraClearFlags.SolidColor;camera.cullingMask=0;
            using var cancellation=new CancellationTokenSource();
            var gpuRequested=Environment.GetEnvironmentVariable("EG_ENCODER_GPU_DISPATCH_TRACE")=="1";
            using var gpu=gpuRequested?new EncoderGpuDispatchTrace():null;
            var oldProfiler=Profiler.enabled;var oldDriver=ProfilerDriver.enabled;var oldEditor=ProfilerDriver.profileEditor;
            var oldGpuArea=Profiler.GetAreaEnabled(ProfilerArea.GPU);
            try
            {
                if(gpuRequested){ProfilerDriver.profileEditor=true;ProfilerDriver.enabled=true;Profiler.enabled=true;Profiler.SetAreaEnabled(ProfilerArea.GPU,true);}
                var task=Run(cancellation.Token,gpu);long peak=0;
                while(!task.IsCompleted)
                {
                    gpu?.Capture();
                    peak=Math.Max(peak,MediaExecutionMemory.Peak());
                    if(peak>GpuMediaExecutionPolicy.CurrentMemoryLimitBytes)cancellation.Cancel();
                    EditorApplication.QueuePlayerLoopUpdate();yield return null;
                }
                task.GetAwaiter().GetResult();
                if(gpu!=null)
                {
                    var until=Time.frameCount+6;var drain=Stopwatch.StartNew();
                    while(Time.frameCount<until && drain.Elapsed.TotalSeconds<15){gpu.Capture();EditorApplication.QueuePlayerLoopUpdate();yield return null;}
                    gpu.Capture();Assert.That(Time.frameCount,Is.GreaterThanOrEqualTo(until),"GPU observation requires actual rendered frames.");
                }
                Assert.That(peak,Is.GreaterThan(0).And.LessThanOrEqualTo(GpuMediaExecutionPolicy.CurrentMemoryLimitBytes));
                var output=Environment.GetEnvironmentVariable("EG_MEDIA_EXECUTION_REPORT");
                var report=JObject.Parse(File.ReadAllText(output));report["host_native_peak_bytes"]=peak;File.WriteAllText(output,report.ToString());
                if(gpu!=null)
                {
                    report["encoder_gpu_dispatch_intervals"]=gpu.Snapshot();
                    report["media_gpu_submission_intervals"]=gpu.Snapshot(true);
                    report["encoder_gpu_dispatch_scope"]="Optional instrumented call; model segments and projections exclude inter-stage fence waits and cache cleanup. Includes commanded copies/queue effects; not pure kernel time. Missing intervals stay null, never zero.";
                    File.WriteAllText(output,report.ToString());
                    Assert.That(gpu.Snapshot().Any(row=>row["gpu_ns"].Type!=JTokenType.Null),Is.True,"No GPU interval was observed; this is not a GPU timing pass.");
                    Assert.That(gpu.Snapshot(true).Count,Is.EqualTo(8),"Each of four calls needs preprocessor and backbone scopes.");
                    Assert.That(gpu.Snapshot(true).All(row=>row["gpu_ns"].Type!=JTokenType.Null && (long)row["gpu_ns"]>0 && !(bool)row["aborted"]),Is.True,"Unavailable GPU submission scopes cannot pass the new measurement gate.");
                }
            }
            finally
            {
                camera.targetTexture=null;UnityEngine.Object.DestroyImmediate(host);target.Release();UnityEngine.Object.DestroyImmediate(target);
                Application.runInBackground=previous;Application.targetFrameRate=fps;QualitySettings.vSyncCount=vsync;
                if(gpuRequested){Profiler.SetAreaEnabled(ProfilerArea.GPU,oldGpuArea);ProfilerDriver.profileEditor=oldEditor;ProfilerDriver.enabled=oldDriver;Profiler.enabled=oldProfiler;}
            }
            yield return new ExitPlayMode();
        }

        static async Task Run(CancellationToken token,EncoderGpuDispatchTrace gpu)
        {
            var directory=Environment.GetEnvironmentVariable("EG_MEDIA_BUNDLE_OUTPUT");
            var module=Environment.GetEnvironmentVariable("EG_MEDIA_BUNDLE_MODULE");
            var output=Environment.GetEnvironmentVariable("EG_MEDIA_EXECUTION_REPORT");
            Assert.That(output,Is.Not.Null.And.Not.Empty);Assert.That(File.Exists(output),Is.False);
            GpuMediaExecutionPolicy.VerifyCurrent(directory);
            var image=module.StartsWith("image-",StringComparison.Ordinal);
            Assert.That(image||module.StartsWith("audio-",StringComparison.Ordinal),Is.True);
            var bundle=GpuMediaBundle.Open(directory);var metadata=bundle.Module(module);
            var report=new JObject{["success"]=false,["module"]=module,["precision"]=GpuMediaExecutionPolicy.CurrentPrecision,
                ["execution_source"]=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE"),
                ["bundle_descriptor_sha256"]=Hash(Path.Combine(directory,"bundle.json")),
                ["host_memory_limit_bytes"]=GpuMediaExecutionPolicy.CurrentMemoryLimitBytes,
                ["unity"]=Application.unityVersion,["gpu"]=SystemInfo.graphicsDeviceName,["graphics_api"]=SystemInfo.graphicsDeviceType.ToString(),
                ["cases"]=new JArray(),["stage_measurements"]=new JArray(),
                ["scope"]="Same fixed Texture/PCM input through existing GPU SDK API; stage wall and whole-process CPU intervals only.",
                ["cold_definition"]="New process/session, first inference after loading; OS/shader caches and project Library are not purged.",
                ["instrumentation_scope"]="Case wall includes callbacks/report writes. Stage timers exclude report observer writes. CPU includes all Editor threads; GPU timestamps/VRAM/GC are unavailable here, never zero.",
                ["gpu_execution_time_measured"]=false,["throughput_measured"]=false};
            void Save()=>File.WriteAllText(output,report.ToString());
            Texture2D texture=null;AudioPcmInput pcm=null;GpuMediaSession session=null;
            try
            {
                report["phase"]="input_setup";Save();var setup=Stopwatch.StartNew();
                var referencePath=Artifact(image?"numerical-acceptance-state/image-role-reference-real/image-role-reference/document/image-embedding-reference.json":"k/artifacts/audio-final-backbone/audio-embedding-reference.json");
                Assert.That(Hash(referencePath),Is.EqualTo(image?"90692ad38c6d7d844dc6b6e79c7d5cfa719ea59866946dbf94a40bca10ac9eac":"281d12573b431667e8841033db1c6a10dce4382ef98c9a5345c18044a1194001"));
                var reference=JObject.Parse(File.ReadAllText(referencePath));string id;string inputHash;
                if(image)
                {
                    var inputRoot=Artifact("k/artifacts/vision-encoder");
                    var original=JObject.Parse(File.ReadAllText(Path.Combine(inputRoot,"vision-reference.json")));
                    var row=original["cases"].First(r=>"image-"+(string)r["profile"]==module);id=(string)row["id"];
                    var name=(string)row["image"];var path=Path.Combine(inputRoot,name);
                    var audit=JObject.Parse(File.ReadAllText(Path.Combine(inputRoot,"vision-validation.json")));
                    Assert.That(Hash(path),Is.EqualTo((string)audit["files"][name]["sha256"]));inputHash=Hash(path);
                    var pinned=module=="image-square"?"c2a1900a1656ce2d843e9d06801c62f927b98a51e9b95ae1159bb2c7515fa399":
                        module=="image-landscape"?"f2b3f32a22e6282b9876b98c4c5a7bbf75e7469dff933128e17d6503dd6d6e55":
                        "fce3474792c6401c4bae6ac0261a386f400bac0cece977ffda86388ee6682489";
                    Assert.That(id,Is.EqualTo(module.Substring("image-".Length)+"-0"));Assert.That(inputHash,Is.EqualTo(pinned));
                    texture=new Texture2D(2,2,TextureFormat.RGBA32,false,true);
                    Assert.That(ImageConversion.LoadImage(texture,File.ReadAllBytes(path),true),Is.True);
                    report["input_path"]=path;
                }
                else
                {
                    var valid=(int)metadata["valid_audio_frames"];
                    id=(string)reference["cases"].First(r=>(string)r["profile"]=="frames-1119-valid-"+valid)["id"];
                    var inputRoot=Artifact("k/artifacts/audio-real-chain/audio-runtime-preprocessing");
                    var auditPath=Path.Combine(inputRoot,"audio-validation.json");
                    Assert.That(Hash(auditPath),Is.EqualTo("4e325f708e8d103a9de358a411380ddbb51cfe953984e9dcb73e291d919630fa"));
                    var audit=JObject.Parse(File.ReadAllText(auditPath));var waveName=id+"-pcm-fp32.bin";var maskName=id+"-samples-int32.bin";
                    var waveBytes=File.ReadAllBytes(Path.Combine(inputRoot,waveName));var maskBytes=File.ReadAllBytes(Path.Combine(inputRoot,maskName));
                    Assert.That(Hash(waveBytes),Is.EqualTo((string)audit["files"][waveName]["sha256"]));Assert.That(Hash(maskBytes),Is.EqualTo((string)audit["files"][maskName]["sha256"]));
                    var waveform=new float[waveBytes.Length/4];var mask=new int[maskBytes.Length/4];
                    Buffer.BlockCopy(waveBytes,0,waveform,0,waveBytes.Length);Buffer.BlockCopy(maskBytes,0,mask,0,maskBytes.Length);
                    var count=mask.Count(v=>v==1);Assert.That(waveform.Length,Is.EqualTo(179200));
                    Assert.That(mask,Is.EqualTo(Enumerable.Repeat(1,count).Concat(Enumerable.Repeat(0,179200-count)).ToArray()));
                    Assert.That(waveform.Skip(count),Is.All.EqualTo(0));pcm=new AudioPcmInput(waveform.Take(count).ToArray());
                    Assert.That(pcm.ValidFrameCount,Is.EqualTo(valid));Assert.That(pcm.Waveform.ToArray(),Is.EqualTo(waveform));Assert.That(pcm.SampleMask.ToArray(),Is.EqualTo(mask));
                    inputHash=Hash(waveBytes.Concat(maskBytes).ToArray());report["input_path"]=Path.Combine(inputRoot,waveName);report["mask_path"]=Path.Combine(inputRoot,maskName);
                    report["valid_audio_frames"]=valid;report["source_sample_count"]=count;
                }
                var expected=reference["cases"].Single(r=>(string)r["id"]==id)["embedding"].Values<float>().ToArray();
                report["reference_sha256"]=Hash(referencePath);report["reference_case_id"]=id;report["input_sha256"]=inputHash;
                setup.Stop();report["input_setup_wall_seconds"]=setup.Elapsed.TotalSeconds;
                report["phase"]="load";report["load_begin_utc"]=DateTime.UtcNow.ToString("O");Save();
                var loadCpu=Cpu();var load=Stopwatch.StartNew();session=await bundle.LoadAsync(module,token);load.Stop();
                report["load_wall_seconds"]=load.Elapsed.TotalSeconds;report["load_process_cpu_seconds"]=(Cpu()-loadCpu).TotalSeconds;
                report["load_end_utc"]=DateTime.UtcNow.ToString("O");report["actual_backend"]=session.Backend.ToString();
                Assert.That(session.Backend,Is.EqualTo(Unity.InferenceEngine.BackendType.GPUCompute));Save();
                gpu?.Attach(session);
                using var timing=MediaStageTiming.Attach(session,e=>
                {
                    if(gpu?.ObserveSubmission(e)==true)return;
                    var encoder=(string)e["stage"]=="encoder";var begin=(string)e["event"]=="begin";
                    if(encoder&&!begin)gpu?.EndCall();
                    e["round"]=report["phase"]?.DeepClone();((JArray)report["stage_measurements"]).Add(e);Save();
                    if(encoder&&begin)gpu?.BeginCall((string)report["phase"]);
                },gpu!=null);
                foreach(var round in new[]{"first","warm","repeat-1","repeat-2"})
                {
                    gpu?.SetRound(round);
                    token.ThrowIfCancellationRequested();report["phase"]=round;Save();var cpu=Cpu();var watch=Stopwatch.StartNew();
                    var actual=image?await session.EmbedImageAsync(texture,cancellationToken:token):await session.EmbedAudioAsync(pcm,cancellationToken:token);
                    watch.Stop();var cpuSeconds=(Cpu()-cpu).TotalSeconds;
                    Assert.That(actual.Length,Is.EqualTo(768));Assert.That(actual.All(v=>!float.IsNaN(v)&&!float.IsInfinity(v)),Is.True);
                    var norm=actual.Sum(v=>(double)v*v);var cosine=TextSearchIndex.Cosine(actual,expected);
                    Assert.That(cosine,Is.GreaterThanOrEqualTo(GpuMediaExecutionPolicy.CurrentMinimumCosine));Assert.That(Math.Abs(norm-1),Is.LessThanOrEqualTo(.002));
                    var capture=Path.ChangeExtension(output,null)+"-"+round+".bin";Assert.That(File.Exists(capture),Is.False);
                    var bytes=new byte[768*4];Buffer.BlockCopy(actual,0,bytes,0,bytes.Length);File.WriteAllBytes(capture,bytes);
                    ((JArray)report["cases"]).Add(new JObject{["round"]=round,["id"]=id,["input_sha256"]=inputHash,
                        ["wall_seconds"]=watch.Elapsed.TotalSeconds,["process_cpu_seconds"]=cpuSeconds,["cosine"]=cosine,["norm_squared"]=norm,
                        ["capture"]=Path.GetFileName(capture),["capture_sha256"]=Hash(capture)});Save();
                }
                report["phase"]="release";report["release_begin_utc"]=DateTime.UtcNow.ToString("O");Save();
                var closeCpu=Cpu();var close=Stopwatch.StartNew();await session.CloseAsync();close.Stop();
                report["release_wall_seconds"]=close.Elapsed.TotalSeconds;report["release_process_cpu_seconds"]=(Cpu()-closeCpu).TotalSeconds;
                report["release_end_utc"]=DateTime.UtcNow.ToString("O");MultimodalSearchSampleGpuTests.AssertReleased(session);
                report["closed_worker_kernel_and_tensor_fields_inspected"]=true;report["session_closed"]=true;report["success"]=true;report["phase"]="closed";
            }
            catch(Exception e){report["error"]=e.ToString();throw;}
            finally{if(texture!=null)UnityEngine.Object.DestroyImmediate(texture);session?.Dispose();Save();}
        }
    }
}
