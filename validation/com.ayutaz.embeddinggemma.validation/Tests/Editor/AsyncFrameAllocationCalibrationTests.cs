using System;
using System.Collections;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditorInternal;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class AsyncFrameAllocationCalibrationTests
    {
        [UnityTearDown] public IEnumerator Exit(){if(Application.isPlaying)yield return new ExitPlayMode();}
        static async Task AllocateAcrossFrames(int bytes,CustomSampler before,CustomSampler after,SynchronizationContext context,JObject row)
        {
            Assert.That(SynchronizationContext.Current,Is.SameAs(context));
            byte[] first=null,second=null;row["payload_before_render_frame"]=Time.frameCount;
            before.Begin();try{if(bytes>0){first=new byte[bytes/2];first[0]=1;}}finally{before.End();}
            var frame=Time.frameCount;do{await Task.Yield();}while(Time.frameCount<=frame);
            Assert.That(SynchronizationContext.Current,Is.SameAs(context));
            row["payload_after_render_frame"]=Time.frameCount;
            after.Begin();try{if(bytes>0){second=new byte[bytes/2];second[0]=1;}}finally{after.End();}
            GC.KeepAlive(first);GC.KeepAlive(second);
        }
        [UnityTest,Timeout(120000)]
        public IEnumerator KnownArraysAcrossAwaitedFramesAppearInCalibratedFrameWideCounters()
        {
            yield return new EnterPlayMode();
            var output=Environment.GetEnvironmentVariable("EG_ASYNC_ALLOCATION_CALIBRATION_REPORT");
            Assert.That(output,Is.Not.Null.And.Not.Empty);Assert.That(File.Exists(output),Is.False);
            var context=SynchronizationContext.Current;Assert.That(context,Is.Not.Null);
            var report=new JObject { ["success"]=false,["unity"]=Application.unityVersion,["model_loaded"]=false,["gpu_execution_validated"]=false,
                ["validation_source"]=Environment.GetEnvironmentVariable("EG_VALIDATION_SOURCE"),["cases"]=new JArray(),
                ["scope"]="Known allocations across actual awaited render frames. Frame-wide GC counter includes unrelated Editor activity and overhead; no claim of exclusive SDK allocations or real-model execution." };
            var enabled=Profiler.enabled;var driver=ProfilerDriver.enabled;var editor=ProfilerDriver.profileEditor;
            var cpu=Profiler.GetAreaEnabled(ProfilerArea.CPU);var memory=Profiler.GetAreaEnabled(ProfilerArea.Memory);
            var background=Application.runInBackground;var fps=Application.targetFrameRate;var vsync=QualitySettings.vSyncCount;
            var host=new GameObject("Owned allocation calibration camera");var target=new RenderTexture(16,16,0);var camera=host.AddComponent<Camera>();
            camera.targetTexture=target;camera.cullingMask=0;camera.clearFlags=CameraClearFlags.SolidColor;
            try
            {
                Application.runInBackground=true;Application.targetFrameRate=60;QualitySettings.vSyncCount=0;
                foreach(var bytes in new[]{0,4*1024*1024,8*1024*1024})
                {
                    ProfilerDriver.enabled=false;Profiler.enabled=false;ProfilerDriver.ClearAllFrames();
                    ProfilerDriver.profileEditor=true;ProfilerDriver.enabled=true;Profiler.enabled=true;
                    Profiler.SetAreaEnabled(ProfilerArea.CPU,true);Profiler.SetAreaEnabled(ProfilerArea.Memory,true);
                    for(var i=0;i<4;i++){EditorApplication.QueuePlayerLoopUpdate();yield return null;}
                    var name="FrameAllocation.Control."+Guid.NewGuid().ToString("N");
                    var before=CustomSampler.Create(name+".before");var after=CustomSampler.Create(name+".after");
                    var row=new JObject { ["payload_bytes"]=bytes };((JArray)report["cases"]).Add(row);
                    using(var capture=new FrameManagedAllocationCapture(name))
                    {
                        var task=AllocateAcrossFrames(bytes,before,after,context,row);var wait=Stopwatch.StartNew();
                        while(!task.IsCompleted&&wait.Elapsed.TotalSeconds<15){EditorApplication.QueuePlayerLoopUpdate();yield return null;}
                        Assert.That(task.IsCompleted,Is.True,"The controlled Task must cross actual render frames.");task.GetAwaiter().GetResult();capture.Complete();
                        wait.Restart();while(!capture.TryClose()&&wait.Elapsed.TotalSeconds<15){EditorApplication.QueuePlayerLoopUpdate();yield return null;}
                        row["counter"]=capture.Snapshot();
                    }
                    for(var i=0;i<5;i++){EditorApplication.QueuePlayerLoopUpdate();yield return null;}
                    ProfilerDriver.enabled=false;Profiler.enabled=false;
                    var history=new JObject();row["history"]=history;
                    var exact=StoppedManagedAllocationHistory.Read(new[]{before.name,after.name},history);
                    row["before_marker_managed_bytes"]=exact[before.name];row["after_marker_managed_bytes"]=exact[after.name];
                    Assert.That((int)row["payload_after_render_frame"],Is.GreaterThan((int)row["payload_before_render_frame"]));
                    foreach(var marker in new[]{before.name,after.name})
                        if(bytes==0)Assert.That(exact[marker],Is.Zero);else Assert.That(exact[marker],Is.InRange((long)bytes/2,(long)bytes/2+255));
                    var total=(long)row["counter"]["managed_bytes_in_recorded_frames"];
                    Assert.That(total,Is.GreaterThanOrEqualTo((long)bytes).And.LessThanOrEqualTo((long)bytes+1024*1024),"Predeclared 1MiB background/observer ceiling; failure does not relax it.");
                    if(bytes>0)Assert.That(row["counter"]["samples"].Values<long>().Count(value=>value>=bytes/2),Is.GreaterThanOrEqualTo(2),"Both distinct frame payloads must be visible.");
                    Assert.That(SynchronizationContext.Current,Is.SameAs(context));File.WriteAllText(output,report.ToString());
                }
                var cases=(JArray)report["cases"];var difference=(long)cases[2]["counter"]["managed_bytes_in_recorded_frames"]-(long)cases[1]["counter"]["managed_bytes_in_recorded_frames"];
                Assert.That(difference,Is.InRange(3L*1024*1024,5L*1024*1024));report["success"]=true;
            }
            finally
            {
                ProfilerDriver.enabled=false;Profiler.enabled=false;ProfilerDriver.profileEditor=editor;
                Profiler.SetAreaEnabled(ProfilerArea.CPU,cpu);Profiler.SetAreaEnabled(ProfilerArea.Memory,memory);
                Profiler.enabled=enabled;ProfilerDriver.enabled=driver;Application.runInBackground=background;Application.targetFrameRate=fps;QualitySettings.vSyncCount=vsync;
                camera.targetTexture=null;UnityEngine.Object.DestroyImmediate(host);target.Release();UnityEngine.Object.DestroyImmediate(target);File.WriteAllText(output,report.ToString());
            }
            yield return new ExitPlayMode();
        }
    }
}
