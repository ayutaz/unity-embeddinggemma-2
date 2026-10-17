using System;
using System.Collections;
using System.IO;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using UnityEngine.Profiling;
using UnityEditor.Profiling;
using UnityEditorInternal;
using UnityEditor;

namespace OwnedAllocation.Tests
{
    public sealed class ManagedAllocationCalibrationTests
    {
        [UnityTearDown] public IEnumerator Exit() { if (Application.isPlaying) yield return new ExitPlayMode(); }
        [UnityTest] public IEnumerator KnownByteArrayAllocationsHaveDistinctScopedByteTotals()
        {
            yield return new EnterPlayMode();
            var output=Environment.GetEnvironmentVariable("EG_ALLOCATION_CALIBRATION_REPORT");
            Assert.That(output,Is.Not.Null.And.Not.Empty); Assert.That(File.Exists(output),Is.False);
            var report=new JObject { ["success"]=false,["unity"]=Application.unityVersion,
                ["model_loaded"]=false,["scope"]="Known main-thread managed allocations only; no model/GPU qualification." };
            try
            {
                yield return ManagedAllocationProbe.Capture(report);
                Assert.That(report.Value<long>("noop_bytes"),Is.EqualTo(0));
                Assert.That(report.Value<long>("small_bytes"),Is.GreaterThanOrEqualTo(65536));
                Assert.That(report.Value<long>("small_bytes"),Is.LessThan(65536+256));
                Assert.That(report.Value<long>("large_bytes")-report.Value<long>("small_bytes"),Is.EqualTo(65536));
                report["success"]=true;
            }
            finally { File.WriteAllText(output,report.ToString()); }
            yield return new ExitPlayMode();
        }
    }
    public static class ManagedAllocationProbe
    {
        public static IEnumerator Capture(JObject report)
        {
            Assert.That(ProfilerDriver.GetConnectionIdentifier(ProfilerDriver.connectedProfiler),Does.Contain("Editor"));
            var enabled=Profiler.enabled;var driver=ProfilerDriver.enabled;var editor=ProfilerDriver.profileEditor;
            var background=Application.runInBackground;
            var cpu=Profiler.GetAreaEnabled(ProfilerArea.CPU);var memory=Profiler.GetAreaEnabled(ProfilerArea.Memory);
            var names=new[]{"noop","small","large"};
            var actions=new Action[]{()=>{},()=>{var data=new byte[65536];data[0]=1;GC.KeepAlive(data);},
                ()=>{var data=new byte[131072];data[0]=1;GC.KeepAlive(data);}};
            var samplers=new CustomSampler[names.Length];
            for(var i=0;i<names.Length;i++)samplers[i]=CustomSampler.Create("OwnedAllocation."+names[i]);
            try
            {
                Application.runInBackground=true;
                ProfilerDriver.enabled=false;Profiler.enabled=false;ProfilerDriver.ClearAllFrames();
                ProfilerDriver.profileEditor=true;ProfilerDriver.enabled=true;Profiler.enabled=true;
                Profiler.SetAreaEnabled(ProfilerArea.CPU,true);Profiler.SetAreaEnabled(ProfilerArea.Memory,true);
                for(var frame=0;frame<4;frame++){EditorApplication.QueuePlayerLoopUpdate();yield return null;}
                report["action_frame"]=Time.frameCount;
                for(var i=0;i<names.Length;i++)
                { samplers[i].Begin();try{actions[i]();}finally{samplers[i].End();} }
                for(var frame=0;frame<5;frame++){EditorApplication.QueuePlayerLoopUpdate();yield return null;}
                ProfilerDriver.enabled=false;Profiler.enabled=false;
                report["after_action_frame"]=Time.frameCount;
                report["first_frame"]=ProfilerDriver.firstFrameIndex;report["last_frame"]=ProfilerDriver.lastFrameIndex;
                report["measurement_api"]="HierarchyFrameDataView.columnGcMemory";
                report["column"]=HierarchyFrameDataView.columnGcMemory;
                report["scope"]="Stopped CPU profiler history: inclusive managed bytes within three exact main-thread markers. Excludes other threads, allocations outside markers and native memory.";
                var rows=new JArray();report["marker_rows"]=rows;
                var visited=0;using var iterator=new ProfilerFrameDataIterator();
                for(var frame=Math.Max(0,ProfilerDriver.firstFrameIndex);frame<=ProfilerDriver.lastFrameIndex;frame++)
                for(var thread=0;thread<iterator.GetThreadCount(frame);thread++)
                {
                    using var view=ProfilerDriver.GetHierarchyFrameDataView(frame,thread,HierarchyFrameDataView.ViewModes.Default,HierarchyFrameDataView.columnGcMemory,false);
                    if(!view.valid||view.threadName!="Main Thread")continue;
                    var pending=new Stack<int>();pending.Push(view.GetRootItemID());var children=new List<int>();
                    while(pending.Count>0)
                    {
                        if(++visited>200000)throw new InvalidOperationException("Allocation history exceeds bounded observation budget.");
                        var id=pending.Pop();if(id<0)continue;view.GetItemChildren(id,children);foreach(var child in children)pending.Push(child);
                        var name=view.GetItemName(id);var index=Array.FindIndex(samplers,s=>s.name==name);if(index<0)continue;
                        var bytes=view.GetItemColumnDataAsSingle(id,HierarchyFrameDataView.columnGcMemory);
                        Assert.That(float.IsNaN(bytes)||float.IsInfinity(bytes),Is.False);Assert.That(bytes,Is.GreaterThanOrEqualTo(0));
                        Assert.That(report[names[index]+"_bytes"],Is.Null,"Each calibration marker must occur exactly once.");
                        report[names[index]+"_bytes"]=(long)Math.Round(bytes);
                        rows.Add(new JObject{["marker"]=name,["frame"]=frame,["thread"]=view.threadName,["bytes"]=(long)Math.Round(bytes)});
                    }
                }
                report["nodes_examined"]=visited;
                File.WriteAllText(Environment.GetEnvironmentVariable("EG_ALLOCATION_CALIBRATION_REPORT"),report.ToString());
                Assert.That(rows.Count,Is.EqualTo(3),"Missing marker is unavailable, never zero allocation.");
            }
            finally
            {
                ProfilerDriver.enabled=false;ProfilerDriver.profileEditor=editor;
                Profiler.SetAreaEnabled(ProfilerArea.CPU,cpu);Profiler.SetAreaEnabled(ProfilerArea.Memory,memory);
                Profiler.enabled=enabled;ProfilerDriver.enabled=driver;
                Application.runInBackground=background;
            }
        }
    }
}
