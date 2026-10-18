using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.Search;
using UnityEditorInternal;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class EditorSearchAllocationTests
    {
        static void Sample(CustomSampler sampler,Action action)
        { sampler.Begin();try{action();}finally{sampler.End();} }

        [UnityTest,Timeout(120000)]
        public IEnumerator SearchAndEnumerationAllocationsHaveCalibratedExactCpuMarkers()
        {
            var output=Environment.GetEnvironmentVariable("EG_EDITOR_ALLOCATION_REPORT");
            Assert.That(output,Is.Not.Null.And.Not.Empty);Assert.That(File.Exists(output),Is.False);
            const string expected="Packages/com.ayutaz.embeddinggemma/Runtime/TextEmbedder.cs";
            var guid=AssetDatabase.AssetPathToGUID(expected);Assert.That(guid,Is.Not.Null.And.Not.Empty);
            var report=new JObject { ["success"]=false,["unity"]=Application.unityVersion,["model_loaded"]=false,["gpu_execution_validated"]=false,
                ["runtime_source"]=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE"),["validation_source"]=Environment.GetEnvironmentVariable("EG_VALIDATION_SOURCE"),
                ["scope"]="Main-thread inclusive managed bytes in exact synchronous Editor Search enumeration/query markers. Calibration and searches share one stopped CPU capture. Other threads, background index jobs, native memory and async completion are excluded.",
                ["search_route"]="Synchronous asset provider with Packages and WantsMore; includes path candidates, not isolated index latency or explicit rebuild.",
                ["play_mode"]=Application.isPlaying,["expected_asset"]=expected,["expected_guid"]=guid,["rounds"]=new JArray() };
            var enabled=Profiler.enabled;var driver=ProfilerDriver.enabled;var editor=ProfilerDriver.profileEditor;
            var cpu=Profiler.GetAreaEnabled(ProfilerArea.CPU);var memory=Profiler.GetAreaEnabled(ProfilerArea.Memory);
            var background=Application.runInBackground;
            try
            {
                report["phase"]="index_readiness";File.WriteAllText(output,report.ToString());
                Assert.That(SearchService.EnumerateDatabases().Any(),Is.True);
                var readiness=Stopwatch.StartNew();
                while(!SearchService.IsIndexReady(null)&&readiness.Elapsed.TotalSeconds<90)
                { EditorApplication.QueuePlayerLoopUpdate();yield return null; }
                report["index_ready_before_measurement"]=SearchService.IsIndexReady(null);
                Assert.That(report.Value<bool>("index_ready_before_measurement"),Is.True);
                Assert.That(ProfilerDriver.GetConnectionIdentifier(ProfilerDriver.connectedProfiler),Does.Contain("Editor"));
                var prefix="EmbeddingGemma.EditorAllocation."+Guid.NewGuid().ToString("N")+".";
                var names=new List<string>();
                CustomSampler Marker(string name){var full=prefix+name;names.Add(full);return CustomSampler.Create(full);}
                var noop=Marker("noop");var small=Marker("small");var large=Marker("large");
                var labels=new[]{"first","warm","repeat-1","repeat-2"};
                var enumeration=labels.Select(label=>Marker(label+".enumeration")).ToArray();
                var searches=labels.Select(label=>Marker(label+".search")).ToArray();
                Action nothing=()=>{};
                Action allocateSmall=()=>{var data=new byte[65536];data[0]=1;GC.KeepAlive(data);};
                Action allocateLarge=()=>{var data=new byte[131072];data[0]=1;GC.KeepAlive(data);};
                Application.runInBackground=true;
                ProfilerDriver.enabled=false;Profiler.enabled=false;ProfilerDriver.ClearAllFrames();
                ProfilerDriver.profileEditor=true;ProfilerDriver.enabled=true;Profiler.enabled=true;
                Profiler.SetAreaEnabled(ProfilerArea.CPU,true);Profiler.SetAreaEnabled(ProfilerArea.Memory,true);
                for(var frame=0;frame<4;frame++){EditorApplication.QueuePlayerLoopUpdate();yield return null;}
                report["phase"]="recording";
                Sample(noop,nothing);Sample(small,allocateSmall);Sample(large,allocateLarge);
                for(var index=0;index<labels.Length;index++)
                {
                    var databaseCount=0;SearchItem[] items=null;
                    Action enumerate=()=>{var databases=SearchService.EnumerateDatabases().ToArray();databaseCount=databases.Length;GC.KeepAlive(databases);};
                    const SearchFlags flags=SearchFlags.Synchronous|SearchFlags.Packages|SearchFlags.WantsMore;
                    Action search=()=>{using var context=SearchService.CreateContext("asset","TextEmbedder",flags);items=SearchService.GetItems(context,flags).ToArray();};
                    using var process=Process.GetCurrentProcess();var processCpu=process.TotalProcessorTime;var timer=Stopwatch.StartNew();
                    Sample(enumeration[index],enumerate);timer.Stop();process.Refresh();
                    var enumerationWall=timer.Elapsed.TotalSeconds;var enumerationCpu=(process.TotalProcessorTime-processCpu).TotalSeconds;
                    processCpu=process.TotalProcessorTime;timer.Restart();Sample(searches[index],search);timer.Stop();process.Refresh();
                    var queryCpu=(process.TotalProcessorTime-processCpu).TotalSeconds;
                    var found=items.Any(item=>item.id==expected||item.id==guid||AssetDatabase.GetAssetPath(item.ToObject())==expected);
                    ((JArray)report["rounds"]).Add(new JObject { ["round"]=labels[index],["enumeration_marker"]=enumeration[index].name,["search_marker"]=searches[index].name,
                        ["enumeration_wall_seconds"]=enumerationWall,["enumeration_process_cpu_seconds"]=enumerationCpu,["search_wall_seconds"]=timer.Elapsed.TotalSeconds,
                        ["search_process_cpu_seconds"]=queryCpu,["database_count"]=databaseCount,["result_count"]=items.Length,["expected_asset_found"]=found });
                    Assert.That(found,Is.True,"An allocation sample without the expected SDK asset is not accepted.");
                }
                for(var frame=0;frame<5;frame++){EditorApplication.QueuePlayerLoopUpdate();yield return null;}
                ProfilerDriver.enabled=false;Profiler.enabled=false;
                var diagnostics=new JObject();report["history"]=diagnostics;report["phase"]="stopped_history";
                var bytes=StoppedManagedAllocationHistory.Read(names,diagnostics);
                report["calibration"]=new JObject { ["noop_bytes"]=bytes[noop.name],["small_bytes"]=bytes[small.name],["large_bytes"]=bytes[large.name] };
                Assert.That(bytes[noop.name],Is.Zero);Assert.That(bytes[small.name],Is.InRange(65536L,65791L));
                Assert.That(bytes[large.name]-bytes[small.name],Is.EqualTo(65536));
                foreach(var row in (JArray)report["rounds"])
                { row["enumeration_main_thread_managed_bytes"]=bytes[(string)row["enumeration_marker"]];row["search_main_thread_managed_bytes"]=bytes[(string)row["search_marker"]]; }
                report["phase"]="closed";report["success"]=true;
            }
            finally
            {
                ProfilerDriver.enabled=false;Profiler.enabled=false;ProfilerDriver.profileEditor=editor;
                Profiler.SetAreaEnabled(ProfilerArea.CPU,cpu);Profiler.SetAreaEnabled(ProfilerArea.Memory,memory);
                Profiler.enabled=enabled;ProfilerDriver.enabled=driver;Application.runInBackground=background;
                File.WriteAllText(output,report.ToString());
            }
        }
    }
}
