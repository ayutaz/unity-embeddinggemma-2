using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEditor.Profiling;
using UnityEditorInternal;
using UnityEngine.Profiling;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Exact main-thread CPU markers; never derives allocations from heap size or GPU history.</summary>
    public static class StoppedManagedAllocationHistory
    {
        public static IReadOnlyDictionary<string,long> Read(IReadOnlyCollection<string> names,JObject diagnostics)
        {
            if(Profiler.enabled||ProfilerDriver.enabled)throw new InvalidOperationException("Stop CPU collection before traversing its history.");
            if(names==null)throw new ArgumentNullException(nameof(names));
            if(diagnostics==null)throw new ArgumentNullException(nameof(diagnostics));
            var expected=new HashSet<string>(names,StringComparer.Ordinal);
            var samples=new List<ManagedAllocationSample>();var rows=new JArray();
            var first=Math.Max(0,ProfilerDriver.firstFrameIndex);var last=ProfilerDriver.lastFrameIndex;
            diagnostics["first_frame"]=first;diagnostics["last_frame"]=last;diagnostics["collection_stopped"]=true;
            diagnostics["measurement_api"]="HierarchyFrameDataView.columnGcMemory / GetItemColumnDataAsDouble";
            diagnostics["marker_rows"]=rows;
            if(last-first>24)throw new InvalidOperationException("Editor allocation history exceeds the bounded 25-frame window.");
            var visited=0;using var iterator=new ProfilerFrameDataIterator();
            for(var frame=first;frame<=last;frame++)
            for(var thread=0;thread<iterator.GetThreadCount(frame);thread++)
            {
                using var view=ProfilerDriver.GetHierarchyFrameDataView(frame,thread,HierarchyFrameDataView.ViewModes.Default,HierarchyFrameDataView.columnGcMemory,false);
                if(!view.valid||view.threadName!="Main Thread")continue;
                var pending=new Stack<int>();pending.Push(view.GetRootItemID());var children=new List<int>();
                while(pending.Count>0)
                {
                    if(++visited>200000)throw new InvalidOperationException("Editor allocation history exceeds the bounded node budget.");
                    var item=pending.Pop();if(item<0)continue;
                    view.GetItemChildren(item,children);foreach(var child in children)pending.Push(child);
                    var name=view.GetItemName(item);if(!expected.Contains(name))continue;
                    var bytes=view.GetItemColumnDataAsDouble(item,HierarchyFrameDataView.columnGcMemory);
                    samples.Add(new ManagedAllocationSample(name,view.threadName,frame,bytes));
                    rows.Add(new JObject { ["marker"]=name,["frame"]=frame,["thread"]=view.threadName,["raw_managed_bytes"]=bytes });
                }
            }
            diagnostics["nodes_examined"]=visited;
            return ManagedAllocationSamples.RequireSingleMainThreadSamples(names,samples);
        }
    }
}
