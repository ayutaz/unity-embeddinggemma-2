using System;
using System.Collections.Generic;
using System.Reflection;
using Newtonsoft.Json.Linq;
using UnityEditor.Profiling;
using UnityEditorInternal;
namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Read GPU columns from recorded history. Never substitute CPU columns or export unrelated samples.</summary>
    public static class GpuProfilerHistory
    {
        public static JArray ReadExactMarker(string name, JObject diagnostics = null)
        {
            if (string.IsNullOrEmpty(name)) throw new ArgumentException("An exact benchmark marker is required.", nameof(name));
            var field = typeof(HierarchyFrameDataView).GetField("columnTotalGpuTime", BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public);
            if (field == null || !field.IsLiteral) throw new NotSupportedException("Pinned Unity GPU column metadata is unavailable.");
            var column = (int)field.GetRawConstantValue(); var rows = new JArray(); var visited = 0;
            var first = Math.Max(ProfilerDriver.firstFrameIndex, ProfilerDriver.lastFrameIndex - 24);
            var views = 0; var matches = 0;
            if (diagnostics != null)
            {
                diagnostics["first_frame"] = ProfilerDriver.firstFrameIndex; diagnostics["last_frame"] = ProfilerDriver.lastFrameIndex;
                diagnostics["gpu_column"] = column;
            }
            using var iterator = new ProfilerFrameDataIterator();
            for (var frame = first; frame <= ProfilerDriver.lastFrameIndex; frame++)
            {
                var threads = iterator.GetThreadCount(frame);
                for (var thread = 0; thread < threads; thread++)
                {
                    using var view = ProfilerDriver.GetHierarchyFrameDataView(frame, thread,
                        HierarchyFrameDataView.ViewModes.Default, column, false);
                    if (!view.valid) continue;
                    views++;
                    var pending = new Stack<int>(); pending.Push(view.GetRootItemID()); var children = new List<int>();
                    while (pending.Count > 0)
                    {
                        if (++visited > 200000) throw new InvalidOperationException("GPU history traversal exceeded its bounded observation budget.");
                        var id = pending.Pop(); if (id < 0) continue;
                        view.GetItemChildren(id, children);
                        foreach (var child in children) pending.Push(child);
                        if (view.GetItemName(id) != name) continue;
                        matches++;
                        var ms = view.GetItemColumnDataAsSingle(id, column);
                        if (!(ms > 0) || float.IsNaN(ms) || float.IsInfinity(ms)) continue;
                        var descendants = new JArray();
                        foreach (var child in children)
                        {
                            var childMs = view.GetItemColumnDataAsSingle(child, column);
                            if (childMs > 0 && !float.IsNaN(childMs) && !float.IsInfinity(childMs))
                                descendants.Add(new JObject { ["marker"] = view.GetItemName(child), ["gpu_ns"] = (long)Math.Round(childMs * 1000000.0) });
                        }
                        rows.Add(new JObject { ["frame"] = frame, ["thread_index"] = thread, ["thread_name"] = view.threadName,
                            ["marker"] = name, ["gpu_ns"] = (long)Math.Round(ms * 1000000.0), ["gpu_children"] = descendants,
                            ["scope"] = "GPU hierarchy inclusive duration; parent and child durations overlap" });
                    }
                }
            }
            if (diagnostics != null) { diagnostics["views_examined"] = views; diagnostics["nodes_examined"] = visited; diagnostics["matching_marker_count"] = matches; }
            return rows;
        }
    }
}
