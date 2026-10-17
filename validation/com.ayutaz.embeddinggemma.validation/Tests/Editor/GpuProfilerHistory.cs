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
        public static bool TryGpuMillisecondsToNanoseconds(float milliseconds, out long nanoseconds)
        {
            nanoseconds = 0;
            if (!(milliseconds > 0) || float.IsNaN(milliseconds) || float.IsInfinity(milliseconds)) return false;
            var rounded = Math.Round(milliseconds * 1000000.0);
            // Native history can expose an unavailable float.MaxValue sentinel.
            // Reject it before an unchecked cast could turn it into a negative time.
            if (rounded < 1 || rounded >= 9223372036854775808.0) return false;
            nanoseconds = (long)rounded;
            return true;
        }
        public static JArray ReadExactMarker(string name, JObject diagnostics = null)
        {
            if (string.IsNullOrEmpty(name)) throw new ArgumentException("An exact benchmark marker is required.", nameof(name));
            var field = typeof(HierarchyFrameDataView).GetField("columnTotalGpuTime", BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public);
            if (field == null || !field.IsLiteral) throw new NotSupportedException("Pinned Unity GPU column metadata is unavailable.");
            var column = (int)field.GetRawConstantValue(); var rows = new JArray(); var visited = 0;
            var last = ProfilerDriver.lastFrameIndex;
            var first = Math.Max(ProfilerDriver.firstFrameIndex, last - 24);
            var views = 0; var matches = 0; var gpuUnavailableFrames = 0;
            if (diagnostics != null)
            {
                diagnostics["first_frame"] = ProfilerDriver.firstFrameIndex; diagnostics["last_frame"] = last;
                diagnostics["collection_stopped"] = !ProfilerDriver.enabled && !UnityEngine.Profiling.Profiler.enabled;
                diagnostics["gpu_column"] = column;
            }
            using var iterator = new ProfilerFrameDataIterator();
            for (var frame = first; frame <= last; frame++)
            {
                var availability = ProfilerDriver.GetGpuStatisticsAvailabilityState(frame);
                const GpuProfilingStatisticsAvailabilityStates required = GpuProfilingStatisticsAvailabilityStates.Enabled | GpuProfilingStatisticsAvailabilityStates.Gathered;
                if ((availability & required) != required) { gpuUnavailableFrames++; continue; }
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
                        if (!TryGpuMillisecondsToNanoseconds(ms, out var gpuNanoseconds)) continue;
                        var descendants = new JArray();
                        foreach (var child in children)
                        {
                            var childMs = view.GetItemColumnDataAsSingle(child, column);
                            if (TryGpuMillisecondsToNanoseconds(childMs, out var childNanoseconds))
                                descendants.Add(new JObject { ["marker"] = view.GetItemName(child), ["gpu_ns"] = childNanoseconds });
                        }
                        var managedBytes = view.GetItemColumnDataAsSingle(id, HierarchyFrameDataView.columnGcMemory);
                        if (float.IsNaN(managedBytes) || float.IsInfinity(managedBytes) || managedBytes < 0)
                            throw new InvalidOperationException("Managed allocation data is unavailable for this exact CPU marker.");
                        rows.Add(new JObject { ["frame"] = frame, ["thread_index"] = thread, ["thread_name"] = view.threadName,
                            ["marker"] = name, ["gpu_ns"] = gpuNanoseconds, ["gpu_children"] = descendants,
                            ["cpu_gc_allocated_bytes"] = (long)Math.Round(managedBytes),
                            ["cpu_gc_scope"] = "Inclusive managed bytes within this CPU marker on the named thread; not GPU or native memory.",
                            ["cpu_total_ns"] = (long)Math.Round(view.GetItemColumnDataAsSingle(id, HierarchyFrameDataView.columnTotalTime) * 1000000.0),
                            ["cpu_gc_allocated_bytes"] = (long)Math.Round(view.GetItemColumnDataAsDouble(id, HierarchyFrameDataView.columnGcMemory)),
                            ["marker_flags"] = (int)view.GetItemMarkerFlags(id),
                            ["gpu_statistics_availability"] = (int)availability,
                            ["scope"] = "GPU hierarchy inclusive duration; parent and child durations overlap" });
                    }
                }
            }
            if (diagnostics != null) { diagnostics["views_examined"] = views; diagnostics["nodes_examined"] = visited; diagnostics["matching_marker_count"] = matches; diagnostics["gpu_unavailable_frames_skipped"] = gpuUnavailableFrames; }
            return rows;
        }
    }
}
