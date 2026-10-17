using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using System.Threading;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEditorInternal;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Optional, validation-only GPU queue markers. Never changes the SDK scheduler or reads tensors.</summary>
    public sealed class EncoderGpuDispatchTrace : IDisposable
    {
        sealed class Entry
        {
            public CustomSampler Sampler; public Recorder Recorder; public JObject Row;
            public Stopwatch Cpu; public bool Closed;
        }
        static int nextOperation = 1000000;
        readonly List<Entry> entries = new List<Entry>();
        readonly Dictionary<string, Entry> submissions = new Dictionary<string, Entry>();
        string round;
        readonly string id = Guid.NewGuid().ToString("N");
        GpuProjectionPipeline pipeline;
        IGpuTensorOperation[] operations, originalOperations, replacements;
        GpuPipelineStageCallbacks callbacks;
        int lastCapturedFrame = -1;
        const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;
        static object Read(object target, string name) => target.GetType().GetField(name, Private)?.GetValue(target)
            ?? throw new NotSupportedException("Pinned diagnostic field unavailable: " + name);

        public void Attach(GpuMediaSession session)
        {
            if (pipeline != null) throw new InvalidOperationException("Already attached.");
            var image = session.GetType().GetField("image", Private)?.GetValue(session);
            var audio = session.GetType().GetField("audio", Private)?.GetValue(session);
            var encoder = Read(image ?? audio ?? throw new ArgumentException("Open image/audio session required."), "encoder");
            pipeline = (GpuProjectionPipeline)Read(encoder, "pipeline");
        }
        Entry Create(string round, string kind, int stage, string marker)
        {
            var entry = new Entry { Sampler = CustomSampler.Create(marker, true),
                Row = new JObject { ["round"] = round, ["kind"] = kind, ["stage"] = stage,
                    ["marker"] = marker, ["gpu_ns"] = null, ["aborted"] = false,
                    ["scope"] = "GPU queue marker around submitted commands; not pure kernel active time or whole API latency." } };
            entry.Recorder = entry.Sampler.GetRecorder();
            if (!entry.Recorder.isValid) throw new NotSupportedException("GPU marker recorder unavailable.");
            entry.Recorder.enabled = true; entries.Add(entry); return entry;
        }
        public void BeginCall(string round)
        {
            if (pipeline == null || callbacks != null) throw new InvalidOperationException("Attach once and finish each encoder call.");
            var before = (Action[])Read(pipeline, "releaseGpuCache");
            var completed = (Action[])Read(pipeline, "releaseCompletedBackendCache");
            operations = (IGpuTensorOperation[])Read(pipeline, "projections");
            originalOperations = (IGpuTensorOperation[])operations.Clone(); replacements = new IGpuTensorOperation[operations.Length];
            var model = new Entry[before.Length];
            for (var stage = 0; stage < model.Length; stage++)
                model[stage] = Create(round, "model-segment-and-output-copy", stage, "EmbeddingGemma.EncoderDispatch." + id + "." + round + "." + stage);
            try
            {
                for (var i = 0; i < operations.Length; i++)
                {
                    var index = Interlocked.Increment(ref nextOperation); var stage = i; Entry projection = null;
                    replacements[i] = GpuOperationTrace.Wrap(originalOperations[i], index, row =>
                    {
                        if ((string)row["event"] != "begin")
                        {
                            projection.Closed = true;
                            projection.Row["aborted"] = (string)row["event"] != "submitted";
                            projection.Row["cpu_submission_wall_ns"] = row["cpu_wall_ns"]?.DeepClone();
                            projection.Row["operation_type"] = row["operation_type"]?.DeepClone();
                        }
                    }, true);
                    projection = Create(round, "projection-operation", stage, "EmbeddingGemma.Trace.Operation." + index);
                    operations[i] = replacements[i];
                }
                callbacks = new GpuPipelineStageCallbacks(before, completed,
                    stage => { model[stage].Cpu = Stopwatch.StartNew(); model[stage].Sampler.Begin(); },
                    (stage, aborted) =>
                    {
                        var entry = model[stage]; entry.Sampler.End(); entry.Cpu.Stop(); entry.Closed = true;
                        entry.Row["aborted"] = aborted;
                        entry.Row["cpu_submission_wall_ns"] = (long)(entry.Cpu.ElapsedTicks * (1000000000.0 / Stopwatch.Frequency));
                    });
                callbacks.BeginCall();
            }
            catch { EndCall(); throw; }
        }
        public void SetRound(string value)
        {
            if(callbacks!=null || submissions.Count!=0)throw new InvalidOperationException("Finish previous GPU scopes before changing round.");
            round=value ?? throw new ArgumentNullException(nameof(value));
        }
        public bool ObserveSubmission(JObject row)
        {
            var eventName=(string)row["event"];
            if(eventName!="submitting" && eventName!="submitted")return false;
            var stage=(string)row["stage"];
            if(stage=="encoder")return true; // Its original individual model/operation markers are more precise.
            if(stage!="preprocessing" && stage!="backbone")throw new NotSupportedException("Unexpected GPU stage.");
            if(eventName=="submitting")
            {
                if(round==null || submissions.ContainsKey(stage))throw new InvalidOperationException("GPU submissions must be serial within a fixed round.");
                var entry=Create(round,stage=="preprocessing"?"preprocessing-initial-synchronous-submission":"backbone-synchronous-submission",0,
                    "EmbeddingGemma.MediaSubmission."+id+"."+round+"."+stage);
                entry.Row["method"]=row["method"]?.DeepClone();entry.Row["asynchronous"]=row["asynchronous"]?.DeepClone();
                entry.Row["submission_contract"]="Pinned SDK implementation submits these GPU commands before invocation returns; async completion, fence wait and post-await cleanup are excluded. Queue time is not pure kernel time.";
                submissions.Add(stage,entry);entry.Cpu=Stopwatch.StartNew();entry.Sampler.Begin();
            }
            else
            {
                if(!submissions.TryGetValue(stage,out var entry))throw new InvalidOperationException("GPU submission end has no matching start.");
                entry.Sampler.End();entry.Cpu.Stop();entry.Closed=true;
                entry.Row["aborted"]=row["error_type"]?.Type!=JTokenType.Null;
                entry.Row["cpu_submission_wall_ns"]=(long)(entry.Cpu.ElapsedTicks*(1000000000.0/Stopwatch.Frequency));
                submissions.Remove(stage);
            }
            return true;
        }
        public void EndCall()
        {
            try { callbacks?.EndCall(); }
            finally
            {
                callbacks?.Dispose(); callbacks = null;
                if (operations != null && originalOperations != null)
                    for (var i = 0; i < operations.Length; i++)
                        if (ReferenceEquals(operations[i], replacements[i])) operations[i] = originalOperations[i];
                operations = originalOperations = replacements = null;
            }
        }
        public void Capture()
        {
            if (lastCapturedFrame == Time.frameCount) return;
            lastCapturedFrame = Time.frameCount;
            var frame = ProfilerDriver.lastFrameIndex;
            var availability = frame >= 0 ? (int)ProfilerDriver.GetGpuStatisticsAvailabilityState(frame) : 0;
            // Enabled | Gathered, matching the independently inspected history flags.
            if ((availability & 3) != 3) return;
            foreach (var entry in entries)
            {
                if (!entry.Closed || (bool)entry.Row["aborted"] || entry.Row["gpu_ns"].Type != JTokenType.Null) continue;
                var ns = entry.Recorder.gpuElapsedNanoseconds; var blocks = entry.Recorder.gpuSampleBlockCount;
                if (ns <= 0 || blocks <= 0) continue;
                entry.Row["gpu_ns"] = ns; entry.Row["gpu_blocks"] = blocks;
                entry.Row["observed_render_frame"] = Time.frameCount;
                entry.Row["observed_history_frame"] = frame;
                entry.Row["gpu_statistics_availability"] = availability;
            }
        }
        public JArray Snapshot(bool submissionScopes = false)
        {
            var result = new JArray();
            foreach (var entry in entries)
            {
                var kind=(string)entry.Row["kind"];
                var submission=kind=="preprocessing-initial-synchronous-submission" || kind=="backbone-synchronous-submission";
                if(submission==submissionScopes)result.Add(entry.Row.DeepClone());
            }
            return result;
        }
        public void Dispose()
        {
            try { EndCall(); }
            finally
            {
                foreach(var entry in submissions.Values)
                {
                    if(!entry.Closed){entry.Sampler.End();entry.Cpu?.Stop();entry.Closed=true;entry.Row["aborted"]=true;}
                }
                submissions.Clear();foreach (var entry in entries) entry.Recorder.enabled = false; pipeline = null;
            }
        }
    }
}
