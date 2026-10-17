using System;
using System.Linq;
using System.Threading;
using Newtonsoft.Json.Linq;
using Unity.Profiling;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Frame-wide counter window; includes unrelated Editor work and never changes a SynchronizationContext.</summary>
    public sealed class FrameManagedAllocationCapture : IDisposable
    {
        const int Capacity=65536;
        readonly int owner=Thread.CurrentThread.ManagedThreadId;
        readonly JObject row;
        ProfilerRecorder recorder;
        bool completed,closed,disposed;
        int countAtCompletion;
        public FrameManagedAllocationCapture(string name)
        {
            if(string.IsNullOrWhiteSpace(name))throw new ArgumentException("A window name is required.",nameof(name));
            row=new JObject { ["name"]=name,["begin_utc"]=DateTime.UtcNow.ToString("O"),["begin_render_frame"]=Time.frameCount,
                ["scope"]="GC Allocated In Frame counter samples collected through the first new sample after completion. Includes containing frames, observation overhead and unrelated Editor work. Overlapping windows must not be summed; not exclusive SDK allocation." };
            recorder=ProfilerRecorder.StartNew(ProfilerCategory.Memory,"GC Allocated In Frame",Capacity,
                ProfilerRecorderOptions.SumAllSamplesInFrame|ProfilerRecorderOptions.StartImmediately);
            if(!recorder.Valid||recorder.UnitType.ToString()!="Bytes"||recorder.DataType.ToString()!="Int64")
            {
                var description=$"valid={recorder.Valid}, unit={recorder.UnitType}, type={recorder.DataType}";
                recorder.Dispose();disposed=true;throw new NotSupportedException("GC frame recorder is unavailable or has unexpected units: "+description);
            }
            row["counter"]="GC Allocated In Frame";row["unit"]=recorder.UnitType.ToString();row["data_type"]=recorder.DataType.ToString();row["capacity"]=Capacity;
        }
        void CheckOwner()
        {
            if(disposed)throw new ObjectDisposedException(nameof(FrameManagedAllocationCapture));
            if(Thread.CurrentThread.ManagedThreadId!=owner)throw new InvalidOperationException("Read the frame counter on its creating thread.");
        }
        public void Complete()
        {
            CheckOwner();if(completed)throw new InvalidOperationException("Complete the operation window exactly once.");
            completed=true;countAtCompletion=recorder.Count;
            row["completed_utc"]=DateTime.UtcNow.ToString("O");row["completed_render_frame"]=Time.frameCount;row["samples_at_completion"]=countAtCompletion;
        }
        public bool TryClose()
        {
            CheckOwner();if(closed)return true;if(!completed)return false;
            if(recorder.WrappedAround||recorder.Count>=Capacity)throw new InvalidOperationException("Allocation window exceeded its bounded sample storage.");
            if(recorder.Count<=countAtCompletion)return false;
            recorder.Stop();var values=recorder.ToArray().Select(sample=>sample.Value).ToArray();
            var result=FrameAllocationWindow.ThroughFirstSampleAfterCompletion(values,countAtCompletion,Capacity,recorder.WrappedAround);
            row["closed_utc"]=DateTime.UtcNow.ToString("O");row["closed_render_frame"]=Time.frameCount;
            row["included_samples"]=result.IncludedSamples;row["discarded_tail_samples"]=result.DiscardedTailSamples;
            row["managed_bytes_in_recorded_frames"]=result.ManagedBytes;row["samples"]=JArray.FromObject(values.Take(result.IncludedSamples));
            row["wrapped"]=recorder.WrappedAround;closed=true;return true;
        }
        public JObject Snapshot()
        {if(!closed)throw new InvalidOperationException("A pending allocation window is not a result.");return (JObject)row.DeepClone();}
        public void Dispose(){if(disposed)return;recorder.Dispose();disposed=true;}
    }
}
