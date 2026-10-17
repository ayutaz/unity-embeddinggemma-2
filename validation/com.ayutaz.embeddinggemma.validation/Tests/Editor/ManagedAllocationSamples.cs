using System;
using System.Collections.Generic;

namespace EmbeddingGemma.Validation.Tests
{
    public readonly struct ManagedAllocationSample
    {
        public readonly string Marker, Thread;
        public readonly int Frame;
        public readonly double Bytes;
        public ManagedAllocationSample(string marker, string thread, int frame, double bytes)
        { Marker=marker; Thread=thread; Frame=frame; Bytes=bytes; }
    }

    public static class ManagedAllocationSamples
    {
        public static IReadOnlyDictionary<string,long> RequireSingleMainThreadSamples(
            IEnumerable<string> expected, IEnumerable<ManagedAllocationSample> samples)
        {
            if(expected==null)throw new ArgumentNullException(nameof(expected));
            if(samples==null)throw new ArgumentNullException(nameof(samples));
            var names=new HashSet<string>(StringComparer.Ordinal);
            foreach(var name in expected)
                if(string.IsNullOrWhiteSpace(name)||!names.Add(name))
                    throw new ArgumentException("Expected allocation markers must be distinct and nonempty.",nameof(expected));
            if(names.Count==0)throw new ArgumentException("Expected allocation markers are required.",nameof(expected));
            var result=new Dictionary<string,long>(StringComparer.Ordinal);
            foreach(var sample in samples)
            {
                if(sample.Marker==null||!names.Contains(sample.Marker)||sample.Thread!="Main Thread")continue;
                var bytes=sample.Bytes;
                if(sample.Frame<0||double.IsNaN(bytes)||double.IsInfinity(bytes)||bytes<0||bytes>9007199254740992.0||bytes!=Math.Truncate(bytes))
                    throw new InvalidOperationException("Allocation marker does not have exact nonnegative managed bytes on a valid frame.");
                if(result.ContainsKey(sample.Marker))throw new InvalidOperationException("Allocation marker was observed more than once.");
                result.Add(sample.Marker,(long)bytes);
            }
            if(result.Count!=names.Count)throw new InvalidOperationException("Missing allocation marker is unavailable, never zero.");
            return result;
        }
    }
}
