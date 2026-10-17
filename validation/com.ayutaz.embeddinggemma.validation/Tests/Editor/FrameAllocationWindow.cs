using System;

namespace EmbeddingGemma.Validation.Tests
{
    public readonly struct FrameAllocationSummary
    {
        public readonly long ManagedBytes;
        public readonly int IncludedSamples,DiscardedTailSamples;
        public FrameAllocationSummary(long bytes,int included,int discarded)
        { ManagedBytes=bytes;IncludedSamples=included;DiscardedTailSamples=discarded; }
    }

    public static class FrameAllocationWindow
    {
        public static FrameAllocationSummary ThroughFirstSampleAfterCompletion(long[] values,int countAtCompletion,int capacity,bool wrapped)
        {
            if(values==null)throw new ArgumentNullException(nameof(values));
            if(countAtCompletion<0)throw new ArgumentOutOfRangeException(nameof(countAtCompletion));
            if(capacity<2||values.Length>capacity)throw new ArgumentOutOfRangeException(nameof(capacity));
            if(wrapped||values.Length==capacity)throw new InvalidOperationException("Full or wrapped storage cannot establish a complete allocation window.");
            if(countAtCompletion>=values.Length)throw new InvalidOperationException("A counter sample after operation completion is required; missing is not zero.");
            foreach(var value in values)if(value<0)throw new InvalidOperationException("Allocation samples must be nonnegative bytes.");
            var included=countAtCompletion+1;long total=0;
            checked {for(var i=0;i<included;i++)total+=values[i];}
            return new FrameAllocationSummary(total,included,values.Length-included);
        }
    }
}
