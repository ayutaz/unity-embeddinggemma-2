using System;
using System.Linq;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Snapshot the exact generation/precision before measuring four identical-input GPU calls.</summary>
    public sealed class MediaRepeatedInputPolicy
    {
        public bool IsCandidate { get; }
        public string Precision { get; }
        public string Module { get; }
        public string ExpectedDescriptorSha256 { get; }
        public double MinimumCosine { get; }
        public BackendType Backend => BackendType.GPUCompute;
        public string[] Rounds => new[]{"first","warm","repeat-1","repeat-2"};

        MediaRepeatedInputPolicy(bool candidate,string precision,string module)
        {
            IsCandidate=candidate;Precision=candidate?precision:precision??"fp32";
            ExpectedDescriptorSha256=candidate?CandidateMediaBundlePolicy.DescriptorSha256(Precision):GpuMediaExecutionPolicy.DescriptorSha256(Precision);
            MinimumCosine=candidate?CandidateMediaBundlePolicy.MinimumCosine(Precision):GpuMediaExecutionPolicy.MinimumCosine(Precision);
            if(!new[]{"image-square","image-landscape","image-portrait","audio-1119","audio-839"}.Contains(module,StringComparer.Ordinal))
                throw new ArgumentException("Choose one of the five fixed image/audio measurement profiles.",nameof(module));
            Module=module;
        }
        public static MediaRepeatedInputPolicy CreateCandidate(string precision,string module)=>new MediaRepeatedInputPolicy(true,precision,module);
        public static MediaRepeatedInputPolicy CreateRetained(string precision,string module)=>new MediaRepeatedInputPolicy(false,precision,module);
        public void Verify(string directory)
        {
            if(IsCandidate)CandidateMediaBundlePolicy.Verify(directory,Precision);
            else GpuMediaExecutionPolicy.Verify(directory,Precision);
        }
    }
}
