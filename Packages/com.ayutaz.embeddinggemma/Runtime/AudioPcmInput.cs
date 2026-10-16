using System;

namespace EmbeddingGemma
{
    /// <summary>Owned, fixed-profile input boundary for the pinned 16 kHz mono preprocessing graph.</summary>
    public sealed class AudioPcmInput
    {
        public const int RequiredSamplingRate = 16000;
        public const int ProfileSampleCount = 179200;
        public const int ProfileFrameCount = 1119;
        public const int MinimumSampleCount = 161;
        public int SourceSampleCount { get; }
        public int ValidFrameCount { get; }
        readonly float[] waveform;
        readonly int[] sampleMask;
        public ReadOnlySpan<float> Waveform => waveform;
        public ReadOnlySpan<int> SampleMask => sampleMask;

        public AudioPcmInput(float[] samples, int samplingRate = RequiredSamplingRate, int channels = 1)
        {
            if (samples == null) throw new ArgumentNullException(nameof(samples));
            if (samplingRate != RequiredSamplingRate)
                throw new ArgumentException("The fixed audio profile requires 16000 Hz PCM.", nameof(samplingRate));
            if (channels != 1)
                throw new ArgumentException("The fixed audio profile requires mono PCM.", nameof(channels));
            if (samples.Length < MinimumSampleCount || samples.Length > ProfileSampleCount)
                throw new ArgumentException("Expected 161 to 179200 samples; input is never truncated.", nameof(samples));

            waveform = new float[ProfileSampleCount];
            Array.Copy(samples, waveform, samples.Length);
            // Validate the owned snapshot, including the last real sample.
            for (var index = 0; index < samples.Length; index++)
            {
                var value = waveform[index];
                if (float.IsNaN(value) || float.IsInfinity(value) || value < -1 || value > 1)
                    throw new ArgumentException("Expected finite PCM amplitudes in [-1,1]; input is never clipped.", nameof(samples));
            }
            sampleMask = new int[ProfileSampleCount];
            Array.Fill(sampleMask, 1, 0, samples.Length);
            SourceSampleCount = samples.Length;
            // Official semicausal framing samples the real-prefix mask at
            // source indices 160, 320, 480, ... after 160 left-padding samples.
            ValidFrameCount = (SourceSampleCount - MinimumSampleCount) / 160 + 1;
        }
    }
}
