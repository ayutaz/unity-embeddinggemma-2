using System;

namespace EmbeddingGemma.Validation
{
    public static class PlayerMediaInput
    {
        public static AudioPcmInput DecodePcm(byte[] pcm, byte[] mask, int expectedValidFrames)
        {
            var count = AudioPcmInput.ProfileSampleCount;
            if (!BitConverter.IsLittleEndian) throw new NotSupportedException("Pinned PCM is little-endian FP32.");
            if (pcm == null || mask == null || pcm.Length != count * 4 || mask.Length != count * 4 ||
                expectedValidFrames != 1119 && expectedValidFrames != 839)
                throw new ArgumentException("The complete fixed PCM and sample mask are required.");
            var waveform = new float[count]; var samples = new int[count];
            Buffer.BlockCopy(pcm, 0, waveform, 0, pcm.Length); Buffer.BlockCopy(mask, 0, samples, 0, mask.Length);
            var real = 0; var padding = false;
            for (var index = 0; index < count; index++)
            {
                if (samples[index] == 0) padding = true;
                else if (samples[index] != 1 || padding) throw new ArgumentException("Sample mask must be a binary real prefix.");
                else real++;
                if (padding && waveform[index] != 0) throw new ArgumentException("PCM padding must remain exactly zero.");
            }
            var input = new float[real]; Array.Copy(waveform, input, real);
            var result = new AudioPcmInput(input);
            if (result.ValidFrameCount != expectedValidFrames) throw new ArgumentException("PCM profile differs from the module.");
            return result;
        }
    }
}
