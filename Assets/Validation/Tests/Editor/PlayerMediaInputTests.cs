using System;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerMediaInputTests
    {
        static byte[] Bytes(Array values)
        { var bytes = new byte[Buffer.ByteLength(values)]; Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length); return bytes; }
        static (byte[] pcm, byte[] mask) Signal(int count)
        {
            var pcm = new float[179200]; var mask = new int[179200];
            for (var i = 0; i < count; i++) { pcm[i] = .25f; mask[i] = 1; }
            return (Bytes(pcm), Bytes(mask));
        }
        [TestCase(179200, 1119)] [TestCase(134400, 839)]
        public void OriginalProfileAndRealPrefixArePreserved(int count, int frames)
        {
            var (pcm, mask) = Signal(count); var result = PlayerMediaInput.DecodePcm(pcm, mask, frames);
            Assert.That(result.SourceSampleCount, Is.EqualTo(count));
            Assert.That(result.ValidFrameCount, Is.EqualTo(frames));
            Assert.That(result.Waveform[count - 1], Is.EqualTo(.25f));
            Assert.That(result.SampleMask[count - 1], Is.EqualTo(1));
        }
        [TestCase("hole")] [TestCase("nonbinary")] [TestCase("tail")]
        [TestCase("short_pcm")] [TestCase("short_mask")] [TestCase("wrong_profile")]
        [TestCase("nonfinite")] [TestCase("amplitude")]
        public void CorruptPcmMaskOrProfileIsRejectedWithoutClippingOrTruncation(string problem)
        {
            var (pcm, mask) = Signal(134400);
            if (problem == "hole") Array.Copy(BitConverter.GetBytes(0), 0, mask, 40, 4);
            if (problem == "nonbinary") Array.Copy(BitConverter.GetBytes(2), 0, mask, 40, 4);
            if (problem == "tail") Array.Copy(BitConverter.GetBytes(.5f), 0, pcm, 179199 * 4, 4);
            if (problem == "short_pcm") Array.Resize(ref pcm, pcm.Length - 4);
            if (problem == "short_mask") Array.Resize(ref mask, mask.Length - 4);
            if (problem == "nonfinite") Array.Copy(BitConverter.GetBytes(float.NaN), 0, pcm, 40, 4);
            if (problem == "amplitude") Array.Copy(BitConverter.GetBytes(2f), 0, pcm, 40, 4);
            Assert.Throws<ArgumentException>(() => PlayerMediaInput.DecodePcm(pcm, mask, problem == "wrong_profile" ? 1119 : 839));
        }
    }
}
