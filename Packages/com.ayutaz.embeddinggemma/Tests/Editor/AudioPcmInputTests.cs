using System;
using System.Linq;
using NUnit.Framework;

namespace EmbeddingGemma.Tests
{
    public sealed class AudioPcmInputTests
    {
        // Pinned official semicausal mask: the first frame samples index 160,
        // then index 320, 480, ... . These are explicit independent expectations.
        [TestCase(161, 1)]
        [TestCase(319, 1)]
        [TestCase(320, 1)]
        [TestCase(321, 2)]
        [TestCase(479, 2)]
        [TestCase(480, 2)]
        [TestCase(481, 3)]
        [TestCase(179199, 1119)]
        [TestCase(179200, 1119)]
        public void AcceptedPrefixMatchesOfficialFrameBoundariesAndRightZeroPadding(int count, int frames)
        {
            var source = Enumerable.Repeat(0.25f, count).ToArray();
            var input = new AudioPcmInput(source);
            Assert.That(input.SourceSampleCount, Is.EqualTo(count));
            Assert.That(input.ValidFrameCount, Is.EqualTo(frames));
            Assert.That(AudioPcmInput.ProfileFrameCount, Is.EqualTo(1119));
            var waveform = input.Waveform.ToArray();
            var mask = input.SampleMask.ToArray();
            Assert.That(waveform.Length, Is.EqualTo(179200));
            Assert.That(mask.Length, Is.EqualTo(179200));
            Assert.That(waveform.Take(count), Is.All.EqualTo(0.25f));
            Assert.That(waveform.Skip(count), Is.All.EqualTo(0));
            Assert.That(mask.Take(count), Is.All.EqualTo(1));
            Assert.That(mask.Skip(count), Is.All.EqualTo(0));
        }

        [Test]
        public void InputOwnsItsCopyAndPreservesSignedPcmEndpoints()
        {
            var samples = new float[321];
            samples[0] = -1; samples[320] = 1;
            var input = new AudioPcmInput(samples);
            samples[0] = 0; samples[320] = 0;
            Assert.That(input.Waveform.Length, Is.EqualTo(179200));
            Assert.That(input.Waveform[0], Is.EqualTo(-1));
            Assert.That(input.Waveform[320], Is.EqualTo(1));
            var exposedCopy = input.Waveform.ToArray();
            exposedCopy[0] = 0;
            Assert.That(input.Waveform[0], Is.EqualTo(-1));
            var maskCopy = input.SampleMask.ToArray();
            maskCopy[0] = 0;
            Assert.That(input.SampleMask[0], Is.EqualTo(1));
        }

        [Test]
        public void NullInputIsRejectedBeforePreparation() =>
            Assert.That(() => new AudioPcmInput(null), Throws.ArgumentNullException);

        [TestCase(0)]
        [TestCase(1)]
        [TestCase(160)]
        [TestCase(179201)]
        public void UnsupportedLengthIsRejectedWithoutTruncation(int count) =>
            Assert.That(() => new AudioPcmInput(new float[count]), Throws.ArgumentException);

        [TestCase(0, 1)]
        [TestCase(8000, 1)]
        [TestCase(48000, 1)]
        [TestCase(16000, 0)]
        [TestCase(16000, 2)]
        public void UnsupportedRateOrChannelLayoutIsRejectedExplicitly(int rate, int channels) =>
            Assert.That(() => new AudioPcmInput(new float[321], rate, channels), Throws.ArgumentException);

        [TestCase(float.NaN)]
        [TestCase(float.PositiveInfinity)]
        [TestCase(float.NegativeInfinity)]
        [TestCase(1.001f)]
        [TestCase(-1.001f)]
        public void NonfiniteOrOutOfRangePcmIsRejectedWithoutClipping(float value)
        {
            var source = new float[321];
            source[320] = value;
            Assert.That(() => new AudioPcmInput(source), Throws.ArgumentException);
        }
    }
}
