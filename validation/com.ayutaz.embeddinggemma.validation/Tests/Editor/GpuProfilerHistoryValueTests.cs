using NUnit.Framework;
namespace EmbeddingGemma.Validation.Tests
{
    public sealed class GpuProfilerHistoryValueTests
    {
        [TestCase(1.25f,1250000L)]
        [TestCase(.001f,1000L)]
        [TestCase(.01f,10000L)]
        public void PositiveRepresentableGpuIntervalsConvertToNanoseconds(float milliseconds,long expected)
        {
            Assert.That(GpuProfilerHistory.TryGpuMillisecondsToNanoseconds(milliseconds,out var actual),Is.True);
            Assert.That(actual,Is.EqualTo(expected));
        }
        [TestCase(0f)]
        [TestCase(-1f)]
        [TestCase(float.NaN)]
        [TestCase(float.PositiveInfinity)]
        [TestCase(float.NegativeInfinity)]
        [TestCase(float.MaxValue)]
        [TestCase(float.Epsilon)]
        public void MissingOverflowAndSubNanosecondValuesCannotBecomeGpuPasses(float milliseconds)
        {
            Assert.That(GpuProfilerHistory.TryGpuMillisecondsToNanoseconds(milliseconds,out var actual),Is.False);
            Assert.That(actual,Is.Zero);
        }
    }
}
