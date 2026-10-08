using System.IO;
using NUnit.Framework;

namespace EmbeddingGemma.Tests
{
    public sealed class M1CompletionTests
    {
        [Test]
        public void SavedFp32AndFloat16PassBothBackendsAndRecordMeasurements()
        {
            var report = M1CompletionRunner.Run();
            Assert.That(report.success, Is.True);
            Assert.That(report.formats, Has.Count.EqualTo(2));
            foreach (var format in report.formats)
            {
                Assert.That(File.Exists(format.path), Is.True);
                Assert.That(format.bytes, Is.GreaterThan(0));
                Assert.That(format.sha256, Has.Length.EqualTo(64));
                Assert.That(format.backends, Has.Count.EqualTo(2));
                foreach (var backend in format.backends)
                {
                    Assert.That(backend.cases, Has.Count.EqualTo(15));
                    Assert.That(backend.samplesMs, Has.Count.EqualTo(45));
                    Assert.That(backend.minimumCosine, Is.GreaterThanOrEqualTo(format.precision == "fp32" ? 0.999 : 0.99));
                    Assert.That(backend.firstInferenceMs, Is.GreaterThan(0));
                    Assert.That(backend.medianMs, Is.GreaterThan(0));
                    Assert.That(backend.p95Ms, Is.GreaterThanOrEqualTo(backend.medianMs));
                    if (format.precision == "fp32") Assert.That(backend.maximumRoundTripError, Is.LessThanOrEqualTo(1e-5));
                }
            }
            Assert.That(report.formats[1].bytes, Is.LessThan(report.formats[0].bytes), "Float16 must reduce the actual stored model size.");
        }
    }
}
