using NUnit.Framework;

namespace EmbeddingGemma.Tests
{
    public sealed class M1MeasurementContractTests
    {
        [Test]
        public void MemorySnapshotLabelsUnsupportedCountersAndIncludesManagedHeap()
        {
            var sample = M1CompletionRunner.CaptureMemory("contract");
            Assert.That(sample.stage, Is.EqualTo("contract"));
            Assert.That(sample.managedHeapBytes, Is.GreaterThan(0));
            Assert.That(sample.processCountersAvailable,
                Is.EqualTo(sample.processWorkingSetBytes > 0 && sample.processPrivateBytes > 0));
            Assert.That(sample.graphicsCounterAvailable, Is.EqualTo(sample.graphicsDriverBytes > 0));
        }
    }
}
