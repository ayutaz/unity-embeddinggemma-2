using System;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class GpuExecutionProfileTests
    {
        [Test]
        public void TinyGpuProbeRecordsEveryPhaseWithoutCpuFallback() => Verify(false);

        [Test]
        public void OneRealSquareGpuInputRecordsLoadTransformAndInference() => Verify(true);

        static void Verify(bool real)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies()
                .Select(a => a.GetType("EmbeddingGemma.Validation.Tests.GpuExecutionProfiler"))
                .FirstOrDefault(t => t != null);
            Assert.That(type, Is.Not.Null, "Bounded GPU phase profiler is missing.");
            var report = (JObject)type.GetMethod("Run").Invoke(null, new object[] { real });
            Assert.That(report.Value<bool>("success"), Is.True);
            Assert.That(report.Value<string>("requested_backend"), Is.EqualTo("GPUCompute"));
            Assert.That(report.Value<string>("actual_backend"), Is.EqualTo("GPUCompute"));
            Assert.That(report.Value<int>("cpu_computation_or_input_count"), Is.Zero);
            Assert.That(report.Value<int>("readback_flush_count"), Is.Zero);
            Assert.That(report.Value<bool>("numeric_acceptance_claimed"), Is.False);
            var phases = (JArray)report["phases"];
            Assert.That(phases.Select(p => p.Value<string>("name")), Is.EqualTo(new[] {
                "load", "transform", "worker", "upload", "schedule", "final_readback", "dispose" }));
            foreach (var phase in phases)
            {
                Assert.That(phase.Value<bool>("completed"), Is.True);
                Assert.That(phase.Value<double>("wall_seconds"), Is.GreaterThanOrEqualTo(0));
                Assert.That(phase.Value<double>("cpu_seconds"), Is.GreaterThanOrEqualTo(0));
                Assert.That(phase.Value<long>("working_set_bytes"), Is.GreaterThan(0));
            }
            if (!real) Assert.That(report["output"].ToObject<float[]>(), Is.EqualTo(new[] { 2f, 3f }));
        }
    }
}
