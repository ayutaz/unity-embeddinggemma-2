using System;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerGpuExecutionAuditTests
    {
        static Type Required()
        {
            var type = typeof(PlayerBuildInfo).Assembly.GetType("EmbeddingGemma.Validation.PlayerGpuExecutionAudit");
            Assert.That(type, Is.Not.Null, "Real Player acceptance must inspect the live Worker's CPU assignments.");
            return type;
        }
        static JObject Evaluate(string backend, int[] computed, int[] cpu, int[] flush)
            => (JObject)Required().GetMethod("Evaluate").Invoke(null, new object[] { backend, computed, cpu, flush });
        [Test]
        public void ComputationAndIntermediateFlushAreAuditedSeparatelyFromCpuInputs()
        {
            var result = Evaluate("GPUCompute", new[] { 2, 3 }, new[] { 0 }, Array.Empty<int>());
            Assert.That((bool)result["success"], Is.True);
            Assert.That((int)result["computed_layer_output_count"], Is.EqualTo(2));
            Assert.That(result["computation_cpu_fallback_indices"].HasValues, Is.False);
            Assert.That(result["intermediate_flush_indices"].HasValues, Is.False);
        }
        [TestCase("cpu_worker")] [TestCase("computation_fallback")] [TestCase("flush")] [TestCase("no_graph")]
        public void IncompleteGpuExecutionCannotBeReportedAsSuccess(string problem)
        {
            var error = Assert.Throws<TargetInvocationException>(() => Evaluate(problem == "cpu_worker" ? "CPU" : "GPUCompute",
                problem == "no_graph" ? Array.Empty<int>() : new[] { 2, 3 },
                problem == "computation_fallback" ? new[] { 3 } : Array.Empty<int>(),
                problem == "flush" ? new[] { 3 } : Array.Empty<int>()));
            Assert.That(error.InnerException, Is.InstanceOf<InvalidOperationException>());
        }
    }
}
